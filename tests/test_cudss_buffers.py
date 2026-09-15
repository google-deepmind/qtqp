# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Real-device checks for cuDSS's persistent buffers and sparse views."""

import numpy as np
import pytest
from scipy import sparse

from qtqp import solvers_gpu


@pytest.fixture
def cudss_backend():
  cp = pytest.importorskip("cupy")
  pytest.importorskip("nvmath")
  try:
    devices = cp.cuda.runtime.getDeviceCount()
  except cp.cuda.runtime.CUDARuntimeError as error:
    pytest.skip(f"CUDA device unavailable: {error}")
  if not devices:
    pytest.skip("No CUDA device available")
  backend = solvers_gpu.CuDssSolver()
  try:
    yield backend
  finally:
    backend.free()


def test_cudss_reuses_buffers_and_transpose(monkeypatch, cudss_backend):
  backend = cudss_backend
  triangle = sparse.coo_matrix([
      [4.0, 1.0, 2.0], [0.0, 5.0, -1.0], [0.0, 0.0, -3.0]
  ]).asformat(backend.format())
  transpose_calls = 0
  matrix_type = backend._cp_sparse.csr_matrix
  original_transpose = matrix_type.transpose

  def transpose(matrix, *args, **kwargs):
    nonlocal transpose_calls
    if matrix is backend._kkt_gpu:
      transpose_calls += 1
    return original_transpose(matrix, *args, **kwargs)

  monkeypatch.setattr(matrix_type, "transpose", transpose)
  backend.set_kkt(triangle)
  assert transpose_calls == 1
  data_pointer = backend._kkt_gpu.data.data.ptr
  diagonal_pointer = backend._kkt_diag_gpu.data.ptr
  assert backend._kkt_gpu_t.data.data.ptr == data_pointer

  expected = np.array([0.25, -0.5, 2.0])
  solver = None
  for diagonal in ([4.0, 5.0, -3.0], [6.0, 7.0, -2.0], [4.0, 5.0, -3.0]):
    diagonal = np.array(diagonal)

    def unexpected_copy(*args, **kwargs):
      pytest.fail("Diagonal updates must reuse the existing GPU buffer")

    # Neither allocating a temporary upload with asarray nor copying it
    # into the persistent buffer is needed. Scope this to update_diag:
    # CuPy/nvmath may legitimately use these primitives elsewhere.
    with monkeypatch.context() as update_patch:
      update_patch.setattr(backend._cp, "asarray", unexpected_copy)
      update_patch.setattr(backend._cp, "copyto", unexpected_copy)
      backend.update_diag(diagonal)

    assert backend._kkt_gpu.data.data.ptr == data_pointer
    assert backend._kkt_diag_gpu.data.ptr == diagonal_pointer
    matrix = triangle.toarray()
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, diagonal)
    rhs = matrix @ expected
    backend.factorize()
    if solver is None:
      solver = backend._solver
    assert backend._solver is solver
    np.testing.assert_allclose(backend @ expected, rhs, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        backend.solve(rhs), expected, rtol=1e-11, atol=1e-11
    )
    assert transpose_calls == 1
