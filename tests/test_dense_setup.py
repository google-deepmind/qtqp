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
"""Dense backend setup must preserve the Gram reduction's memory bound."""

import numpy as np
import pytest
from scipy import sparse

from qtqp import direct


@pytest.mark.parametrize("backend_name", ["scipy", "cupy"])
@pytest.mark.parametrize("sparse_format", ["csr", "csc"])
def test_dense_setup_only_densifies_required_blocks(
    monkeypatch, backend_name, sparse_format
):
  n, m = 4, 31
  rng = np.random.default_rng(921)
  a = rng.normal(size=(m, n))
  p_factor = rng.normal(size=(n, n))
  p = p_factor.T @ p_factor + np.eye(n)
  kkt = sparse.bmat(
      [[sparse.csc_matrix(np.triu(p)), sparse.csc_matrix(a.T)],
       [None, -sparse.eye(m)]],
      format=sparse_format,
  )
  dense_shapes = []

  def track_toarray(original):
    def checked(matrix, *args, **kwargs):
      # Check before allocating, so the regression remains cheap even if
      # somebody restores the full KKT conversion.
      assert matrix.shape in ((m, n), (n, n))
      dense_shapes.append(matrix.shape)
      return original(matrix, *args, **kwargs)
    return checked

  for matrix_type in (sparse.csr_matrix, sparse.csc_matrix):
    monkeypatch.setattr(
        matrix_type, "toarray", track_toarray(matrix_type.toarray)
    )

  if backend_name == "scipy":
    backend = direct.ScipyDenseSolver()
  else:
    # Setup only uses allocation/transfer primitives. NumPy stands in for
    # those so the host-memory regression is covered without a CUDA device.
    backend = direct.CupyDenseSolver.__new__(direct.CupyDenseSolver)
    backend._cp = np
  backend.set_dims(n=n, m=m, z=0)
  backend.set_kkt(kkt)

  assert dense_shapes == [(m, n), (n, n)]
  p_offdiag = p.copy()
  np.fill_diagonal(p_offdiag, 0.0)
  if backend_name == "cupy":
    np.testing.assert_array_equal(backend._A_gpu, a)
    np.testing.assert_array_equal(backend._P_offdiag_gpu, p_offdiag)
  else:
    np.testing.assert_array_equal(backend._A, a)
    np.testing.assert_array_equal(backend._P_offdiag, p_offdiag)
    assert backend._A.flags.f_contiguous
    assert backend._P_offdiag.flags.f_contiguous

    # Validate the factorization and block matvec independently against
    # the original saddle-point system, including nonzero P off-diagonals.
    backend.update_diag(np.concatenate([p.diagonal(), -np.ones(m)]))
    backend.factorize()
    expected = rng.normal(size=n + m)
    rhs = np.concatenate([
        p @ expected[:n] + a.T @ expected[n:],
        a @ expected[:n] - expected[n:],
    ])
    np.testing.assert_allclose(backend @ expected, rhs, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        backend.solve(rhs), expected, rtol=1e-11, atol=1e-11
    )
