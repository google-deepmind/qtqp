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
"""KKT mirroring must not double diagonal values before removing them."""

import numpy as np
import pytest
from scipy import sparse

from qtqp import direct, solvers_dense, solvers_gpu, solvers_sparse


@pytest.mark.parametrize("matrix_format", ["csc", "csr"])
def test_mirror_preserves_large_diagonals(matrix_format):
  upper = sparse.csc_matrix([
      [1e308, 3., 2.],
      [0., -1e308, 4.],
      [0., 0., 1.],
  ]).asformat(matrix_format)
  expected = np.array([
      [1e308, 3., 2.],
      [3., -1e308, 4.],
      [2., 4., 1.],
  ])
  mirrored = solvers_sparse._full_symmetric_from_upper(upper, matrix_format)
  assert mirrored.format == matrix_format
  np.testing.assert_array_equal(mirrored.toarray(), expected)
  np.testing.assert_array_equal(upper.diagonal(), expected.diagonal())


def test_mirror_keeps_explicit_zero_diagonal_for_updates():
  upper = sparse.csc_matrix([[1., 2.], [0., -1.]])
  upper.setdiag([0., -1.])
  full = solvers_sparse._full_symmetric_from_upper(upper, "csc")
  indices = direct.diag_data_indices(full)
  full.data[indices] = [3., -4.]
  np.testing.assert_array_equal(full.toarray(), [[3., 2.], [2., -4.]])


@pytest.mark.parametrize("backend_name", ["scipy", "cupy"])
def test_dense_setup_does_not_double_unused_diagonal(backend_name):
  # The diagonal is stored separately. Mirroring only the off-diagonal
  # block must work even when twice a finite diagonal would overflow.
  kkt = sparse.csr_matrix([
      [1e308, 2., 3.],
      [0., 1e308, 4.],
      [0., 0., -1.],
  ])
  if backend_name == "scipy":
    backend = solvers_dense.ScipyDenseSolver()
  else:
    # The host-side block construction is identical without a CUDA device.
    backend = solvers_gpu.CupyDenseSolver.__new__(solvers_gpu.CupyDenseSolver)
    backend._cp = np
  backend.set_dims(n=2, m=1, z=0)
  with np.errstate(over="raise", invalid="raise"):
    backend.set_kkt(kkt)
  offdiag = (backend._P_offdiag if backend_name == "scipy"
             else backend._P_offdiag_gpu)
  np.testing.assert_array_equal(offdiag, [[0., 2.], [2., 0.]])
  np.testing.assert_array_equal(kkt.diagonal(), [1e308, 1e308, -1.])
