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

"""Stored sparse zeros must not change LP initialization or backend selection."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _store_all_entries(matrix):
  rows, cols = matrix.shape
  return sparse.csc_matrix(
      (matrix.T.ravel(), np.tile(np.arange(rows), cols),
       np.arange(cols + 1) * rows), shape=matrix.shape
  )


@pytest.mark.parametrize("encoding", ["stored", "cancelled"])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_zero_quadratic_data_uses_the_lp_path(encoding, dtype):
  a = sparse.csc_matrix([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
  b, c = np.array([1., 2., 0., 0.]), np.array([-1., -1.])
  if encoding == "stored":
    p = _store_all_entries(np.zeros((2, 2), dtype=dtype))
  else:
    p = sparse.csc_matrix(
        (np.array([1, -1, 2, -2], dtype=dtype),
         np.array([0, 0, 1, 1]), np.array([0, 2, 4])), shape=(2, 2)
    )
  original = (p.data.copy(), p.indices.copy(), p.indptr.copy())
  problem = qtqp.QTQP(a=a, b=b, c=c, p=p, z=0)
  reference = qtqp.QTQP(a=a, b=b, c=c, z=0)
  settings = dict(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY,
                  collect_stats=True)
  expected = reference.solve(**settings)
  actual = problem.solve(**settings)
  assert expected.status is qtqp.SolutionStatus.SOLVED
  assert actual.status is expected.status
  assert actual.iterations == expected.iterations
  for field in ("x", "y", "s"):
    np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
  assert problem.p.nnz == 0
  for before, after in zip(original, (p.data, p.indices, p.indptr)):
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("padded", ["a", "p", "both"])
def test_auto_density_ignores_explicit_zeros(padded):
  # A has a square identity sparsity pattern; the dense Gram route is needless.
  n = 10
  a, p = sparse.eye(n, format="csc"), sparse.eye(n, format="csc")
  if padded in ("a", "both"):
    a = _store_all_entries(a.toarray())
  if padded in ("p", "both"):
    p = _store_all_entries(p.toarray())
  original_a, original_p = a.data.copy(), p.data.copy()
  problem = qtqp.QTQP(a=a, b=np.ones(n), c=-np.ones(n), p=p, z=0)
  assert not qtqp._auto_prefers_dense(problem.p, problem.a, 0, 1e-8)
  assert problem.a.nnz == n
  assert problem.p.nnz == n
  np.testing.assert_array_equal(a.data, original_a)
  np.testing.assert_array_equal(p.data, original_p)


def test_nonzero_quadratic_values_survive_zero_removal():
  dense = np.array([[2., 0., -0.5], [0., 3., 0.], [-0.5, 0., 4.]])
  p = _store_all_entries(dense)
  problem = qtqp.QTQP(a=sparse.eye(3, format="csc"), b=np.ones(3),
                      c=np.zeros(3), p=p, z=0)
  np.testing.assert_array_equal(problem.p.toarray(), dense)
  assert problem.p.nnz == np.count_nonzero(dense)
  assert p.nnz == 9
