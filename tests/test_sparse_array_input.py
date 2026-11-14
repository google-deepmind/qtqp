# Copyright 2026 Google LLC
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

"""SciPy's CSC sparse arrays are CSC data and must be accepted as such."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _readme_problem():
  p = sparse.csc_matrix([[3.0, -1.0], [-1.0, 2.0]])
  a = sparse.csc_matrix([[-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
  return a, np.array([-1.0, 0.3, -0.5]), np.array([-1.0, -1.0]), p


@pytest.mark.parametrize("as_array", ["a", "p", "both"])
def test_csc_array_matches_csc_matrix(as_array):
  a, b, c, p = _readme_problem()
  expected = qtqp.QTQP(a=a, b=b, c=c, z=1, p=p).solve(verbose=False)
  if as_array in ("a", "both"):
    a = sparse.csc_array(a)
  if as_array in ("p", "both"):
    p = sparse.csc_array(p)
  actual = qtqp.QTQP(a=a, b=b, c=c, z=1, p=p).solve(verbose=False)
  assert expected.status is qtqp.SolutionStatus.SOLVED
  assert actual.status is expected.status
  np.testing.assert_allclose(actual.x, [0.3, -0.7], atol=1e-6)
  for field in ("x", "y", "s"):
    np.testing.assert_array_equal(getattr(actual, field),
                                  getattr(expected, field))


def test_csc_array_input_is_not_mutated():
  # A stored zero and a duplicate entry are both rewritten during
  # canonicalization; that must happen on a copy.
  data = np.array([1.0, 0.0, 2.0, -1.0, 1.0])
  indices = np.array([0, 1, 0, 1, 1])
  indptr = np.array([0, 2, 5])
  a = sparse.csc_array((data, indices, indptr), shape=(2, 2))
  # Symmetric [[2, 0], [0, 3]] with a split diagonal entry and two stored
  # zeros.
  p = sparse.csc_array(
      (np.array([1.0, 0.0, 1.0, 0.0, 3.0]), np.array([0, 1, 0, 0, 1]),
       np.array([0, 3, 5])), shape=(2, 2))
  originals = [(m.data.copy(), m.indices.copy(), m.indptr.copy())
               for m in (a, p)]
  problem = qtqp.QTQP(a=a, b=np.ones(2), c=np.zeros(2), z=0, p=p)
  np.testing.assert_array_equal(problem.a.toarray(), a.toarray())
  np.testing.assert_array_equal(problem.p.toarray(), p.toarray())
  for matrix, original in zip((a, p), originals):
    for before, after in zip(original,
                             (matrix.data, matrix.indices, matrix.indptr)):
      np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("which", ["a", "p"])
def test_non_csc_sparse_arrays_are_still_rejected(which):
  a, b, c, p = _readme_problem()
  if which == "a":
    a = sparse.csr_array(a)
  else:
    p = sparse.csr_array(p)
  with pytest.raises(TypeError, match="CSC format"):
    qtqp.QTQP(a=a, b=b, c=c, z=1, p=p)
