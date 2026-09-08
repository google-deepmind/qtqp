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

"""AUTO must choose its backend from the data, not from the platform alone.

The sparse backends factorize the (n+m) KKT system; SCIPY_DENSE eliminates y
and factorizes an n x n Gram instead. AUTO used to pick on platform and
availability only, so a constraint matrix with no zeros in it still got a
sparse factorization -- measured at up to 17x slower than SCIPY_DENSE on the
same problem.

What these tests pin is the decision, not the timings: which backend AUTO
resolves to, and the three cases where it must decline to switch even though
the data looks dense.
"""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _dense_psd(n, rng):
  """Return a fully dense positive definite matrix."""
  factor = rng.standard_normal((n, n)) / np.sqrt(n)
  return sparse.csc_matrix(factor @ factor.T + 0.05 * np.eye(n))


def _banded_psd(n, half_bandwidth=1):
  """Return a positive definite band matrix: sparse, and cheap to factorize."""
  offsets, diagonals = [0], [np.full(n, 2.0 * half_bandwidth + 1)]
  for k in range(1, half_bandwidth + 1):
    offsets += [k, -k]
    diagonals += [-0.4 * np.ones(n - k)] * 2
  return sparse.diags(diagonals, offsets, format="csc")


def _box(n):
  """Return the constraint matrix of a box: two nonzeros per column."""
  return sparse.vstack([sparse.eye(n), -sparse.eye(n)], format="csc")


def test_dense_a_and_p_prefers_dense():
  """The reported case: no zeros in either block."""
  rng = np.random.default_rng(0)
  n = 40
  assert qtqp._auto_prefers_dense(  # pylint: disable=protected-access
      _dense_psd(n, rng),
      sparse.csc_matrix(rng.standard_normal((2 * n, n))),
      0,
      1e-8,
  )


def test_dense_p_alone_prefers_dense():
  """A dense P makes the Gram's `H` term dense whatever `a` looks like."""
  rng = np.random.default_rng(1)
  n = 40
  assert qtqp._auto_prefers_dense(  # pylint: disable=protected-access
      _dense_psd(n, rng), _box(n), 0, 1e-8
  )


def test_dense_a_alone_prefers_dense():
  """A dense `a` makes the Gram's `A' D^-1 A` term dense whatever P looks like.

  Covers the LP case too, where `p` is the all-zero matrix and so contributes
  no density at all.
  """
  rng = np.random.default_rng(2)
  n = 40
  dense_a = sparse.csc_matrix(rng.standard_normal((2 * n, n)))
  for p in (_banded_psd(n), sparse.csc_matrix((n, n))):
    assert qtqp._auto_prefers_dense(p, dense_a, 0, 1e-8)  # pylint: disable=protected-access


def test_sparse_data_stays_sparse():
  """Band structure is what the sparse backends exist for."""
  n = 200
  assert not qtqp._auto_prefers_dense(  # pylint: disable=protected-access
      _banded_psd(n), _box(n), 0, 1e-8
  )


def test_single_dense_row_stays_sparse():
  """One dense row must not speak for the whole constraint matrix.

  A budget row over an otherwise banded problem is the case a clique bound on
  fill -- the sum of squared row counts -- gets wrong: it reads as fully dense
  while the sparse backend still wins the problem, measured at 1.7x for
  n = 3000. Row density averaged over `a` is what keeps this on the sparse
  path.
  """
  n = 500
  a = sparse.vstack(
      [sparse.csc_matrix(np.ones((1, n))), sparse.eye(n), -sparse.eye(n)],
      format="csc",
  )
  assert not qtqp._auto_prefers_dense(_banded_psd(n), a, 1, 1e-8)  # pylint: disable=protected-access


def test_zero_regularization_with_equalities_stays_sparse():
  """AUTO must not turn a working solve into a ValueError.

  Dense Gram elimination cannot invert a zero equality diagonal and raises;
  the sparse backends solve the same problem. So however dense the data, AUTO
  has to decline the switch when there are equality rows and no static
  regularization to shift them.
  """
  rng = np.random.default_rng(3)
  n = 40
  p = _dense_psd(n, rng)
  a = sparse.csc_matrix(rng.standard_normal((2 * n, n)))

  assert not qtqp._auto_prefers_dense(p, a, 1, 0.0)  # pylint: disable=protected-access
  # Without equality rows there is no zero diagonal to invert, so the same
  # data at the same regularization is fine.
  assert qtqp._auto_prefers_dense(p, a, 0, 0.0)  # pylint: disable=protected-access
  # And with regularization the equality rows are shifted, so it is fine too.
  assert qtqp._auto_prefers_dense(p, a, 1, 1e-8)  # pylint: disable=protected-access


def test_dense_but_too_large_stays_sparse(monkeypatch):
  """Past the memory cap AUTO keeps the sparse path rather than allocate.

  The cap is lowered here rather than building a problem big enough to trip
  the real one, which at 256MB would make the test cost more than the rest of
  the suite.
  """
  rng = np.random.default_rng(4)
  n = 40
  p = _dense_psd(n, rng)
  a = sparse.csc_matrix(rng.standard_normal((2 * n, n)))
  assert qtqp._auto_prefers_dense(p, a, 0, 1e-8)  # pylint: disable=protected-access

  # 3n^2 + 2mn entries for this problem; set the cap just under it.
  entries = 3 * n * n + 2 * (2 * n) * n
  monkeypatch.setattr(qtqp, "_AUTO_DENSE_MAX_ENTRIES", entries - 1)
  assert not qtqp._auto_prefers_dense(p, a, 0, 1e-8)  # pylint: disable=protected-access


def test_empty_problem_stays_sparse():
  """A degenerate shape must not divide by zero on the density test."""
  assert not qtqp._auto_prefers_dense(  # pylint: disable=protected-access
      sparse.csc_matrix((0, 0)), sparse.csc_matrix((0, 0)), 0, 1e-8
  )


def test_prefer_dense_resolves_to_scipy_dense():
  """The flag reaches the resolver, and does not poison the platform cache."""
  before = dict(qtqp._AUTO_SOLVER_CACHE)  # pylint: disable=protected-access
  resolved, backend = qtqp._resolve_linear_solver(  # pylint: disable=protected-access
      qtqp.LinearSolver.AUTO, prefer_dense=True
  )
  backend.free()
  assert resolved is qtqp.LinearSolver.SCIPY_DENSE
  # The cache records which *sparse* backend imported; a per-problem dense
  # choice has no business in a platform-keyed cache.
  assert qtqp._AUTO_SOLVER_CACHE == before  # pylint: disable=protected-access


def test_explicit_backend_overrides_the_density_choice():
  """A caller who names a backend gets it, dense data or not."""
  resolved, backend = qtqp._resolve_linear_solver(  # pylint: disable=protected-access
      qtqp.LinearSolver.SCIPY, prefer_dense=True
  )
  backend.free()
  assert resolved is qtqp.LinearSolver.SCIPY


@pytest.mark.parametrize(
    "case", ["dense_p_dense_a", "dense_p_sparse_a", "banded", "single_dense_row"]
)
def test_auto_still_solves_whatever_it_picks(case):
  """Routing is only worth anything if the resolved backend solves.

  Each case is feasible by construction, so a status other than SOLVED means
  the backend choice broke the solve rather than that the problem was hard.
  """
  rng = np.random.default_rng(6)
  n = 60

  if case == "dense_p_dense_a":
    p = _dense_psd(n, rng)
    rows = rng.standard_normal((2 * n, n))
    a = sparse.csc_matrix(rows)
    x0 = rng.standard_normal(n) / np.sqrt(n)
    b = rows @ x0 + rng.uniform(0.05, 0.5, 2 * n)
    z = 0
  elif case == "dense_p_sparse_a":
    p = _dense_psd(n, rng)
    a = _box(n)
    b = np.full(2 * n, 1.0)
    z = 0
  elif case == "banded":
    p = _banded_psd(n)
    a = _box(n)
    b = np.full(2 * n, 1.0)
    z = 0
  else:
    p = _banded_psd(n)
    a = sparse.vstack(
        [sparse.csc_matrix(np.ones((1, n))), sparse.eye(n), -sparse.eye(n)],
        format="csc",
    )
    b = np.concatenate([[1.0], np.full(n, 2.0 / n), np.zeros(n)])
    z = 1

  c = rng.standard_normal(n)
  solution = qtqp.QTQP(p=p, a=a, b=b, c=c, z=z).solve(verbose=False)
  assert solution.status == qtqp.SolutionStatus.SOLVED

  # And the answer agrees with the backend the caller could have named.
  explicit = qtqp.QTQP(p=p, a=a, b=b, c=c, z=z).solve(
      verbose=False, linear_solver=qtqp.LinearSolver.SCIPY
  )
  assert explicit.status == qtqp.SolutionStatus.SOLVED
  objective = lambda x: 0.5 * x @ (p @ x) + c @ x
  np.testing.assert_allclose(
      objective(solution.x), objective(explicit.x), rtol=1e-6, atol=1e-8
  )
