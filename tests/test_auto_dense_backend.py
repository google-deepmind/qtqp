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

The sparse backends factorize the (n+m) augmented KKT system; SCIPY_DENSE
eliminates y and factorizes an n x n Gram instead. AUTO used to pick on
platform and availability only, so a constraint matrix with no zeros in it
still got a sparse factorization -- measured at up to 17x slower than
SCIPY_DENSE on the same problem.

What these tests pin is the decision, not the timings: the shapes where the
dense backend is at or below the sparse flop count for any ordering, the ones
where it is not, and the case where it must decline even on dense data.
"""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _dense_psd(n, rng):
  """Return a fully dense positive definite matrix."""
  factor = rng.standard_normal((n, n)) / np.sqrt(n)
  return sparse.csc_matrix(factor @ factor.T + 0.05 * np.eye(n))


def _dense(m, n, rng):
  """Return a fully dense m-by-n matrix."""
  return sparse.csc_matrix(rng.standard_normal((m, n)))


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


def _prefers_dense(p, a, z=0, reg=1e-8):
  return qtqp._auto_prefers_dense(p, a, z, reg)  # pylint: disable=protected-access


@pytest.mark.parametrize("m_over_n", [0.25, 1.0, 2.0])
def test_both_blocks_dense_prefers_dense(m_over_n):
  """The reported case: no zeros in either block.

  Eliminating y first is the sparse solver's best ordering here and costs
  `m n^2 + n^3/3`, exactly what the Gram costs, so the switch is safe at every
  aspect ratio -- including the tall-x shapes where a dense `a` alone would
  not be enough.
  """
  rng = np.random.default_rng(0)
  n = 40
  m = int(m_over_n * n)
  assert _prefers_dense(_dense_psd(n, rng), _dense(m, n, rng))


@pytest.mark.parametrize("m_over_n", [1.0, 2.0])
def test_dense_a_alone_prefers_dense_from_m_equals_n(m_over_n):
  """A dense `a` reaches parity at `m == n` whatever `p` looks like.

  The all-zero `p` is the LP case, where `p` contributes no density at all and
  only `a` can trigger the switch.
  """
  rng = np.random.default_rng(1)
  n = 40
  a = _dense(int(m_over_n * n), n, rng)
  for p in (_banded_psd(n), sparse.csc_matrix((n, n))):
    assert _prefers_dense(p, a)


def test_dense_a_with_fewer_rows_than_columns_stays_sparse():
  """Below `m == n` the sparse solver forms only an m x m clique.

  It eliminates the x nodes first for `n m^2 + m^3/3` against the Gram's
  `n^3/3 + m n^2`, a factor of about `(n/m)^2`. At n=3000, m=10 and `p = I`
  the data is fully dense and the dense backend would still be hopeless: a
  3000x3000 Cholesky per iteration on a problem the sparse path solves in
  milliseconds.
  """
  rng = np.random.default_rng(2)
  n = 40
  assert not _prefers_dense(_banded_psd(n), _dense(n // 4, n, rng))
  assert not _prefers_dense(sparse.eye(n, format="csc"), _dense(1, n, rng))


def test_dense_p_alone_stays_sparse():
  """A dense `p` with a sparse `a` is not safe while the backend densifies A.

  `ScipyDenseSolver.factorize` runs dsyrk through `a`'s zeros, paying
  `n^3/3 + m n^2` against the sparse solver's `n^3/3` plus fill-limited work
  on `a`. Wins measured there are sparse-solver constants, not a flop
  argument, so the rule leaves this case alone until the backend forms
  `A' D^-1 A` as a sparse product.
  """
  rng = np.random.default_rng(3)
  n = 40
  p = _dense_psd(n, rng)
  assert not _prefers_dense(p, _box(n))
  assert not _prefers_dense(p, sparse.eye(n, format="csc"))


def test_sparse_data_stays_sparse():
  """Band structure is what the sparse backends exist for."""
  n = 200
  assert not _prefers_dense(_banded_psd(n), _box(n))


def test_partially_filled_blocks_stay_sparse():
  """Blocks with zeros left in them keep the sparse path.

  A row of `a` with `d n` nonzeros costs the sparse solver a `(d n)^2` clique
  update against the dense backend's full `n^2`, so the parity argument holds
  only next to `d = 1`. Half-filled blocks are well inside the band where
  structure, not density, decides.
  """
  rng = np.random.default_rng(4)
  n = 40
  half_full = sparse.csc_matrix(
      np.where(rng.random((2 * n, n)) < 0.5, rng.standard_normal((2 * n, n)), 0.0)
  )
  assert not _prefers_dense(_dense_psd(n, rng), half_full)


def test_zero_regularization_with_equalities_stays_sparse():
  """AUTO must not turn a working solve into a ValueError.

  Dense Gram elimination cannot invert a zero equality diagonal and raises;
  the sparse backends solve the same problem. So however dense the data, AUTO
  has to decline the switch when there are equality rows and no static
  regularization to shift them.
  """
  rng = np.random.default_rng(5)
  n = 40
  p, a = _dense_psd(n, rng), _dense(2 * n, n, rng)

  assert not _prefers_dense(p, a, z=1, reg=0.0)
  # Without equality rows there is no zero diagonal to invert, so the same
  # data at the same regularization is fine.
  assert _prefers_dense(p, a, z=0, reg=0.0)
  # And with regularization the equality rows are shifted, so it is fine too.
  assert _prefers_dense(p, a, z=1, reg=1e-8)


def test_empty_problem_stays_sparse():
  """An empty block satisfies the nnz test vacuously; route nothing."""
  empty = sparse.csc_matrix((0, 0))
  assert not _prefers_dense(empty, empty)
  assert not _prefers_dense(sparse.csc_matrix((4, 4)), sparse.csc_matrix((0, 4)))


def _record_resolved(monkeypatch):
  """Capture the enum `_solve_impl` hands to the resolver."""
  seen = []
  resolve = qtqp._resolve_linear_solver  # pylint: disable=protected-access

  def recording(linear_solver):
    seen.append(linear_solver)
    return resolve(linear_solver)

  monkeypatch.setattr(qtqp, "_resolve_linear_solver", recording)
  return seen


def test_auto_routes_dense_data_to_scipy_dense(monkeypatch):
  """The redirect happens before the resolver, so it shows in the header."""
  rng = np.random.default_rng(6)
  n = 20
  rows = rng.standard_normal((2 * n, n))
  seen = _record_resolved(monkeypatch)
  qtqp.QTQP(
      p=_dense_psd(n, rng),
      a=sparse.csc_matrix(rows),
      b=rows @ (rng.standard_normal(n) / np.sqrt(n)) + 1.0,
      c=rng.standard_normal(n),
      z=0,
  ).solve(verbose=False)
  assert seen == [qtqp.LinearSolver.SCIPY_DENSE]


def test_named_backend_is_left_alone(monkeypatch):
  """A caller who names a backend gets it, dense data or not."""
  rng = np.random.default_rng(7)
  n = 20
  rows = rng.standard_normal((2 * n, n))
  seen = _record_resolved(monkeypatch)
  qtqp.QTQP(
      p=_dense_psd(n, rng),
      a=sparse.csc_matrix(rows),
      b=rows @ (rng.standard_normal(n) / np.sqrt(n)) + 1.0,
      c=rng.standard_normal(n),
      z=0,
  ).solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)
  assert seen == [qtqp.LinearSolver.SCIPY]


def _problem(case, rng, n):
  """Return a feasible (p, a, b, c, z) for one routing case."""
  c = rng.standard_normal(n)
  if case in ("dense_p_dense_a", "dense_p_dense_a_equalities"):
    z = 3 if case.endswith("equalities") else 0
    p = _dense_psd(n, rng)
    rows = rng.standard_normal((2 * n, n))
    x0 = rng.standard_normal(n) / np.sqrt(n)
    b = rows @ x0
    b[z:] += rng.uniform(0.05, 0.5, 2 * n - z)
    return p, sparse.csc_matrix(rows), b, c, z
  if case == "dense_a_sparse_p":
    rows = rng.standard_normal((2 * n, n))
    x0 = rng.standard_normal(n) / np.sqrt(n)
    b = rows @ x0 + rng.uniform(0.05, 0.5, 2 * n)
    return _banded_psd(n), sparse.csc_matrix(rows), b, c, 0
  if case == "dense_p_sparse_a":
    return _dense_psd(n, rng), _box(n), np.full(2 * n, 1.0), c, 0
  return _banded_psd(n), _box(n), np.full(2 * n, 1.0), c, 0


@pytest.mark.parametrize(
    "case",
    [
        "dense_p_dense_a",
        "dense_p_dense_a_equalities",
        "dense_a_sparse_p",
        "dense_p_sparse_a",
        "banded",
    ],
)
def test_auto_still_solves_whatever_it_picks(case):
  """Routing is only worth anything if the resolved backend solves.

  Each case is feasible by construction, so a status other than SOLVED means
  the backend choice broke the solve rather than that the problem was hard.
  """
  rng = np.random.default_rng(8)
  p, a, b, c, z = _problem(case, rng, n=60)

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
