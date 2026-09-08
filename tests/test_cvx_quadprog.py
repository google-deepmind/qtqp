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

"""Cross-check QTQP against cvx-quadprog, an independent QP implementation.

cvx-quadprog is a NumPy/SciPy implementation of the Goldfarb/Idnani dual
active-set method. It shares no code and no algorithm with QTQP's interior
point method, so agreement on a problem both accept is evidence about the
answer rather than about a shared implementation.

The two use different conventions. QTQP solves

    min. (1/2) x.T @ p @ x + c.T @ x
    s.t. a @ x + s = b, s[:z] == 0, s[z:] >= 0

that is, `a[:z] @ x == b[:z]` and `a[z:] @ x <= b[z:]`, while `solve_qp`
solves

    min. (1/2) x.T @ G @ x - a_gi.T @ x
    s.t. C.T @ x >= b_gi, first `meq` rows held as equalities.

so the translation is `G = p`, `a_gi = -c`, `C = -a.T`, `b_gi = -b`,
`meq = z`. Flipping the sign of `C` and `b_gi` together -- rather than
negating the inequality rows of `a` -- is what leaves the two solvers'
multipliers directly comparable, with no sign correction on the duals.

Both matrices reach `solve_qp` dense, which is what caps the sizes here:
these are correctness cross-checks, not a scaling benchmark.
"""

import numpy as np
import pytest
from scipy import sparse

import qtqp

quadprog = pytest.importorskip(
    "cvx.quadprog",
    reason="cvx-quadprog not installed (pip install 'qtqp[test]')",
)


def _feasible_qp(seed, n, m_eq, m_in):
  """Return a strictly convex QP that is feasible by construction.

  The right-hand side is built from an interior point `x0`, so feasibility
  does not depend on the draw: equality rows pass through `x0` exactly and
  inequality rows leave it slack.

  Args:
    seed: Seed for the random draw.
    n: Number of variables.
    m_eq: Number of equality rows.
    m_in: Number of inequality rows.

  Returns:
    The tuple `(p, a, b, c, z)` in QTQP's convention.
  """
  rng = np.random.default_rng(seed)
  factor = rng.standard_normal((n, n))
  # + n * I keeps p comfortably positive definite: Goldfarb/Idnani is a
  # strictly convex method and rejects anything else (see the tests below).
  p = factor @ factor.T + n * np.eye(n)
  c = rng.standard_normal(n)
  x0 = rng.standard_normal(n)
  a_eq = rng.standard_normal((m_eq, n))
  a_in = rng.standard_normal((m_in, n))
  a = np.vstack([a_eq, a_in])
  b = np.concatenate(
      [a_eq @ x0, a_in @ x0 + rng.uniform(0.1, 2.0, m_in)]
  )
  return p, a, b, c, m_eq


def _capped_portfolio_qp(seed, n, factors, cap):
  """Return a long-only, position-capped minimum-variance QP.

  Its active set is deliberately degenerate: many weights sit exactly on the
  cap, which pins the objective while leaving `x` weakly determined. That is
  the case where the two solvers' iterates are furthest apart, so it belongs
  in a cross-check.

  Args:
    seed: Seed for the random draw.
    n: Number of assets.
    factors: Number of factors behind the covariance.
    cap: Upper bound on each weight.

  Returns:
    The tuple `(p, a, b, c, z)` in QTQP's convention.
  """
  rng = np.random.default_rng(seed)
  loadings = rng.standard_normal((n, factors))
  p = loadings @ loadings.T + 0.5 * np.eye(n)
  c = -rng.uniform(0.0, 0.1, n)
  # sum(x) == 1, then x <= cap and -x <= 0.
  a = np.vstack([np.ones((1, n)), np.eye(n), -np.eye(n)])
  b = np.concatenate([[1.0], np.full(n, cap), np.zeros(n)])
  return p, a, b, c, 1


# The example from the README, in turn taken from SCS's basic_qp.
_README_QP = (
    np.array([[3.0, -1.0], [-1.0, 2.0]]),
    np.array([[-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    np.array([-1.0, 0.3, -0.5]),
    np.array([-1.0, -1.0]),
    1,
)

# (name, problem, atol on x). The tolerance on `x` is per problem because it
# is set by how well the problem determines `x`, not by either solver: the
# capped portfolio's degenerate active set leaves it loose while the
# objective still agrees to 1e-9. Objective agreement is asserted at one
# tolerance for all three.
_PROBLEMS = [
    ("readme", _README_QP, 1e-9),
    ("random_eq_ineq", _feasible_qp(seed=42, n=25, m_eq=3, m_in=40), 1e-5),
    ("capped_portfolio", _capped_portfolio_qp(
        seed=43, n=60, factors=8, cap=0.1), 1e-4),
]


def _solve_qtqp(p, a, b, c, z, **kwargs):
  """Solve a QTQP-convention problem with QTQP itself."""
  solver = qtqp.QTQP(
      p=None if p is None else sparse.csc_matrix(p),
      a=sparse.csc_matrix(a),
      b=b,
      c=c,
      z=z,
  )
  return solver.solve(verbose=False, **kwargs)


def _solve_goldfarb_idnani(p, a, b, c, z):
  """Solve the same problem with cvx-quadprog, translating the convention."""
  return quadprog.solve_qp(G=p, a=-c, C=-a.T, b=-b, meq=z)


def _objective(p, c, x):
  return 0.5 * x @ p @ x + c @ x


def _violation(a, b, z, x):
  """Return the largest constraint violation of `x`, in QTQP's convention."""
  equality = np.max(np.abs(a[:z] @ x - b[:z])) if z else 0.0
  inequality = np.max(a[z:] @ x - b[z:]) if z < a.shape[0] else 0.0
  return max(equality, inequality)


@pytest.mark.parametrize(
    "problem, x_atol",
    [pytest.param(problem, x_atol, id=name)
     for name, problem, x_atol in _PROBLEMS],
)
def test_agrees_with_goldfarb_idnani(problem, x_atol):
  """Both solvers must reach the same minimiser and the same objective."""
  p, a, b, c, z = problem

  solution = _solve_qtqp(p, a, b, c, z)
  reference = _solve_goldfarb_idnani(p, a, b, c, z)

  assert solution.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(
      _objective(p, c, solution.x),
      _objective(p, c, reference.x),
      rtol=1e-7,
      atol=1e-8,
  )
  np.testing.assert_allclose(solution.x, reference.x, rtol=0, atol=x_atol)

  # The reference is an exact active-set method, so it satisfies the
  # constraints to machine precision; QTQP is an interior point method and
  # stops at its own feasibility tolerance. Hold each to what it promises.
  assert _violation(a, b, z, reference.x) < 1e-12
  assert _violation(a, b, z, solution.x) < 1e-8


@pytest.mark.parametrize(
    "problem",
    [pytest.param(problem, id=name) for name, problem, _ in _PROBLEMS],
)
def test_reference_objective_matches_own_convention(problem):
  """`solution.f` must be the translated objective, not QTQP's.

  This is what pins the translation down: `solve_qp` reports its own
  objective, `1/2 x'Gx - a_gi'x`, and with `a_gi = -c` that is QTQP's
  `1/2 x'px + c'x` at the same point. If the mapping were wrong in a way
  that still produced a feasible answer, the two objectives would part.
  """
  p, a, b, c, z = problem
  reference = _solve_goldfarb_idnani(p, a, b, c, z)
  np.testing.assert_allclose(
      reference.f, _objective(p, c, reference.x), rtol=1e-9, atol=1e-12
  )


def test_duals_agree_on_readme_problem():
  """The dual variables agree entrywise, with no sign correction.

  Only the two rows the reference names in `iact` carry a multiplier; the
  third is inactive and zero in both.
  """
  p, a, b, c, z = _README_QP

  solution = _solve_qtqp(p, a, b, c, z)
  reference = _solve_goldfarb_idnani(p, a, b, c, z)

  np.testing.assert_allclose(
      solution.y, reference.lagrangian, rtol=1e-7, atol=1e-9
  )
  active = reference.iact - 1  # `iact` is 1-based.
  assert np.all(reference.lagrangian[active] > 0.0)
  inactive = np.setdiff1d(np.arange(a.shape[0]), active)
  np.testing.assert_allclose(reference.lagrangian[inactive], 0.0)


def test_qtqp_solves_semidefinite_p_that_reference_rejects():
  """A singular `p` is inside QTQP's problem class and outside the reference's."""
  _, a, b, c, z = _README_QP
  p = np.array([[1.0, 0.0], [0.0, 0.0]])

  assert _solve_qtqp(p, a, b, c, z).status == qtqp.SolutionStatus.SOLVED
  with pytest.raises(ValueError, match="not positive definite"):
    _solve_goldfarb_idnani(p, a, b, c, z)


def test_qtqp_solves_lp_that_reference_rejects():
  """With `p=None` the problem is an LP, which the dual method cannot take."""
  a = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
  b = np.array([1.0, 0.0, 0.0])
  c = np.array([-1.0, -2.0])

  solution = _solve_qtqp(None, a, b, c, z=0)
  assert solution.status == qtqp.SolutionStatus.SOLVED
  # min -x0 - 2 x1 over the unit simplex puts everything on the richer asset.
  np.testing.assert_allclose(solution.x, [0.0, 1.0], rtol=0, atol=1e-7)

  with pytest.raises(ValueError, match="not positive definite"):
    quadprog.solve_qp(G=np.zeros((2, 2)), a=-c, C=-a.T, b=-b, meq=0)


def test_infeasibility_is_reported_rather_than_raised():
  """QTQP returns a status where the reference raises."""
  # x0 <= -1 together with x0 >= 1.
  a = np.array([[1.0, 0.0], [-1.0, 0.0]])
  b = np.array([-1.0, -1.0])
  c = np.zeros(2)
  p = np.eye(2)

  assert _solve_qtqp(p, a, b, c, z=0).status == qtqp.SolutionStatus.INFEASIBLE
  with pytest.raises(ValueError, match="constraints are inconsistent"):
    _solve_goldfarb_idnani(p, a, b, c, 0)
