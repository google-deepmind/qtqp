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

"""Tests for QTQP solver."""

import sys
import numpy as np
import pytest
import qtqp
from scipy import sparse

_SOLVERS = [
    qtqp.LinearSolver.SCIPY,
    qtqp.LinearSolver.QDLDL,
    qtqp.LinearSolver.CHOLMOD,
    qtqp.LinearSolver.EIGEN,
    # Requires GPU:
    # qtqp.LinearSolver.CUDSS,
]

# Only run PARDISO on linux for now.
if sys.platform.startswith('linux'):
  _SOLVERS.append(qtqp.LinearSolver.PARDISO)

# Petsc4py not available on windows
if not sys.platform.startswith('win32'):
  _SOLVERS.append(qtqp.LinearSolver.MUMPS)


def _gen_feasible(m, n, z, random_state=None):
  """Generate a feasible QP."""
  rng = np.random.default_rng(random_state)
  w = rng.random(size=m)
  a = rng.random(size=(m, n))
  x = rng.random(size=n)
  p = rng.random(size=(n, n))
  y = w.copy()
  y[z:] = 0.5 * (w[z:] + np.abs(w[z:]))  # y = s - z;
  s = y - w
  p = p.T @ p * 0.01
  c = -a.T @ y
  b = a @ x + s
  return sparse.csc_matrix(a), b, c, sparse.csc_matrix(p)


def _gen_infeasible(m, n, z, random_state=None):
  """Generate an infeasible QP."""
  rng = np.random.default_rng(random_state)
  w = rng.random(size=m)
  a = rng.random(size=(m, n))
  b = rng.random(size=m)
  p = rng.random(size=(n, n))
  y = w.copy()
  y[z:] = 0.5 * (w[z:] + np.abs(w[z:]))  # y = s - z;
  a = a - np.outer(y, np.transpose(a).dot(y)) / np.linalg.norm(y) ** 2
  b = -b / np.dot(b, y)
  p = p.T @ p * 0.01
  c = -a.T @ y
  return sparse.csc_matrix(a), b, c, sparse.csc_matrix(p)


def _gen_unbounded(m, n, z, random_state=None):
  """Generate an unbounded QP."""
  rng = np.random.default_rng(random_state)
  w = rng.random(size=m)
  a = rng.random(size=(m, n))
  p = rng.random(size=(n, n))
  c = rng.random(size=n)
  s = np.zeros(m)
  s[z:] = 0.5 * (w[z:] + np.abs(w[z:]))
  p = p.T @ p * 0.01
  e, v = np.linalg.eig(p)
  e[-1] = 0
  x = v[:, -1]
  p = v @ np.diag(e) @ v.T
  a = a - np.outer(s + a.dot(x), x) / np.linalg.norm(x) ** 2
  c = -c / np.dot(c, x)
  b = s + a.dot(x)
  return sparse.csc_matrix(a), b, c, sparse.csc_matrix(p)


def _assert_solution(solution, a, b, c, p, z, atol=1e-7, rtol=1e-8):
  """Assert that the solution satisfies KKT conditions."""
  x = solution.x
  y = solution.y
  s = solution.s

  pcost = c @ x + 0.5 * x @ p @ x
  dcost = -b @ y - 0.5 * x @ p @ x
  pres = np.linalg.norm(a @ x + s - b, np.inf)
  dres = np.linalg.norm(p @ x + a.T @ y + c, np.inf)
  gap = np.abs(c @ x + b @ y + x @ p @ x)
  prelrhs = max(
      np.linalg.norm(a @ x, np.inf),
      np.linalg.norm(s, np.inf),
      np.linalg.norm(b, np.inf),
  )
  drelrhs = max(
      np.linalg.norm(p @ x, np.inf),
      np.linalg.norm(a.T @ y, np.inf),
      np.linalg.norm(c, np.inf),
  )
  assert solution.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_array_less(gap, atol + rtol * min(abs(pcost), abs(dcost)))
  np.testing.assert_array_less(pres, atol + rtol * prelrhs)
  np.testing.assert_array_less(dres, atol + rtol * drelrhs)
  np.testing.assert_array_less(-1e-9, np.min(y[z:], initial=0.0))
  np.testing.assert_array_less(-1e-9, np.min(s[z:], initial=0.0))


def _assert_infeasible(solution, a, b, z, atol=1e-8, rtol=1e-9):
  """Assert that the solution satisfies KKT conditions for primal infeasibility."""
  x = solution.x
  y = solution.y
  s = solution.s

  pinfeas = np.linalg.norm(a.T @ y, np.inf)

  assert solution.status == qtqp.SolutionStatus.INFEASIBLE
  np.testing.assert_array_equal(np.isnan(x), True)
  np.testing.assert_array_equal(np.isnan(s), True)
  np.testing.assert_allclose(b @ y, -1.0, atol=atol, rtol=rtol)
  np.testing.assert_array_less(-1e-9, np.min(y[z:], initial=0.0))
  np.testing.assert_array_less(pinfeas, atol + rtol * np.linalg.norm(y, np.inf))


def _assert_unbounded(solution, a, c, p, z, atol=1e-8, rtol=1e-9):
  """Assert that the solution satisfies KKT conditions for primal unboundedness."""
  x = solution.x
  y = solution.y
  s = solution.s

  dinfeas_a = np.linalg.norm(a @ x + s, np.inf)
  dinfeas_p = np.linalg.norm(p @ x, np.inf)

  assert solution.status == qtqp.SolutionStatus.UNBOUNDED
  np.testing.assert_array_equal(np.isnan(y), True)
  np.testing.assert_allclose(c @ x, -1.0, atol=atol, rtol=rtol)
  np.testing.assert_array_less(-1e-9, np.min(s[z:], initial=0.0))
  np.testing.assert_array_less(
      dinfeas_a, atol + rtol * np.linalg.norm(x, np.inf)
  )
  np.testing.assert_array_less(
      dinfeas_p, atol + rtol * np.linalg.norm(x, np.inf)
  )


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 42 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve(equilibrate, seed, linear_solver):
  """Test the QTQP solver."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  _assert_solution(solution, a, b, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 142 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_infeasible(equilibrate, seed, linear_solver):
  """Test the QTQP solver with infeasible QP."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  _assert_infeasible(solution, a, b, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', list(242 + np.arange(10)))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_unbounded(equilibrate, seed, linear_solver):
  """Test the QTQP solver with unbounded QP."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  _assert_unbounded(solution, a, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 342 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve_warm_start(equilibrate, seed, linear_solver):
  """Test the QTQP solver with warm start."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      x=solution.x,
      y=solution.y,
      s=solution.s,
      equilibrate=equilibrate,
      linear_solver=linear_solver,
  )
  _assert_solution(solution, a, b, c, p, z)
  assert solution.stats[-1]['iter'] == 0


def test_raise_error_no_positive_constraints():
  """Test that an error is raised when z >= m."""
  rng = np.random.default_rng(442)
  m, n = 10, 10
  z = m
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve()


def test_raise_error_invalid_initial_y():
  """Test that an error is raised when initial y has negative values."""
  rng = np.random.default_rng(542)
  m, n, z = 3, 2, 1
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):  # Violates y[z:] >= 0
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(y=np.array([1.0, -1.0, 1.0]))
  with pytest.raises(ValueError):  # Shape mismatch
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(y=np.array([1.0]))


def test_raise_error_invalid_initial_s():
  """Test that an error is raised when initial s has negative values."""
  rng = np.random.default_rng(642)
  m, n, z = 3, 2, 1
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):  # Violates s[z:] >= 0
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(s=np.array([0.0, -1.0, 1.0]))
  with pytest.raises(ValueError):  # Violates s[:z] == 0
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(s=np.array([1.0, 0.0, 0.0]))
  with pytest.raises(ValueError):  # Shape mismatch
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(s=np.array([0.0]))


def test_raise_error_negative_invalid_shapes():
  """Test that an error is raised when shapes are invalid."""
  rng = np.random.default_rng(742)
  m, n, z = 6, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=np.zeros(m + 1), c=c, z=z, p=p).solve()
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=b, c=np.zeros(m + 1), z=z, p=p).solve()
  with pytest.raises(ValueError):
    p_invalid = sparse.csc_matrix(np.ones((n + 1, n)))
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p_invalid).solve()


@pytest.mark.parametrize('seed', 842 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_direct_linear_solver(seed, linear_solver):
  """Test that the direct linear solver works as expected."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  mu = rng.uniform()
  s = rng.uniform(size=m)
  y = rng.uniform(size=m)
  s[:z] = 0.0
  d = np.concatenate([np.zeros(z), s[z:] / y[z:]])
  linear_solver = qtqp.direct.DirectKktSolver(
      a=a,
      p=p,
      z=z,
      min_static_regularization=1e-8,
      max_iterative_refinement_steps=10,
      atol=1e-12,
      rtol=1e-12,
      solver=linear_solver.value(),
  )
  q = np.concatenate([c, b])
  linear_solver.update(mu=mu, s=s, y=y)
  sol, _ = linear_solver.solve(rhs=q, warm_start=np.zeros(n + m))
  rhx_x = p @ sol[:n] + mu * sol[:n] + a.T @ sol[n:]
  rhs_y = -a @ sol[:n] + (d + mu) * sol[n:]
  np.testing.assert_allclose(rhx_x, c)
  np.testing.assert_allclose(rhs_y, b)


@pytest.mark.parametrize('seed', 942 + np.arange(20))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_resolvent_operator(seed, linear_solver):
  """Test that the resolvent operator is correctly computed with regularization."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  mu = rng.uniform()
  sigma = rng.uniform()  # sigma < 1 applies regularization.
  s = rng.uniform(size=m)
  y = rng.uniform(size=m)
  s[:z] = 0.0
  tau = np.array([1.0])
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)
  solver.q = np.concatenate([c, b])
  solver._linear_solver = qtqp.direct.DirectKktSolver(  # pylint: disable=protected-access
      a=a,
      p=p,
      z=z,
      min_static_regularization=1e-8,
      max_iterative_refinement_steps=10,
      atol=1e-12,
      rtol=1e-12,
      solver=linear_solver.value(),
  )
  solver._linear_solver.update(mu=mu, s=s, y=y)  # pylint: disable=protected-access
  solver.kinv_q, _ = solver._linear_solver.solve(  # pylint: disable=protected-access
      rhs=solver.q, warm_start=np.zeros(n + m)
  )
  r_anchor = rng.uniform(size=n + m)
  tau_anchor = np.array([rng.uniform()])
  x_new, y_new, tau_new, _ = solver._newton_step(  # pylint: disable=protected-access
      p=p,
      mu=mu,
      mu_target=sigma * mu,
      r_anchor=r_anchor,
      tau_anchor=tau_anchor,
      y=y,
      s=s,
      tau=tau,
      correction=None,
  )
  d = np.concatenate([np.zeros(z), s[z:] / y[z:]])
  np.testing.assert_allclose(
      p @ x_new + mu * x_new + a.T @ y_new + c * tau_new,
      (mu - sigma * mu) * r_anchor[:n],
  )
  np.testing.assert_allclose(
      -a @ x_new + (d + mu) * y_new + b * tau_new,
      np.concatenate([np.zeros(z), sigma * mu / y[z:] + s[z:]])
      + (mu - sigma * mu) * r_anchor[n:],
  )
  np.testing.assert_allclose(
      -c @ x_new
      - b @ y_new
      + mu * tau_new
      - x_new.T @ p @ x_new / tau_new
      - sigma * mu / tau_new,
      (mu - sigma * mu) * tau_anchor,
  )


@pytest.mark.parametrize('seed', 1042 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_newton_step(seed, linear_solver):
  """Test that taking a few Newton steps converges for a fixed mu."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  mu = rng.uniform()
  s = np.ones(m)
  y = np.ones(m)
  s[:z] = 0.0
  x = np.zeros(n)
  tau = np.array([1.0])
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)
  solver.q = np.concatenate([c, b])
  solver._linear_solver = qtqp.direct.DirectKktSolver(  # pylint: disable=protected-access
      a=a,
      p=p,
      z=z,
      min_static_regularization=1e-8,
      max_iterative_refinement_steps=10,
      atol=1e-12,
      rtol=1e-12,
      solver=linear_solver.value(),
  )
  for _ in range(10):  # 10 steps should be enough for convergence.
    solver._linear_solver.update(mu=mu, s=s, y=y)  # pylint: disable=protected-access
    solver.kinv_q, _ = solver._linear_solver.solve(  # pylint: disable=protected-access
        rhs=solver.q, warm_start=np.zeros(n + m)
    )
    x_t, y_t, tau_t, _ = solver._newton_step(  # pylint: disable=protected-access
        p=p,
        mu=mu,
        mu_target=mu,  # fixed mu = mu_target for testing.
        r_anchor=np.zeros(n + m),
        tau_anchor=np.array([0.0]),
        y=y,
        s=s,
        tau=tau,
        correction=None,
    )
    d_x, d_y, d_tau = x_t - x, y_t - y, tau_t - tau
    d_s = np.zeros(m)
    d_s[z:] = mu / y[z:] - y_t[z:] * s[z:] / y[z:]

    step_size = 0.99 * solver._compute_step_size(y, s, d_y, d_s)  # pylint: disable=protected-access
    x += step_size * d_x
    y += step_size * d_y
    tau += step_size * d_tau
    s += step_size * d_s

    # Ensure variables stay strictly in the cone to prevent numerical issu

    y[z:] = np.maximum(y[z:], 1e-30)
    s[z:] = np.maximum(s[z:], 1e-30)
    tau = np.maximum(tau, 1e-30)

  np.testing.assert_allclose(p @ x + mu * x + a.T @ y, -c * tau)
  np.testing.assert_allclose(
      -a @ x + mu * y,
      -b * tau + np.concatenate([np.zeros(z), s[z:]]),
  )
  np.testing.assert_allclose(
      -c @ x - b @ y + mu * tau,
      x.T @ p @ x / tau + mu / tau,
  )
  np.testing.assert_allclose(s[z:] * y[z:], mu * np.ones(m - z))
