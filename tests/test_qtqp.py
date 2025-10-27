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

import numpy as np
import pytest
import qtqp
from scipy import sparse

_SOLVERS = [
    qtqp.LinearSolver.SCIPY,
    # Some tests fail with PARDISO due to numerical issues.
    qtqp.LinearSolver.PARDISO,
    qtqp.LinearSolver.QDLDL,
    qtqp.LinearSolver.CHOLMOD,
    qtqp.LinearSolver.EIGEN,
    # Requires GPU:
    # qtqp.LinearSolver.CUDSS,
]


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
  c = a @ y
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
@pytest.mark.parametrize('seed', np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve(equilibrate, seed, linear_solver):
  """Test the QTQP solver."""
  rng = np.random.default_rng(seed)
  m, n, z = 100, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  _assert_solution(solution, a, b, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_unbounded(equilibrate, seed, linear_solver):
  """Test the QTQP solver with unbounded QP."""
  rng = np.random.default_rng(seed)
  m, n, z = 100, 100, 10
  a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  _assert_unbounded(solution, a, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_infeasible(equilibrate, seed, linear_solver):
  """Test the QTQP solver with infeasible QP."""
  rng = np.random.default_rng(seed)
  m, n, z = 100, 100, 10
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver
  )
  _assert_infeasible(solution, a, b, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve_warm_start(equilibrate, seed, linear_solver):
  """Test the QTQP solver with warm start."""
  rng = np.random.default_rng(seed)
  m, n, z = 100, 100, 10
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
  rng = np.random.default_rng(42)
  m, n = 10, 10
  z = m
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve()


def test_raise_error_invalid_initial_y():
  """Test that an error is raised when initial y has negative values."""
  rng = np.random.default_rng(42)
  m, n, z = 3, 2, 1
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):  # Violates y[z:] >= 0
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(y=np.array([1.0, -1.0, 1.0]))
  with pytest.raises(ValueError):  # Shape mismatch
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(y=np.array([1.0]))


def test_raise_error_invalid_initial_s():
  """Test that an error is raised when initial s has negative values."""
  rng = np.random.default_rng(42)
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
  rng = np.random.default_rng(42)
  m, n, z = 6, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=np.zeros(m + 1), c=c, z=z, p=p).solve()
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=b, c=np.zeros(m + 1), z=z, p=p).solve()
  with pytest.raises(ValueError):
    p_invalid = sparse.csc_matrix(np.ones((n + 1, n)))
    _ = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p_invalid).solve()
