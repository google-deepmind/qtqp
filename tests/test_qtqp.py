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
    qtqp.LinearSolver.SCIPY_DENSE,
    qtqp.LinearSolver.UMFPACK,
    qtqp.LinearSolver.QDLDL,
    qtqp.LinearSolver.CHOLMOD,
    qtqp.LinearSolver.EIGEN,
]


class _TriangularMatvecSolver(qtqp.direct.LinearSolver):
  """Minimal solver used only to exercise the base symmetric-triangle matvec."""

  def factorize(self):
    pass

  def solve(self, rhs):
    raise NotImplementedError("Test-only solver; not intended for solve().")

  def format(self):
    return 'csc'


def test_auto_prefers_linux_windows_primary_backend(monkeypatch):
  """AUTO should try PARDISO first on non-macOS platforms."""
  attempts = []
  scipy_backend = qtqp.direct.ScipySolver()
  monkeypatch.setattr(qtqp, '_AUTO_SOLVER_CACHE', {})

  def fake_instantiate(linear_solver):
    attempts.append(linear_solver)
    if linear_solver is qtqp.LinearSolver.PARDISO:
      raise ImportError("pymklpardiso not installed")
    if linear_solver is qtqp.LinearSolver.QDLDL:
      raise ImportError("qdldl not installed")
    if linear_solver is qtqp.LinearSolver.UMFPACK:
      raise ImportError("umfpack not installed")
    if linear_solver is qtqp.LinearSolver.CHOLMOD:
      raise ImportError("cholmod not installed")
    if linear_solver is qtqp.LinearSolver.EIGEN:
      raise ImportError("nanoeigenpy not installed")
    if linear_solver is qtqp.LinearSolver.MUMPS:
      raise ImportError("petsc4py not installed")
    if linear_solver is qtqp.LinearSolver.SCIPY:
      return scipy_backend
    raise AssertionError(f"Unexpected AUTO candidate: {linear_solver}")

  monkeypatch.setattr(qtqp.sys, 'platform', 'linux')
  monkeypatch.setattr(qtqp, '_instantiate_linear_solver', fake_instantiate)

  resolved, backend = qtqp._resolve_linear_solver(qtqp.LinearSolver.AUTO)

  assert attempts == [
      qtqp.LinearSolver.PARDISO,
      qtqp.LinearSolver.CHOLMOD,
      qtqp.LinearSolver.QDLDL,
      qtqp.LinearSolver.EIGEN,
      qtqp.LinearSolver.MUMPS,
      qtqp.LinearSolver.UMFPACK,
      qtqp.LinearSolver.SCIPY,
  ]
  assert resolved is qtqp.LinearSolver.SCIPY
  assert backend is scipy_backend


def test_auto_prefers_macos_primary_backend(monkeypatch):
  """AUTO should try ACCELERATE first on macOS."""
  attempts = []
  scipy_backend = qtqp.direct.ScipySolver()
  monkeypatch.setattr(qtqp, '_AUTO_SOLVER_CACHE', {})

  def fake_instantiate(linear_solver):
    attempts.append(linear_solver)
    if linear_solver is qtqp.LinearSolver.ACCELERATE:
      raise ImportError("macldlt not installed")
    if linear_solver is qtqp.LinearSolver.QDLDL:
      raise ImportError("qdldl not installed")
    if linear_solver is qtqp.LinearSolver.UMFPACK:
      raise ImportError("umfpack not installed")
    if linear_solver is qtqp.LinearSolver.CHOLMOD:
      raise ImportError("cholmod not installed")
    if linear_solver is qtqp.LinearSolver.EIGEN:
      raise ImportError("nanoeigenpy not installed")
    if linear_solver is qtqp.LinearSolver.MUMPS:
      raise ImportError("petsc4py not installed")
    if linear_solver is qtqp.LinearSolver.SCIPY:
      return scipy_backend
    raise AssertionError(f"Unexpected AUTO candidate: {linear_solver}")

  monkeypatch.setattr(qtqp.sys, 'platform', 'darwin')
  monkeypatch.setattr(qtqp, '_instantiate_linear_solver', fake_instantiate)

  resolved, backend = qtqp._resolve_linear_solver(qtqp.LinearSolver.AUTO)

  assert attempts == [
      qtqp.LinearSolver.ACCELERATE,
      qtqp.LinearSolver.CHOLMOD,
      qtqp.LinearSolver.QDLDL,
      qtqp.LinearSolver.EIGEN,
      qtqp.LinearSolver.MUMPS,
      qtqp.LinearSolver.UMFPACK,
      qtqp.LinearSolver.SCIPY,
  ]
  assert resolved is qtqp.LinearSolver.SCIPY
  assert backend is scipy_backend


def test_auto_caches_resolved_backend(monkeypatch):
  """AUTO should probe once per platform and reuse the resolved backend."""
  attempts = []
  monkeypatch.setattr(qtqp.sys, 'platform', 'linux')
  monkeypatch.setattr(qtqp, '_AUTO_SOLVER_CACHE', {})

  def fake_instantiate(linear_solver):
    attempts.append(linear_solver)
    if linear_solver in (
        qtqp.LinearSolver.PARDISO,
        qtqp.LinearSolver.CHOLMOD,
        qtqp.LinearSolver.QDLDL,
        qtqp.LinearSolver.EIGEN,
        qtqp.LinearSolver.MUMPS,
        qtqp.LinearSolver.UMFPACK,
    ):
      raise ImportError(f"{linear_solver.name} unavailable")
    if linear_solver is qtqp.LinearSolver.SCIPY:
      return qtqp.direct.ScipySolver()
    raise AssertionError(f"Unexpected AUTO candidate: {linear_solver}")

  monkeypatch.setattr(qtqp, '_instantiate_linear_solver', fake_instantiate)

  first_resolved, first_backend = qtqp._resolve_linear_solver(
      qtqp.LinearSolver.AUTO
  )
  second_resolved, second_backend = qtqp._resolve_linear_solver(
      qtqp.LinearSolver.AUTO
  )

  assert first_resolved is qtqp.LinearSolver.SCIPY
  assert second_resolved is qtqp.LinearSolver.SCIPY
  assert isinstance(first_backend, qtqp.direct.ScipySolver)
  assert isinstance(second_backend, qtqp.direct.ScipySolver)
  assert attempts == [
      qtqp.LinearSolver.PARDISO,
      qtqp.LinearSolver.CHOLMOD,
      qtqp.LinearSolver.QDLDL,
      qtqp.LinearSolver.EIGEN,
      qtqp.LinearSolver.MUMPS,
      qtqp.LinearSolver.UMFPACK,
      qtqp.LinearSolver.SCIPY,
      qtqp.LinearSolver.SCIPY,
  ]

try:
  import pymklpardiso  # noqa: F401
  _SOLVERS.append(qtqp.LinearSolver.PARDISO)
except (ImportError, ModuleNotFoundError) as e:
  print(f'Skipping PARDISO tests: {e}')

# Accelerate is macOS only.
if sys.platform == 'darwin':
  _SOLVERS.append(qtqp.LinearSolver.ACCELERATE)

# Petsc4py not available on windows; some conda builds also fail to load
# (e.g. CUDA-linked builds on machines without a GPU).
try:
  import petsc4py.PETSc  # noqa: F401
  _SOLVERS.append(qtqp.LinearSolver.MUMPS)
except (ImportError, ModuleNotFoundError) as e:
  print(f'Skipping MUMPS tests: {e}')

try:
  import cupy  # noqa: F401
  if cupy.cuda.runtime.getDeviceCount() > 0:
    _SOLVERS.append(qtqp.LinearSolver.CUPY_DENSE)
except Exception as e:  # pylint: disable=broad-exception-caught
  print(f'Skipping CUPY_DENSE tests: {e}')

try:
  import cupy  # noqa: F401
  import nvmath  # noqa: F401
  if cupy.cuda.runtime.getDeviceCount() > 0:
    _SOLVERS.append(qtqp.LinearSolver.CUDSS)
except Exception as e:  # pylint: disable=broad-exception-caught
  print(f'Skipping CUDSS tests: {e}')


def _gen_feasible(m, n, z, random_state=None):
  """Generate a feasible QP."""
  rng = np.random.default_rng(random_state)
  w = rng.normal(size=m)
  x = rng.normal(size=n)
  y = w.copy()
  y[z:] = 0.5 * (w[z:] + np.abs(w[z:]))  # y = s - z;
  s = y - w

  a = sparse.random(
      m,
      n,
      density=0.1,
      format='csc',
      rng=rng,
      data_rvs=lambda x: rng.normal(size=x),
  )
  p = sparse.random(
      n,
      n,
      density=0.01,
      format='csc',
      rng=rng,
      data_rvs=lambda x: rng.normal(size=x),
  )

  c = -a.T @ y
  b = a @ x + s
  p = p.T @ p * 0.01
  return sparse.csc_matrix(a), b, c, sparse.csc_matrix(p)


def _gen_infeasible(m, n, z, random_state=None):
  """Generate an infeasible QP."""
  rng = np.random.default_rng(random_state)
  w = rng.random(size=m)
  b = rng.normal(size=m)
  y = w.copy()
  y[z:] = 0.5 * (w[z:] + np.abs(w[z:]))  # y = s - z;

  a = rng.normal(size=(m, n))
  p = rng.normal(size=(n, n))

  a = a - np.outer(y, a.T @ y) / np.linalg.norm(y) ** 2
  b = -b / (b @ y)
  p = p.T @ p * 0.01
  c = rng.normal(size=n)
  return sparse.csc_matrix(a), b, c, sparse.csc_matrix(p)


def _gen_unbounded(m, n, z, random_state=None):
  """Generate an unbounded QP."""
  rng = np.random.default_rng(random_state)
  w = rng.random(size=m)
  c = rng.normal(size=n)
  s = np.zeros(m)
  s[z:] = 0.5 * (w[z:] + np.abs(w[z:]))

  a = rng.normal(size=(m, n))
  p = rng.normal(size=(n, n))

  p = p.T @ p * 0.01
  e, v = np.linalg.eig(p)
  e[-1] = 0.0
  x = v[:, -1]
  p = v @ np.diag(e) @ v.T
  a = a - np.outer(s + a @ x, x) / np.linalg.norm(x) ** 2
  c = -c / (c @ x)
  b = rng.normal(size=m)
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
@pytest.mark.parametrize('mnz', ((150, 100, 10), (10, 5, 3), (500, 300, 30)))
def test_solve(equilibrate, seed, linear_solver, mnz, record_iterations):
  """Test the QTQP solver."""
  rng = np.random.default_rng(seed)
  m, n, z = mnz
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True
  )

  # Record stats
  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])

  _assert_solution(solution, a, b, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 142 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
@pytest.mark.parametrize('mnz', ((150, 100, 10), (500, 300, 30)))
def test_infeasible(equilibrate, seed, linear_solver, mnz, record_iterations):
  """Test the QTQP solver with infeasible QP."""
  rng = np.random.default_rng(seed)
  m, n, z = mnz
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True
  )

  # Record stats
  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])

  _assert_infeasible(solution, a, b, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', list(242 + np.arange(10)))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
@pytest.mark.parametrize('mnz', ((150, 100, 10), (500, 300, 30)))
def test_unbounded(equilibrate, seed, linear_solver, mnz, record_iterations):
  """Test the QTQP solver with unbounded QP."""
  rng = np.random.default_rng(seed)
  m, n, z = mnz
  a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True
  )

  # Record stats
  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])

  _assert_unbounded(solution, a, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 6042 + np.arange(3))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve_large(equilibrate, seed, linear_solver, record_iterations):
  """Test solver on larger instances (1000x600) to stress backends."""
  rng = np.random.default_rng(seed)
  m, n, z = 1000, 600, 60
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_solution(solution, a, b, c, p, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 6142 + np.arange(3))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_infeasible_large(equilibrate, seed, linear_solver, record_iterations):
  """Test infeasible detection on larger instances (1000x600)."""
  rng = np.random.default_rng(seed)
  m, n, z = 1000, 600, 60
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_infeasible(solution, a, b, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', list(6242 + np.arange(3)))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_unbounded_large(equilibrate, seed, linear_solver, record_iterations):
  """Test unbounded detection on larger instances (1000x600)."""
  rng = np.random.default_rng(seed)
  m, n, z = 1000, 600, 60
  a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_unbounded(solution, a, c, p, z)


def test_raise_error_z_greater_than_m():
  """Test that an error is raised when z > m."""
  rng = np.random.default_rng(442)
  m, n = 10, 20
  a = sparse.random(m, n, density=0.1, format='csc', random_state=rng)
  b = rng.normal(size=m)
  c = rng.normal(size=n)
  p = sparse.csc_matrix((n, n))
  with pytest.raises(ValueError):
    _ = qtqp.QTQP(a=a, b=b, c=c, z=m + 1, p=p).solve()


def _gen_equality_only(m, n, random_state=None):
  """Generate a well-conditioned equality-only QP with known solution.

  Constructs (P, A, b, c) such that the KKT system is non-singular and the
  optimal (x*, y*) is known. Requires m <= n.
  """
  rng = np.random.default_rng(random_state)
  x_star = rng.normal(size=n)
  y_star = rng.normal(size=m)
  a_dense = rng.normal(size=(m, n))
  a = sparse.csc_matrix(a_dense)
  q = rng.normal(size=(n, n))
  p = sparse.csc_matrix(q.T @ q * 0.01)
  b = a @ x_star
  c = -(p @ x_star + a.T @ y_star)
  return a, b, c, p


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 1542 + np.arange(10))
@pytest.mark.parametrize('mn', ((5, 10), (30, 50), (80, 100)))
def test_equality_only_solve(equilibrate, seed, mn):
  """Test equality-only QP (z == m) solved via direct KKT system."""
  rng = np.random.default_rng(seed)
  m, n = mn
  z = m
  a, b, c, p = _gen_equality_only(m, n, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, equilibrate=equilibrate,
  )
  _assert_solution(solution, a, b, c, p, z)
  assert solution.stats == []


@pytest.mark.parametrize('seed', 2042 + np.arange(5))
def test_equality_only_lp(seed):
  """Test equality-only LP (P=0, z=m) solved via direct KKT system.

  Uses n == m (square A) so the KKT system is non-singular with P=0.
  """
  rng = np.random.default_rng(seed)
  m, n, z = 10, 10, 10
  a_dense = rng.normal(size=(m, n))
  a = sparse.csc_matrix(a_dense)
  x_star = rng.normal(size=n)
  y_star = rng.normal(size=m)
  b = a @ x_star
  c = -(a.T @ y_star)
  p = sparse.csc_matrix((n, n))
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z).solve(verbose=False)
  _assert_solution(solution, a, b, c, p, z)
  assert solution.stats == []


def test_equality_only_singular():
  """Test that a singular equality-only KKT system returns FAILED."""
  n, m, z = 5, 3, 3
  # All rows identical -> singular A.
  a = sparse.csc_matrix(np.ones((m, n)))
  b = np.array([1.0, 2.0, 3.0])  # Inconsistent for identical rows.
  c = np.ones(n)
  p = sparse.csc_matrix((n, n))
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  assert solution.status == qtqp.SolutionStatus.FAILED


def test_presolve_drops_all_inequalities():
  """Test presolve dropping all inequalities triggers direct solve."""
  rng = np.random.default_rng(842)
  m_eq, n = 5, 20
  m_ineq = 5
  m = m_eq + m_ineq
  z = m_eq
  a, b, c, p = _gen_equality_only(m_eq, n, random_state=rng)
  # Append inequality rows with b = inf (will be dropped by presolve).
  a_ineq = sparse.csc_matrix(rng.normal(size=(m_ineq, n)))
  a = sparse.vstack([a, a_ineq], format='csc')
  b = np.concatenate([b, np.full(m_ineq, 1e21)])
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  assert solution.status == qtqp.SolutionStatus.SOLVED
  # Postsolve should restore original dimensions.
  assert solution.y.shape == (m,)
  assert solution.s.shape == (m,)


@pytest.mark.parametrize('linear_solver', _SOLVERS)
@pytest.mark.parametrize('seed', 3042 + np.arange(5))
def test_equality_only_all_backends(linear_solver, seed):
  """Test equality-only QP with every linear solver backend."""
  rng = np.random.default_rng(seed)
  m, n, z = 20, 40, 20
  a, b, c, p = _gen_equality_only(m, n, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, linear_solver=linear_solver,
  )
  _assert_solution(solution, a, b, c, p, z)
  assert solution.stats == []


def test_equality_only_recovers_known_solution():
  """Test that the direct solve recovers the ground-truth (x*, y*)."""
  rng = np.random.default_rng(7742)
  m, n, z = 10, 20, 10
  x_star = rng.normal(size=n)
  y_star = rng.normal(size=m)
  a_dense = rng.normal(size=(m, n))
  a = sparse.csc_matrix(a_dense)
  q = rng.normal(size=(n, n))
  p = sparse.csc_matrix(q.T @ q * 0.01)
  b = a @ x_star
  c = -(p @ x_star + a.T @ y_star)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  assert solution.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, x_star, atol=1e-6)
  np.testing.assert_allclose(solution.y, y_star, atol=1e-6)
  np.testing.assert_allclose(solution.s, np.zeros(m), atol=1e-10)


def test_equality_only_verbose(capsys):
  """Test that equality-only path prints header and footer."""
  rng = np.random.default_rng(8842)
  m, n, z = 5, 10, 5
  a, b, c, p = _gen_equality_only(m, n, random_state=rng)
  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=True)
  assert solution.status == qtqp.SolutionStatus.SOLVED
  captured = capsys.readouterr().out
  assert 'QTQP v' in captured
  assert 'z=5' in captured
  assert 'equality-only' in captured


def test_equality_only_sparse_p():
  """Test equality-only QP where P is sparse (not dense-generated)."""
  rng = np.random.default_rng(9942)
  m, n, z = 15, 30, 15
  x_star = rng.normal(size=n)
  y_star = rng.normal(size=m)
  a = sparse.random(
      m, n, density=0.5, format='csc',
      data_rvs=lambda s: rng.normal(size=s), random_state=rng,
  )
  p = sparse.random(
      n, n, density=0.05, format='csc',
      data_rvs=lambda s: rng.normal(size=s), random_state=rng,
  )
  p = (p.T @ p) * 0.01  # Make PSD.
  b = a @ x_star
  c = -(p @ x_star + a.T @ y_star)
  solution = qtqp.QTQP(
      a=sparse.csc_matrix(a), b=b, c=c, z=z, p=sparse.csc_matrix(p),
  ).solve(verbose=False)
  _assert_solution(solution, sparse.csc_matrix(a), b, c, sparse.csc_matrix(p), z)
  assert solution.stats == []


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
  linear_solver.free()
  np.testing.assert_allclose(
      p @ sol[:n] + mu * sol[:n] + a.T @ sol[n:], c, atol=1e-10, rtol=1e-10
  )
  np.testing.assert_allclose(
      -a @ sol[:n] + (d + mu) * sol[n:], b, atol=1e-10, rtol=1e-10
  )


def test_upper_triangular_kkt_matvec_matches_full():
  """Upper-triangular KKT storage must reproduce the full symmetric matvec."""
  rng = np.random.default_rng(2026)
  m, n, z = 20, 12, 4
  a, _, _, p = _gen_feasible(m, n, z, random_state=rng)
  mu = rng.uniform()
  s = rng.uniform(size=m)
  y = rng.uniform(size=m)
  s[:z] = 0.0
  vec = rng.normal(size=n + m)

  linear_solver = qtqp.direct.DirectKktSolver(
      a=a,
      p=p,
      z=z,
      min_static_regularization=1e-8,
      max_iterative_refinement_steps=2,
      atol=1e-12,
      rtol=1e-12,
      solver=_TriangularMatvecSolver(),
  )
  linear_solver.update(mu=mu, s=s, y=y)

  # The reference KKT below uses unregularized diagonals, so verify
  # that regularization did not alter any diagonal entry.
  diag_x = p.diagonal() + mu
  diag_y = np.full(m, mu, dtype=np.float64)
  diag_y[z:] = s[z:] / y[z:] + mu
  min_reg = 1e-8
  assert np.all(diag_x >= min_reg) and np.all(diag_y >= min_reg)
  kkt_full = sparse.bmat(
      [
          [p + sparse.diags(np.full(n, mu)), a.T],
          [a, -sparse.diags(diag_y)],
      ],
      format='csc',
      dtype=np.float64,
  )

  assert (linear_solver._kkt - sparse.triu(linear_solver._kkt)).nnz == 0  # pylint: disable=protected-access
  np.testing.assert_allclose(
      linear_solver._solver @ vec,  # pylint: disable=protected-access
      kkt_full @ vec,
      rtol=1e-10,
      atol=1e-10,
  )


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
  solver._linear_solver.free()  # pylint: disable=protected-access
  np.testing.assert_allclose(
      p @ x_new + mu * x_new + a.T @ y_new + c * tau_new,
      (mu - sigma * mu) * r_anchor[:n],
      atol=1e-10,
      rtol=1e-10,
  )
  np.testing.assert_allclose(
      -a @ x_new + (d + mu) * y_new + b * tau_new,
      np.concatenate([np.zeros(z), sigma * mu / y[z:] + s[z:]])
      + (mu - sigma * mu) * r_anchor[n:],
      atol=1e-10,
      rtol=1e-10,
  )
  np.testing.assert_allclose(
      -c @ x_new
      - b @ y_new
      + mu * tau_new
      - x_new.T @ p @ x_new / tau_new
      - sigma * mu / tau_new,
      (mu - sigma * mu) * tau_anchor,
      atol=1e-10,
      rtol=1e-10,
  )


@pytest.mark.parametrize('seed', 1042 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_newton_step_converges_to_central_path(seed, linear_solver):
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
  for _ in range(20):  # 20 steps should be enough for convergence.
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

    # Ensure variables stay strictly in the cone to prevent numerical issues.

    y[z:] = np.maximum(y[z:], 1e-30)
    s[z:] = np.maximum(s[z:], 1e-30)
    tau = np.maximum(tau, 1e-30)

  solver._linear_solver.free()  # pylint: disable=protected-access
  np.testing.assert_allclose(
      p @ x + mu * x + a.T @ y + c * tau, np.zeros(n), atol=1e-9, rtol=1e-9
  )
  np.testing.assert_allclose(
      -a @ x + mu * y + b * tau,
      np.concatenate([np.zeros(z), s[z:]]),
      atol=1e-9,
      rtol=1e-9,
  )
  np.testing.assert_allclose(
      -c @ x - b @ y + mu * tau - x.T @ p @ x / tau - mu / tau,
      0.0,
      atol=1e-9,
      rtol=1e-9,
  )
  np.testing.assert_allclose(
      s[z:] * y[z:], mu * np.ones(m - z), atol=1e-9, rtol=1e-9
  )


def _solve_for_tau(n, q, p, kinv_r, kinv_q, mu, mu_target, r_tau):
  """Solves the quadratic equation for the tau step in homogeneous embedding."""
  v = p @ np.stack([kinv_r[:n], kinv_q[:n]], axis=1)
  p_kinv_r, p_kinv_q = v[:, 0], v[:, 1]

  t_a = mu + kinv_q @ q - kinv_q[:n] @ p_kinv_q
  t_b = -r_tau[0] - kinv_r @ q + kinv_r[:n] @ p_kinv_q + kinv_q[:n] @ p_kinv_r
  t_c = -kinv_r[:n] @ p_kinv_r - mu_target
  discriminant = t_b**2 - 4 * t_a * t_c
  return (-t_b + np.sqrt(max(0.0, discriminant))) / (2 * t_a)


def _solve_for_tau_alternative(
    n, kinv_q, kinv_r, mu, mu_target, r, r_tau, s, y
):
  """Solves the quadratic equation for the tau step in homogeneous embedding."""
  kinv_r_d_kinv_r = kinv_r[n:] @ (kinv_r[n:] * s / y)
  kinv_q_d_kinv_q = kinv_q[n:] @ (kinv_q[n:] * s / y)
  kinv_q_d_kinv_r = kinv_q[n:] @ (kinv_r[n:] * s / y)

  t_a = mu + mu * kinv_q @ kinv_q + kinv_q_d_kinv_q
  t_b = -r_tau[0] + kinv_q @ r - 2 * (mu * kinv_q @ kinv_r + kinv_q_d_kinv_r)
  t_c = -kinv_r @ r + mu * kinv_r @ kinv_r + kinv_r_d_kinv_r - mu_target
  discriminant = t_b**2 - 4 * t_a * t_c
  return (-t_b + np.sqrt(max(0.0, discriminant))) / (2 * t_a)


@pytest.mark.parametrize('seed', 1142 + np.arange(10))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_equivalent_tau_solution(seed, linear_solver):
  """Test that solving for tau using different methods gives equivalent results."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  mu = rng.uniform()
  sigma = rng.uniform()  # sigma < 1 applies regularization.
  mu_target = sigma * mu
  s = rng.uniform(size=m)
  y = rng.uniform(size=m)
  s[:z] = 0.0
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
  r = (mu - mu_target) * r_anchor
  r[n + z :] += mu_target / y[z:] + s[z:]
  kinv_r, _ = solver._linear_solver.solve(  # pylint: disable=protected-access
      rhs=r, warm_start=np.zeros(n + m)
  )
  tau_anchor = np.array([rng.uniform()])
  r_tau = (mu - mu_target) * tau_anchor
  tau_qtqp = solver._solve_for_tau(p, kinv_r, mu, mu_target, r_tau)  # pylint: disable=protected-access
  tau_1 = _solve_for_tau_alternative(
      n, solver.kinv_q, kinv_r, mu, mu_target, r, r_tau, s, y
  )
  tau_2 = _solve_for_tau(
      n, solver.q, p, kinv_r, solver.kinv_q, mu, mu_target, r_tau
  )
  np.testing.assert_allclose(tau_qtqp, tau_1, atol=1e-11, rtol=1e-11)
  np.testing.assert_allclose(tau_qtqp, tau_2, atol=1e-11, rtol=1e-11)
  np.testing.assert_allclose(tau_1, tau_2, atol=1e-11, rtol=1e-11)


def _equilibrate_reference(a, p, num_iters=10, min_scale=1e-3, max_scale=1e3):
  """Reference Ruiz equilibration using sparse diagonal matrix products."""
  a, p = a.copy(), p.copy()
  d, e = np.ones(a.shape[0]), np.ones(a.shape[1])
  for _ in range(num_iters):
    d_i = sparse.linalg.norm(a, np.inf, axis=1)
    d_i = np.where(d_i == 0.0, 1.0, d_i)
    d_i = np.clip(1.0 / np.sqrt(d_i), min_scale, max_scale)
    e_i = np.maximum(sparse.linalg.norm(a, np.inf, axis=0),
                     sparse.linalg.norm(p, np.inf, axis=0))
    e_i = np.where(e_i == 0.0, 1.0, e_i)
    e_i = np.clip(1.0 / np.sqrt(e_i), min_scale, max_scale)
    d_mat, e_mat = sparse.diags(d_i), sparse.diags(e_i)
    a = d_mat @ a @ e_mat
    p = e_mat @ p @ e_mat
    d *= d_i
    e *= e_i
  return a, p, d, e


@pytest.mark.parametrize('seed', 2142 + np.arange(10))
def test_equivalent_equilibration(seed):
  """Test that in-place equilibration matches the reference sparse matmul."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)
  a_eq, p_eq, _, _, d, e = solver._equilibrate()  # pylint: disable=protected-access

  a_ref, p_ref, d_ref, e_ref = _equilibrate_reference(a, p)

  np.testing.assert_allclose(a_eq.toarray(), a_ref.toarray(), atol=1e-14, rtol=1e-14)
  np.testing.assert_allclose(p_eq.toarray(), p_ref.toarray(), atol=1e-14, rtol=1e-14)
  np.testing.assert_allclose(d, d_ref, atol=1e-14, rtol=1e-14)
  np.testing.assert_allclose(e, e_ref, atol=1e-14, rtol=1e-14)


def _compute_sigma_reference(solver, mu_curr, x, y, tau, s, alpha, d_x, d_y,
                              d_tau, d_s):
  """Reference sigma using _normalize then mu = (y @ s) / (m - z), as in the
  original implementation before the allocation-free optimisation."""
  _EPS = 1e-15  # matches qtqp._EPS
  m, z = solver.m, solver.z
  x_aff = x + alpha * d_x
  y_aff = y + alpha * d_y
  tau_aff = tau + alpha * d_tau
  s_aff = s + alpha * d_s
  _, y_aff_n, _, s_aff_n = solver._normalize(x_aff, y_aff, tau_aff, s_aff)  # pylint: disable=protected-access
  mu_aff = (y_aff_n @ s_aff_n) / (m - z)
  sigma = (mu_aff / max(_EPS, mu_curr)) ** 3
  return float(np.clip(sigma, 0.0, 1.0))


@pytest.mark.parametrize('seed', 3142 + np.arange(10))
def test_equivalent_compute_sigma(seed):
  """Test that the optimised sigma matches the reference normalise-then-compute."""
  rng = np.random.default_rng(seed)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  mu = rng.uniform(1e-6, 1.0)
  alpha = rng.uniform(0.0, 1.0)
  x = rng.normal(size=n)
  y = rng.uniform(size=m)
  tau = np.array([rng.uniform(0.1, 2.0)])
  s = rng.uniform(size=m)
  d_x = rng.normal(size=n)
  d_y = rng.normal(size=m)
  d_tau = np.array([rng.normal()])
  d_s = rng.normal(size=m)

  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)
  sigma = solver._compute_sigma(mu, x, y, tau, s, alpha, d_x, d_y, d_tau, d_s)  # pylint: disable=protected-access
  sigma_ref = _compute_sigma_reference(solver, mu, x, y, tau, s, alpha, d_x,
                                       d_y, d_tau, d_s)

  np.testing.assert_allclose(sigma, sigma_ref, atol=1e-14, rtol=1e-14)


# =============================================================================
# LP tests (p=None)
# =============================================================================

@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 4042 + np.arange(5))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve_lp(equilibrate, seed, linear_solver, record_iterations):
  """Test QTQP as an LP (p=None); verifies the p.nnz==0 code path."""
  rng = np.random.default_rng(seed)
  m, n, z = 50, 30, 5
  a, b, c, _ = _gen_feasible(m, n, z, random_state=rng)
  p_zero = sparse.csc_matrix((n, n))

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True,
      verbose=False,
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_solution(solution, a, b, c, p_zero, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 4142 + np.arange(5))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_infeasible_lp(equilibrate, seed, linear_solver, record_iterations):
  """Test infeasible LP detection (p=None)."""
  rng = np.random.default_rng(seed)
  m, n, z = 50, 30, 5
  a, b, c, _ = _gen_infeasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True,
      verbose=False,
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_infeasible(solution, a, b, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 4242 + np.arange(5))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_unbounded_lp(equilibrate, seed, linear_solver, record_iterations):
  """Test unbounded LP detection (p=None).

  _gen_unbounded constructs a direction x with c'x=-1 and Ax+s=0, s[z:]>=0.
  That direction is valid for the LP regardless of P, so passing p=None still
  yields an UNBOUNDED solution.
  """
  rng = np.random.default_rng(seed)
  m, n, z = 50, 30, 5
  a, b, c, _ = _gen_unbounded(m, n, z, random_state=rng)
  p_zero = sparse.csc_matrix((n, n))

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True,
      verbose=False,
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_unbounded(solution, a, c, p_zero, z)


# =============================================================================
# p=None equivalence: explicit zero P should give the same result as p=None
# =============================================================================

def test_p_none_equivalent_to_zero_matrix():
  """Test that p=None and p=zeros give identical solutions."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 3
  a, b, c, _ = _gen_feasible(m, n, z, random_state=rng)
  p_zero = sparse.csc_matrix((n, n))

  sol_none = qtqp.QTQP(a=a, b=b, c=c, z=z, p=None).solve(verbose=False)
  sol_zero = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p_zero).solve(verbose=False)

  assert sol_none.status == qtqp.SolutionStatus.SOLVED
  assert sol_zero.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(sol_none.x, sol_zero.x, atol=1e-8, rtol=1e-8)
  np.testing.assert_allclose(sol_none.y, sol_zero.y, atol=1e-8, rtol=1e-8)


# =============================================================================
# All-inequality constraints (z=0)
# =============================================================================

@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 4342 + np.arange(5))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_solve_all_inequalities(equilibrate, seed, linear_solver, record_iterations):
  """Test solver with z=0 (all-inequality constraints, no equalities)."""
  rng = np.random.default_rng(seed)
  m, n, z = 50, 30, 0
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True,
      verbose=False,
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# Small-problem infeasible / unbounded
# =============================================================================

@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 4442 + np.arange(5))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_infeasible_small(equilibrate, seed, linear_solver, record_iterations):
  """Test infeasible detection on small problems (n+m < ~50)."""
  rng = np.random.default_rng(seed)
  m, n, z = 20, 10, 3
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True,
      verbose=False,
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_infeasible(solution, a, b, z)


@pytest.mark.parametrize('equilibrate', [True, False])
@pytest.mark.parametrize('seed', 4542 + np.arange(5))
@pytest.mark.parametrize('linear_solver', _SOLVERS)
def test_unbounded_small(equilibrate, seed, linear_solver, record_iterations):
  """Test unbounded detection on small problems (n+m < ~50)."""
  rng = np.random.default_rng(seed)
  m, n, z = 20, 10, 3
  a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=equilibrate, linear_solver=linear_solver, collect_stats=True,
      verbose=False,
  )

  record_iterations(solution.stats[-1]['iter'], solution.stats[-1]['time'])
  _assert_unbounded(solution, a, c, p, z)


# =============================================================================
# SolutionStatus.FAILED
# =============================================================================

def test_failed_status():
  """Test that FAILED is returned when max_iter is exhausted."""
  rng = np.random.default_rng(42)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      max_iter=1, verbose=False
  )

  assert solution.status == qtqp.SolutionStatus.FAILED
  # FAILED still returns a finite best-effort iterate (not NaN).
  assert np.all(np.isfinite(solution.x))
  assert np.all(np.isfinite(solution.y))
  assert np.all(np.isfinite(solution.s))


# =============================================================================
# collect_stats=False (default)
# =============================================================================

def test_collect_stats_false():
  """Test that stats is empty when collect_stats=False (default)."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, collect_stats=False
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  assert solution.stats == []


# =============================================================================
# Stats content when collect_stats=True
# =============================================================================

def test_stats_keys():
  """Test that collect_stats=True populates the expected per-iteration keys."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, collect_stats=True
  )

  assert len(solution.stats) > 0
  base_keys = {
      'iter', 'pcost', 'dcost', 'pres', 'dres', 'gap', 'mu', 'sigma',
      'alpha', 'tau', 'norm_x', 'norm_y', 'status', 'time',
      'prelrhs', 'drelrhs', 'pinfeas', 'dinfeas',
  }
  collect_stats_keys = {
      'complementarity', 'norm_s',
      'max_sy', 'min_sy', 'std_sy',
      'max_s_over_y', 'min_s_over_y', 'mean_s_over_y', 'std_s_over_y',
  }
  for stats_i in solution.stats:
    missing = base_keys - stats_i.keys()
    assert not missing, f"Missing base keys: {missing}"
    missing = collect_stats_keys - stats_i.keys()
    assert not missing, f"Missing collect_stats keys: {missing}"


# =============================================================================
# Re-solve: calling solve() twice on the same instance
# =============================================================================

def test_resolve():
  """Test that calling solve() twice on the same QTQP instance is consistent."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 10, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)

  sol1 = solver.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)
  sol2 = solver.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY_DENSE)

  assert sol1.status == qtqp.SolutionStatus.SOLVED
  assert sol2.status == qtqp.SolutionStatus.SOLVED
  _assert_solution(sol1, a, b, c, p, z)
  _assert_solution(sol2, a, b, c, p, z)
  obj1 = c @ sol1.x + 0.5 * sol1.x @ p @ sol1.x
  obj2 = c @ sol2.x + 0.5 * sol2.x @ p @ sol2.x
  np.testing.assert_allclose(obj1, obj2, atol=1e-5, rtol=1e-5)


# =============================================================================
# verbose=False produces no output
# =============================================================================

def test_verbose_false(capsys):
  """Test that verbose=False suppresses all printed output."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)

  captured = capsys.readouterr()
  assert captured.out == ""


# =============================================================================
# Non-CSC input raises TypeError
# =============================================================================

def test_raise_error_non_csc_matrix():
  """Test that TypeError is raised when a or p are not in CSC format."""
  rng = np.random.default_rng(42)
  m, n, z = 6, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  with pytest.raises(TypeError):
    qtqp.QTQP(a=a.tocsr(), b=b, c=c, z=z, p=p)

  with pytest.raises(TypeError):
    qtqp.QTQP(a=a, b=b, c=c, z=z, p=p.tocsr())


# =============================================================================
# Known solution (README example)
# =============================================================================

def test_known_solution():
  """Test against the README example with a known optimal solution."""
  p = sparse.csc_matrix([[3.0, -1.0], [-1.0, 2.0]])
  a = sparse.csc_matrix([[-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
  b = np.array([-1.0, 0.3, -0.5])
  c = np.array([-1.0, -1.0])
  z = 1

  solution = qtqp.QTQP(p=p, a=a, b=b, c=c, z=z).solve(verbose=False)

  assert solution.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, [0.3, -0.7], atol=1e-5, rtol=1e-5)


# =============================================================================
# Single inequality constraint (z = m-1)
# =============================================================================

@pytest.mark.parametrize('seed', 5042 + np.arange(5))
def test_single_inequality_constraint(seed):
  """Test with z=m-1 (one inequality, rest equalities) — boundary of valid z."""
  rng = np.random.default_rng(seed)
  m, n = 20, 10
  z = m - 1
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)

  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# LP with all-inequality constraints (p=None, z=0)
# =============================================================================

@pytest.mark.parametrize('seed', 5142 + np.arange(5))
def test_solve_lp_all_inequalities(seed):
  """Test LP (p=None) with z=0 — exercises both LP and all-inequality paths."""
  rng = np.random.default_rng(seed)
  m, n, z = 30, 20, 0
  a, b, c, _ = _gen_feasible(m, n, z, random_state=rng)
  p_zero = sparse.csc_matrix((n, n))

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z).solve(verbose=False)

  _assert_solution(solution, a, b, c, p_zero, z)


# =============================================================================
# verbose=True produces expected output
# =============================================================================

def test_verbose_true(capsys):
  """Test that verbose=True prints a header, iteration rows, and a footer."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=True)

  out = capsys.readouterr().out
  assert "QTQP" in out       # header line with version/dimensions
  assert "iter" in out       # column header
  assert "Solved" in out     # footer


# =============================================================================
# Stats: iteration counter is sequential and final status is correct
# =============================================================================

def test_stats_iter_sequence():
  """Test that iter counts 0,1,2,... and final status matches solution status."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, collect_stats=True
  )

  iters = [s['iter'] for s in solution.stats]
  assert iters == list(range(len(iters))), "iter field should be 0,1,2,..."
  assert solution.stats[-1]['status'] == qtqp.SolutionStatus.SOLVED


# =============================================================================
# _max_step_size unit tests
# =============================================================================

def test_max_step_size():
  """Unit tests for _max_step_size boundary cases."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)

  y = rng.uniform(0.1, 1.0, size=m)

  # All non-negative directions: no variable decreases → step = 1.0.
  delta_pos = rng.uniform(0.0, 1.0, size=m)
  assert solver._max_step_size(y, delta_pos) == 1.0  # pylint: disable=protected-access

  # Direction that limits step to exactly 0.5.
  delta = np.zeros(m)
  delta[0] = -2 * y[0]  # step of 0.5 brings y[0] to zero
  alpha = solver._max_step_size(y, delta)  # pylint: disable=protected-access
  np.testing.assert_allclose(alpha, 0.5, atol=1e-12)

  # Step capped at 1.0 even when the unconstrained step would be larger.
  delta_small = np.zeros(m)
  delta_small[0] = -0.01 * y[0]  # only drives y[0] to 0 at step=100
  alpha_capped = solver._max_step_size(y, delta_small)  # pylint: disable=protected-access
  assert alpha_capped == 1.0


# =============================================================================
# Equilibration/unequilibration roundtrip
# =============================================================================

def test_equilibrate_unequilibrate_roundtrip():
  """Test that equilibrating then unequilibrating iterates is the identity."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)

  _, _, _, _, solver.d, solver.e = solver._equilibrate()  # pylint: disable=protected-access

  x = rng.normal(size=n)
  y = rng.uniform(size=m)
  s = rng.uniform(size=m)

  x_eq, y_eq, s_eq = solver._equilibrate_iterates(x, y, s)  # pylint: disable=protected-access
  x_rec, y_rec, s_rec = solver._unequilibrate_iterates(x_eq, y_eq, s_eq)  # pylint: disable=protected-access

  np.testing.assert_allclose(x_rec, x, atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(y_rec, y, atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(s_rec, s, atol=1e-12, rtol=1e-12)


# =============================================================================
# _normalize invariant
# =============================================================================

def test_normalize_invariant():
  """Test that _normalize enforces ||(x,y,tau)||^2 == m - z + 1."""
  rng = np.random.default_rng(42)
  m, n, z = 20, 10, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)

  x = rng.normal(size=n)
  y = rng.normal(size=m)
  tau = np.array([rng.uniform(0.1, 2.0)])
  s = rng.uniform(size=m)

  x_n, y_n, tau_n, _ = solver._normalize(x, y, tau, s)  # pylint: disable=protected-access

  xyt_norm_sq = x_n @ x_n + y_n @ y_n + tau_n @ tau_n
  np.testing.assert_allclose(xyt_norm_sq, m - z + 1, atol=1e-12, rtol=1e-12)


# =============================================================================
# Looser tolerances require fewer iterations
# =============================================================================

def test_tolerance_effect_on_iterations():
  """Test that looser tolerances converge in fewer iterations."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  sol_loose = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      atol=1e-3, rtol=1e-4, verbose=False, collect_stats=True
  )
  sol_tight = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      atol=1e-7, rtol=1e-8, verbose=False, collect_stats=True
  )

  assert sol_loose.status == qtqp.SolutionStatus.SOLVED
  assert sol_tight.status == qtqp.SolutionStatus.SOLVED
  assert len(sol_loose.stats) <= len(sol_tight.stats)


# =============================================================================
# Verbose footer messages for all non-SOLVED statuses
# =============================================================================

def test_verbose_infeasible(capsys):
  """Test that verbose=True prints the correct footer for infeasible problems."""
  rng = np.random.default_rng(142)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)

  qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=True)

  out = capsys.readouterr().out
  assert "Primal infeasible" in out


def test_verbose_unbounded(capsys):
  """Test that verbose=True prints the correct footer for unbounded problems."""
  rng = np.random.default_rng(242)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)

  qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=True)

  out = capsys.readouterr().out
  assert "Dual infeasible" in out


def test_verbose_failed(capsys):
  """Test that verbose=True prints the correct footer when max_iter is hit."""
  rng = np.random.default_rng(42)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=True, max_iter=1)

  out = capsys.readouterr().out
  assert "Failed to converge" in out


# =============================================================================
# max_iterative_refinement_steps=1 (single linear solve per Newton step)
# =============================================================================

def test_min_iterative_refinement_steps():
  """Test that the solver converges with just one iterative refinement step."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      max_iterative_refinement_steps=1, verbose=False
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# Very small problem — designed use case for ScipyDenseSolver
# =============================================================================

def test_solve_tiny():
  """Test SCIPY_DENSE on a tiny (5×3) problem — its primary intended use case."""
  rng = np.random.default_rng(42)
  m, n, z = 5, 3, 1
  # Use a dense random matrix so the tiny problem is non-degenerate.
  a = sparse.csc_matrix(rng.normal(size=(m, n)))
  p_dense = rng.normal(size=(n, n))
  p = sparse.csc_matrix(p_dense.T @ p_dense * 0.1)
  w = rng.normal(size=m)
  y = w.copy()
  y[z:] = 0.5 * (w[z:] + np.abs(w[z:]))
  s = y - w
  x = rng.normal(size=n)
  b = np.array(a @ x + s).ravel()
  c = np.array(-a.T @ y).ravel()

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      linear_solver=qtqp.LinearSolver.SCIPY_DENSE, verbose=False
  )

  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# equilibrate=True and equilibrate=False produce the same solution
# =============================================================================

def test_equilibrate_same_solution():
  """Test that equilibration does not change the optimal objective value."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  sol_eq = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=True, verbose=False
  )
  sol_noeq = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      equilibrate=False, verbose=False
  )

  assert sol_eq.status == qtqp.SolutionStatus.SOLVED
  assert sol_noeq.status == qtqp.SolutionStatus.SOLVED
  # Both solutions must satisfy KKT independently.
  _assert_solution(sol_eq, a, b, c, p, z)
  _assert_solution(sol_noeq, a, b, c, p, z)
  # Optimal objective values must agree (even if x differs near degeneracy).
  obj_eq = c @ sol_eq.x + 0.5 * sol_eq.x @ p @ sol_eq.x
  obj_noeq = c @ sol_noeq.x + 0.5 * sol_noeq.x @ p @ sol_noeq.x
  np.testing.assert_allclose(obj_eq, obj_noeq, atol=1e-5, rtol=1e-5)


# =============================================================================
# Re-solve with flipped equilibrate setting
# =============================================================================

def test_resolve_flip_equilibrate():
  """Test re-solving the same instance with different equilibrate settings."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
  solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)

  sol1 = solver.solve(equilibrate=True, verbose=False)
  sol2 = solver.solve(equilibrate=False, verbose=False)

  assert sol1.status == qtqp.SolutionStatus.SOLVED
  assert sol2.status == qtqp.SolutionStatus.SOLVED
  _assert_solution(sol1, a, b, c, p, z)
  _assert_solution(sol2, a, b, c, p, z)
  obj1 = c @ sol1.x + 0.5 * sol1.x @ p @ sol1.x
  obj2 = c @ sol2.x + 0.5 * sol2.x @ p @ sol2.x
  np.testing.assert_allclose(obj1, obj2, atol=1e-5, rtol=1e-5)


# =============================================================================
# complementarity is near zero at convergence
# =============================================================================

def test_complementarity_at_convergence():
  """Test that complementarity (y.s / tau^2) is small at the solved iteration."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, collect_stats=True
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  complementarity = solution.stats[-1]['complementarity']
  assert complementarity < 1e-6, f"Complementarity {complementarity} too large at convergence"


# =============================================================================
# Iterative refinement: more steps produce a lower (or equal) residual
# =============================================================================

def test_iterative_refinement_improves_residual():
  """Test that more refinement steps reduce the final linear system residual."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  sol_1 = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      max_iterative_refinement_steps=1, verbose=False, collect_stats=True
  )
  sol_50 = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      max_iterative_refinement_steps=50, verbose=False, collect_stats=True
  )

  # Compare the first iteration (cold start) q-solve residual.
  res_1 = sol_1.stats[0]['q_lin_sys_stats']['final_residual_norm']
  res_50 = sol_50.stats[0]['q_lin_sys_stats']['final_residual_norm']
  assert res_1 >= res_50, (
      f"1-step residual {res_1} should be >= 50-step residual {res_50}"
  )


# =============================================================================
# min_static_regularization: zero and large values both solve correctly
# =============================================================================

@pytest.mark.parametrize('reg', [0.0, 1e-4, 1e-2])
def test_min_static_regularization(reg):
  """Test that different min_static_regularization values still produce SOLVED."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      min_static_regularization=reg, verbose=False
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# Bug fix: z < 0 should raise ValueError (negative indexing would corrupt state)
# =============================================================================

def test_raise_error_negative_z():
  """Test that z < 0 raises ValueError (prevents silent negative-indexing bugs)."""
  rng = np.random.default_rng(42)
  m, n, z = 10, 5, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  with pytest.raises(ValueError):
    qtqp.QTQP(a=a, b=b, c=c, z=-1, p=p)


# =============================================================================
# Stats monotonicity: mu decreases, time increases, alpha in (0, 1]
# =============================================================================

def test_stats_monotonicity():
  """Test that mu decreases and time increases across iterations."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, collect_stats=True
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  assert len(solution.stats) >= 2

  mus = [s['mu'] for s in solution.stats]
  times = [s['time'] for s in solution.stats]
  alphas = [s['alpha'] for s in solution.stats]

  # mu should be strictly decreasing for a well-conditioned solved problem.
  for i in range(1, len(mus)):
    assert mus[i] < mus[i - 1], (
        f"mu not decreasing: mu[{i}]={mus[i]} >= mu[{i-1}]={mus[i-1]}"
    )

  # Time should be monotonically non-decreasing.
  for i in range(1, len(times)):
    assert times[i] >= times[i - 1], (
        f"time not increasing: time[{i}]={times[i]} < time[{i-1}]={times[i-1]}"
    )

  # alpha should be in (0, 1] at every iteration.
  for i, a_val in enumerate(alphas):
    assert 0 < a_val <= 1.0, f"alpha[{i}]={a_val} not in (0, 1]"


# =============================================================================
# Solution shape correctness for every status
# =============================================================================

@pytest.mark.parametrize('status_type', ['solved', 'infeasible', 'unbounded', 'failed'])
def test_solution_shapes(status_type):
  """Test that solution x, y, s have the correct shapes for every status."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5

  if status_type == 'solved':
    a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
    sol = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  elif status_type == 'infeasible':
    a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)
    sol = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  elif status_type == 'unbounded':
    a, b, c, p = _gen_unbounded(m, n, z, random_state=rng)
    sol = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  elif status_type == 'failed':
    a, b, c, p = _gen_feasible(m, n, z, random_state=rng)
    sol = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False, max_iter=1)

  assert sol.x.shape == (n,)
  assert sol.y.shape == (m,)
  assert sol.s.shape == (m,)


# =============================================================================
# step_size_scale effect
# =============================================================================

@pytest.mark.parametrize('scale', [0.5, 0.9, 0.99])
def test_step_size_scale(scale):
  """Test that different step_size_scale values produce a valid solution."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      step_size_scale=scale, verbose=False
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# atol_infeas / rtol_infeas parameters
# =============================================================================

@pytest.mark.parametrize('atol_infeas,rtol_infeas', [(1e-4, 1e-5), (1e-10, 1e-11)])
def test_infeasibility_tolerances(atol_infeas, rtol_infeas):
  """Test that infeasibility detection works with different tolerances."""
  rng = np.random.default_rng(142)
  m, n, z = 150, 100, 10
  a, b, c, p = _gen_infeasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      atol_infeas=atol_infeas, rtol_infeas=rtol_infeas, verbose=False
  )

  assert solution.status == qtqp.SolutionStatus.INFEASIBLE
  _assert_infeasible(solution, a, b, z, atol=atol_infeas, rtol=rtol_infeas)


# =============================================================================
# Smallest valid problem: m=2, z=0, n=1 (two inequalities, one variable)
# =============================================================================

def test_solve_minimal():
  """Test the smallest valid problem: min (1/2)x^2 - x s.t. 0 <= x <= 2."""
  a = sparse.csc_matrix([[-1.0], [1.0]])
  b = np.array([0.0, 2.0])
  c = np.array([-1.0])
  p = sparse.csc_matrix([[1.0]])
  z = 0

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)

  assert solution.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, [1.0], atol=1e-5)


# =============================================================================
# Residuals decrease across iterations
# =============================================================================

def test_residuals_decrease():
  """Test that primal/dual residuals and gap decrease over the solve."""
  rng = np.random.default_rng(42)
  m, n, z = 50, 30, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      verbose=False, collect_stats=True
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  assert len(solution.stats) >= 3

  first = solution.stats[0]
  last = solution.stats[-1]
  assert last['pres'] < first['pres'], "Primal residual did not decrease"
  assert last['dres'] < first['dres'], "Dual residual did not decrease"
  assert last['gap'] < first['gap'], "Gap did not decrease"


# =============================================================================
# linear_solver_atol / linear_solver_rtol parameters
# =============================================================================

def test_linear_solver_tolerances():
  """Test that different linear solver tolerances still produce SOLVED."""
  rng = np.random.default_rng(42)
  m, n, z = 30, 20, 5
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  solution = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(
      linear_solver_atol=1e-6, linear_solver_rtol=1e-6, verbose=False
  )

  assert solution.status == qtqp.SolutionStatus.SOLVED
  _assert_solution(solution, a, b, c, p, z)


# =============================================================================
# Non-symmetric P handling
# =============================================================================

def test_nonsymmetric_p():
  """Test solver behavior with a non-symmetric P (upper triangle only).

  Users are expected to provide symmetric P, but verify the solver doesn't
  crash with the upper triangle alone.
  """
  rng = np.random.default_rng(42)
  m, n, z = 20, 10, 3
  a, b, c, p = _gen_feasible(m, n, z, random_state=rng)

  p_triu = sparse.triu(p, format='csc')

  sol_sym = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p).solve(verbose=False)
  sol_triu = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p_triu).solve(verbose=False)

  assert sol_sym.status == qtqp.SolutionStatus.SOLVED
  # Upper-triangle-only P may or may not converge to the same optimum
  # (it defines a different KKT). Just verify it doesn't crash.
  assert sol_triu.status in (
      qtqp.SolutionStatus.SOLVED,
      qtqp.SolutionStatus.FAILED,
  )
