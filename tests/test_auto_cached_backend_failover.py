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

"""AUTO can reselect a backend if the previously cached one fails to load."""

import pytest

import qtqp


@pytest.mark.parametrize("failure", [ImportError, OSError])
def test_auto_recovers_from_unavailable_cached_solver(monkeypatch, failure):
  monkeypatch.setattr(qtqp.sys, "platform", "linux")
  monkeypatch.setattr(
      qtqp, "_AUTO_SOLVER_CACHE", {"linux": qtqp.LinearSolver.PARDISO}
  )
  chosen = qtqp.solvers_sparse.ScipySolver()
  attempts = []

  def instantiate(candidate):
    attempts.append(candidate)
    if candidate is qtqp.LinearSolver.PARDISO:
      raise failure("cached optional solver unavailable")
    if candidate is qtqp.LinearSolver.CHOLMOD:
      return chosen
    raise AssertionError(f"Unexpected backend {candidate.name}")

  monkeypatch.setattr(qtqp, "_instantiate_linear_solver", instantiate)
  resolved, backend = qtqp._resolve_linear_solver(qtqp.LinearSolver.AUTO)

  assert resolved is qtqp.LinearSolver.CHOLMOD
  assert backend is chosen
  assert qtqp._AUTO_SOLVER_CACHE["linux"] is qtqp.LinearSolver.CHOLMOD
  assert attempts == [
      qtqp.LinearSolver.PARDISO,
      qtqp.LinearSolver.CHOLMOD,
  ]

  attempts.clear()
  resolved_again, backend_again = qtqp._resolve_linear_solver(
      qtqp.LinearSolver.AUTO
  )
  assert resolved_again is qtqp.LinearSolver.CHOLMOD
  assert backend_again is chosen
  assert attempts == [qtqp.LinearSolver.CHOLMOD]


def test_explicit_backend_still_propagates_import_failure(monkeypatch):
  monkeypatch.setattr(qtqp.sys, "platform", "linux")
  monkeypatch.setattr(
      qtqp, "_AUTO_SOLVER_CACHE", {"linux": qtqp.LinearSolver.PARDISO}
  )

  def unavailable(candidate):
    assert candidate is qtqp.LinearSolver.PARDISO
    raise ImportError("pymklpardiso unavailable")

  monkeypatch.setattr(qtqp, "_instantiate_linear_solver", unavailable)
  with pytest.raises(ImportError, match="pymklpardiso"):
    qtqp._resolve_linear_solver(qtqp.LinearSolver.PARDISO)


def test_auto_without_a_cached_solver_preserves_preference_order(monkeypatch):
  monkeypatch.setattr(qtqp.sys, "platform", "linux")
  monkeypatch.setattr(qtqp, "_AUTO_SOLVER_CACHE", {})
  chosen = qtqp.solvers_sparse.ScipySolver()
  attempts = []

  def instantiate(candidate):
    attempts.append(candidate)
    if candidate is qtqp.LinearSolver.SCIPY:
      return chosen
    raise ImportError(candidate.name)

  monkeypatch.setattr(qtqp, "_instantiate_linear_solver", instantiate)
  resolved, backend = qtqp._resolve_linear_solver(qtqp.LinearSolver.AUTO)

  assert resolved is qtqp.LinearSolver.SCIPY
  assert backend is chosen
  assert attempts[0] is qtqp.LinearSolver.PARDISO
  assert attempts[-1] is qtqp.LinearSolver.SCIPY


def test_auto_falls_back_and_solves_after_cached_backend_disappears(monkeypatch):
  """The public solve path must recover, not only the resolver alone."""
  import numpy as np
  from scipy import sparse

  monkeypatch.setattr(qtqp.sys, "platform", "darwin")
  monkeypatch.setattr(
      qtqp, "_AUTO_SOLVER_CACHE", {"darwin": qtqp.LinearSolver.ACCELERATE}
  )
  attempts = []

  def instantiate(candidate):
    attempts.append(candidate)
    if candidate is qtqp.LinearSolver.SCIPY:
      return qtqp.solvers_sparse.ScipySolver()
    raise OSError(f"{candidate.name} unavailable")

  monkeypatch.setattr(qtqp, "_instantiate_linear_solver", instantiate)
  problem = qtqp.QTQP(
      a=sparse.csc_matrix([
          [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]
      ]),
      b=np.array([1.0, 1.0, 0.0, 0.0]),
      c=np.array([-1.0, -1.0]),
      z=0,
  )
  solution = problem.solve(verbose=False, linear_solver=qtqp.LinearSolver.AUTO)
  assert solution.status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, [1.0, 1.0], atol=1e-6)
  assert attempts.count(qtqp.LinearSolver.ACCELERATE) == 1
  assert attempts[-1] is qtqp.LinearSolver.SCIPY
  assert qtqp._AUTO_SOLVER_CACHE["darwin"] is qtqp.LinearSolver.SCIPY
