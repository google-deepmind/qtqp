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
"""Regression tests for backend ownership during QP setup."""

from unittest import mock

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _problem():
  return qtqp.QTQP(
      a=sparse.csc_matrix([[1.0], [-1.0]]),
      b=np.array([1.0, 0.0]),
      c=np.array([-1.0]),
      z=0,
  )


@pytest.mark.parametrize("phase", ["equilibrate", "set_dims", "format", "set_kkt"])
def test_setup_failure_releases_backend(monkeypatch, phase):
  problem = _problem()
  backend = mock.Mock(wraps=qtqp.solvers_sparse.ScipySolver())
  monkeypatch.setattr(
      qtqp, "_resolve_linear_solver",
      lambda _: (qtqp.LinearSolver.SCIPY, backend),
  )
  error = RuntimeError("setup failed")
  if phase == "equilibrate":
    monkeypatch.setattr(problem, "_equilibrate", mock.Mock(side_effect=error))
  else:
    getattr(backend, phase).side_effect = error

  with pytest.raises(RuntimeError, match="setup failed") as raised:
    problem.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)

  assert raised.value is error
  backend.free.assert_called_once_with()
  assert problem._linear_solver is None


def test_successful_solve_releases_backend_once(monkeypatch):
  problem = _problem()
  backend = mock.Mock(wraps=qtqp.solvers_sparse.ScipySolver())
  monkeypatch.setattr(
      qtqp, "_resolve_linear_solver",
      lambda _: (qtqp.LinearSolver.SCIPY, backend),
  )
  solution = problem.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)
  assert solution.status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, [1.0], atol=1e-7)
  backend.free.assert_called_once_with()
  assert problem._linear_solver is None
