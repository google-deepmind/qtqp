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

"""Iteration counts and equality-row counts must have integer types."""

from unittest import mock

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _data():
  return {
      "a": sparse.csc_matrix([[1.0], [-1.0]]),
      "b": np.array([1.0, 0.0]),
      "c": np.array([-1.0]),
      "z": 0,
  }


@pytest.mark.parametrize("invalid_z", [0.0, 0.5, True, np.float64(0.0)])
def test_equality_row_count_rejects_nonintegers(invalid_z):
  data = _data()
  data["z"] = invalid_z
  with pytest.raises(ValueError, match="z.*integer"):
    qtqp.QTQP(**data)


@pytest.mark.parametrize(
    "name,value", [
        ("max_iter", 2.5),
        ("max_iter", True),
        ("max_iterative_refinement_steps", 1.5),
        ("max_iterative_refinement_steps", True),
        ("max_centrality_correctors", 1.9),
        ("max_centrality_correctors", True),
        ("gmres_restart", 1.5),
        ("gmres_restart", True),
    ],
)
def test_iteration_counts_reject_nonintegers_before_solver_setup(name, value):
  problem = qtqp.QTQP(**_data())
  with mock.patch.object(qtqp, "_resolve_linear_solver") as backend:
    with pytest.raises(ValueError, match=name + ".*integer"):
      problem.solve(verbose=False, **{name: value})
    backend.assert_not_called()


def test_numpy_integer_counts_work_with_gmres():
  data = _data()
  data["z"] = np.int64(0)
  problem = qtqp.QTQP(**data)
  result = problem.solve(
      verbose=False,
      linear_solver=qtqp.LinearSolver.SCIPY,
      max_iter=np.int64(80),
      max_iterative_refinement_steps=np.int64(5),
      max_centrality_correctors=np.int64(1),
      gmres_restart=np.int64(3),
  )
  assert result.status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(result.x, [1.0], atol=1e-6)


def test_gmres_restart_remains_ignored_for_richardson():
  problem = qtqp.QTQP(**_data())
  result = problem.solve(
      verbose=False,
      linear_solver=qtqp.LinearSolver.SCIPY,
      refinement_strategy=qtqp.RefinementStrategy.RICHARDSON,
      gmres_restart=0,
  )
  assert result.status is qtqp.SolutionStatus.SOLVED


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__]))
