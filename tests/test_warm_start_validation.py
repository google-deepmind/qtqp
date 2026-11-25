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

"""Warm-start validation before backend allocation."""

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


def _warm_start():
  return (np.array([0.5]), np.ones(2), np.ones(2))


@pytest.mark.parametrize("field", [0, 1, 2])
def test_complex_warm_start_rejected_before_backend_creation(field):
  problem = _problem()
  warm = list(_warm_start())
  warm[field] = warm[field].astype(complex) + 1j
  with mock.patch.object(
      qtqp, "_resolve_linear_solver", wraps=qtqp._resolve_linear_solver
  ) as resolve:
    with pytest.raises(ValueError, match="warm_start.*real"):
      problem.solve(
          verbose=False, linear_solver=qtqp.LinearSolver.SCIPY,
          warm_start=tuple(warm),
      )
    resolve.assert_not_called()


@pytest.mark.parametrize(
    "warm_start,threshold",
    [
        ((np.zeros(2), np.ones(2), np.ones(2)), 100.0),
        ((np.array([np.nan]), np.ones(2), np.ones(2)), 100.0),
        ((np.zeros(1), np.ones(2)), 100.0),
        (_warm_start(), 0.0),
        (_warm_start(), np.inf),
    ],
)
def test_invalid_warm_start_rejected_before_backend_creation(warm_start, threshold):
  with mock.patch.object(
      qtqp, "_resolve_linear_solver", wraps=qtqp._resolve_linear_solver
  ) as resolve:
    with pytest.raises(ValueError, match="warm_start"):
      _problem().solve(
          verbose=False, linear_solver=qtqp.LinearSolver.SCIPY,
          warm_start=warm_start, warm_start_threshold=threshold
      )
    resolve.assert_not_called()


def test_valid_warm_start_and_unused_threshold_preserved():
  problem = _problem()
  original = _warm_start()
  snapshots = tuple(value.copy() for value in original)
  for options in (
      {"warm_start": original},
      {"warm_start": None, "warm_start_threshold": np.nan},
  ):
    sol = problem.solve(
        verbose=False, linear_solver=qtqp.LinearSolver.SCIPY, **options
    )
    assert sol.status is qtqp.SolutionStatus.SOLVED
    np.testing.assert_allclose(sol.x, [1.0], atol=1e-7)
  for value, snapshot in zip(original, snapshots):
    np.testing.assert_array_equal(value, snapshot)
