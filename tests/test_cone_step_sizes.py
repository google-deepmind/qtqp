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

"""Fraction-to-boundary steps must preserve the cone at every scale."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _problem():
  return qtqp.QTQP(
      a=sparse.csc_matrix([[1., 0.], [0., 1.], [0., -1.]]),
      b=np.array([0., 1., 1.]), c=np.zeros(2), z=1,
  )


@pytest.mark.parametrize("scale", [1e-30, 1e-18, 1e-16, 1e-15, 1., 1e15])
def test_step_is_invariant_under_positive_scaling(scale):
  y = scale * np.array([2., 1.])
  direction = scale * np.array([-1., -4.])
  step = _problem()._max_step_size(y, direction)
  assert step == 0.25
  assert np.all(y + 0.99 * step * direction > 0.)


def test_small_coordinate_limits_a_mixed_scale_step():
  y = np.array([1., 1e-18])
  direction = np.array([-0.1, -4e-18])
  assert _problem()._max_step_size(y, direction) == 0.25


@pytest.mark.parametrize("limited", ["slack", "dual"])
def test_combined_step_preserves_small_slacks_and_duals(limited):
  y = np.array([-100., 1., 1.])
  s = np.array([0., 1., 1.])
  dy = np.array([-100., 0., 0.])
  ds = np.array([-100., 0., 0.])
  value, direction = (s, ds) if limited == "slack" else (y, dy)
  value[1] = 1e-18
  direction[1] = -4e-18
  step = _problem()._compute_step_size(y, s, dy, ds)
  assert step == 0.25
  assert np.all(y[1:] + 0.99 * step * dy[1:] > 0.)
  assert np.all(s[1:] + 0.99 * step * ds[1:] > 0.)


def test_zero_coordinate_cannot_take_a_negative_step():
  assert _problem()._max_step_size(np.array([0.]), np.array([-1e-18])) == 0.


def test_nonbinding_directions_do_not_overflow_the_ratio():
  y = np.array([1e20, 1., 0.])
  direction = np.array([-1e-300, -0.5, 0.])
  with np.errstate(over="raise", divide="raise", invalid="raise"):
    assert _problem()._max_step_size(y, direction) == 1.


def test_ordinary_steps_match_the_direct_boundary_ratio():
  rng = np.random.default_rng(42)
  problem = _problem()
  for _ in range(100):
    y = rng.uniform(0.1, 10., 10)
    direction = rng.normal(size=10)
    decreasing = direction < 0.
    expected = min(1., np.min(-y[decreasing] / direction[decreasing], initial=1.))
    assert problem._max_step_size(y, direction) == expected


def test_public_solve_keeps_trial_steps_in_the_cone(monkeypatch):
  rng = np.random.default_rng(5)
  n = 4
  a = sparse.vstack([sparse.eye(n), -sparse.eye(n)], format="csc")
  b = np.concatenate([rng.uniform(0.5, 2., n) * 1e6, np.zeros(n)])
  c = -10.**rng.uniform(-6, 6, n)
  p = sparse.diags(10.**rng.uniform(-6, 6, n), format="csc")
  problem = qtqp.QTQP(a=a, b=b, c=c, p=p, z=0)
  original = problem._compute_step_size
  steps = []

  def checked_step(y, s, dy, ds):
    step = original(y, s, dy, ds)
    for values, direction in ((y, dy), (s, ds)):
      assert np.all(values + 0.99 * step * direction >= 0.)
    steps.append(step)
    return step

  monkeypatch.setattr(problem, "_compute_step_size", checked_step)
  solution = problem.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)
  assert steps
  assert solution.status is qtqp.SolutionStatus.SOLVED
