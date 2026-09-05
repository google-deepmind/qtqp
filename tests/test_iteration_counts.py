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
"""Completed IPM work is distinct from zero-based iteration labels."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _box_lp():
  return qtqp.QTQP(
      a=sparse.csc_matrix([[-1.0], [1.0]]),
      b=np.array([0.0, 1.0]),
      c=np.array([-1.0]),
      z=0,
  )


def _observe_grading(monkeypatch, fail_on_step=None):
  """Observe updated points independently of the reported counter.

  The first call grades initialization; later calls follow complete IPM
  updates. Optionally fail while grading a specified completed step.
  """
  points = []
  original = qtqp.QTQP._check_termination

  def observed(self, x, y, tau, s, *args, **kwargs):
    points.append((x.copy(), y.copy(), tau, s.copy()))
    if fail_on_step is not None and len(points) == fail_on_step + 1:
      raise RuntimeError("Forced failure while grading an updated point")
    return original(self, x, y, tau, s, *args, **kwargs)

  monkeypatch.setattr(qtqp.QTQP, "_check_termination", observed)
  return points


def _solve(qp, **options):
  return qp.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY, **options)


@pytest.mark.parametrize("collect_stats", [False, True])
def test_completed_steps_preserve_zero_based_labels(monkeypatch, collect_stats):
  points = _observe_grading(monkeypatch)
  solution = _solve(_box_lp(), collect_stats=collect_stats)

  assert solution.status is qtqp.SolutionStatus.SOLVED
  # This ordinary LP takes four complete updates, followed by four grades.
  assert len(points) - 1 == 4
  assert solution.iterations == len(points) - 1
  np.testing.assert_allclose(solution.x, [1.0], atol=1e-7)
  if collect_stats:
    assert [row["iter"] for row in solution.stats] == [0, 1, 2, 3]
    assert [row["iterations"] for row in solution.stats] == [1, 2, 3, 4]
  else:
    assert solution.stats == []


@pytest.mark.parametrize("collect_stats", [False, True])
def test_initial_solution_completes_zero_steps(monkeypatch, collect_stats):
  points = _observe_grading(monkeypatch)
  qp = qtqp.QTQP(
      a=sparse.csc_matrix([[1.0]]),
      b=np.array([1.0]),
      c=np.array([-1.0]),
      p=sparse.csc_matrix([[1.0]]),
      z=1,
  )
  solution = _solve(qp, collect_stats=collect_stats)

  assert solution.status is qtqp.SolutionStatus.SOLVED
  assert len(points) == 1
  assert solution.iterations == 0
  if collect_stats:
    assert len(solution.stats) == 1
    assert solution.stats[0]["iter"] == 0
    assert solution.stats[0]["iterations"] == 0
  else:
    assert solution.stats == []


@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize(
    ("max_iter", "expected_status"),
    [(2, qtqp.SolutionStatus.HIT_MAX_ITER),
     (3, qtqp.SolutionStatus.ALMOST_SOLVED)],
)
def test_iteration_cap_and_almost_return_completed_work(
    monkeypatch, collect_stats, max_iter, expected_status,
):
  points = _observe_grading(monkeypatch)
  solution = _solve(_box_lp(), collect_stats=collect_stats, max_iter=max_iter)

  assert solution.status is expected_status
  assert len(points) - 1 == max_iter
  assert solution.iterations == max_iter
  if collect_stats:
    assert solution.stats[-1]["iterations"] == max_iter
    assert solution.stats[-1]["iter"] == max_iter - 1


@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize("fail_on_attempt", [1, 3])
def test_unfinished_update_is_not_counted(
    monkeypatch, collect_stats, fail_on_attempt,
):
  points = _observe_grading(monkeypatch)
  original = qtqp.direct.DirectKktSolver.update
  attempts = []

  def failing_update(self, *args, **kwargs):
    attempts.append(None)
    if len(attempts) == fail_on_attempt:
      raise RuntimeError("Forced failure before the next iterate update")
    return original(self, *args, **kwargs)

  monkeypatch.setattr(qtqp.direct.DirectKktSolver, "update", failing_update)
  solution = _solve(_box_lp(), collect_stats=collect_stats)

  assert solution.status is qtqp.SolutionStatus.FAILED
  assert len(attempts) == fail_on_attempt
  assert len(points) - 1 == fail_on_attempt - 1
  assert solution.iterations == len(points) - 1
  if collect_stats:
    assert len(solution.stats) == solution.iterations


@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize("fail_on_step", [1, 2])
def test_updated_point_counts_even_when_its_grading_fails(
    monkeypatch, collect_stats, fail_on_step,
):
  points = _observe_grading(monkeypatch, fail_on_step=fail_on_step)
  solution = _solve(_box_lp(), collect_stats=collect_stats)

  assert solution.status is qtqp.SolutionStatus.FAILED
  assert len(points) - 1 == fail_on_step
  assert solution.iterations == fail_on_step
  if collect_stats:
    # The failed grade adds no stats row; earlier rows retain their counts.
    assert len(solution.stats) == fail_on_step - 1
    if solution.stats:
      assert solution.stats[-1]["iterations"] == fail_on_step - 1


def test_reusing_solver_resets_completed_step_count(monkeypatch):
  points = _observe_grading(monkeypatch)
  qp = _box_lp()
  first = _solve(qp, max_iter=2)
  assert first.iterations == len(points) - 1 == 2

  points.clear()
  second = _solve(qp)
  assert second.status is qtqp.SolutionStatus.SOLVED
  assert second.iterations == len(points) - 1 == 4
  assert first.iterations == 2


@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize(
    ("b", "c", "expected_status"),
    [([-1.0], [0.0], qtqp.SolutionStatus.INFEASIBLE),
     ([1.0], [-1.0], qtqp.SolutionStatus.UNBOUNDED)],
)
def test_certificate_returns_completed_step_count(
    monkeypatch, collect_stats, b, c, expected_status,
):
  points = _observe_grading(monkeypatch)
  qp = qtqp.QTQP(
      a=sparse.csc_matrix([[0.0]]), b=np.array(b), c=np.array(c), z=0,
  )
  solution = _solve(qp, collect_stats=collect_stats)

  assert solution.status is expected_status
  assert len(points) > 1
  assert solution.iterations == len(points) - 1
  if collect_stats:
    assert solution.stats[-1]["iterations"] == solution.iterations


def test_verbose_footer_reports_completed_steps(capsys):
  solution = _box_lp().solve(
      verbose=True, linear_solver=qtqp.LinearSolver.SCIPY,
  )
  assert solution.iterations == 4
  assert "Completed IPM steps: 4" in capsys.readouterr().out
