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

"""Termination decisions must not depend on overflow in a vector norm."""

import math

import numpy as np
import pytest
from scipy import sparse

import qtqp


@pytest.mark.parametrize('scale', [1e-300, 1e-200, 1.0, 1e154, 1e160, 1e300])
@pytest.mark.parametrize('strided', [False, True])
def test_vector_norm_matches_scaled_reference(scale, strided):
  vector = np.array([3.0, 0.0, -4.0, 0.0]) * scale
  if strided:
    vector = vector[::2]
  expected = math.hypot(*vector)
  actual = qtqp._norm(vector)
  assert math.isfinite(actual)
  assert actual == pytest.approx(expected, rel=2e-15, abs=0.0)
  assert qtqp._norm(vector, 2) == actual
  assert qtqp._norm(vector, np.inf) == np.max(np.abs(vector))


def _problem(a, b, c, z=0, p=None):
  problem = qtqp.QTQP(
      a=sparse.csc_matrix(a),
      b=np.asarray(b, dtype=float),
      c=np.asarray(c, dtype=float),
      z=z,
      p=None if p is None else sparse.csc_matrix(p),
  )
  # Initialize the public solver's tolerances and bookkeeping normally.
  problem.solve(
      verbose=False,
      max_iter=1,
      linear_solver=qtqp.LinearSolver.SCIPY,
      equilibration_strategy=qtqp.EquilibrationStrategy.NONE,
  )
  return problem


@pytest.mark.parametrize('scale', [1.0, 1e154, 1e155, 1e160])
@pytest.mark.parametrize('collect_stats', [False, True])
def test_invalid_primal_certificate_is_not_accepted_after_norm_overflow(
    scale, collect_stats
):
  problem = _problem([[1e-6]], [-1.0], [0.0])
  # Feasible witness: x=-2e6, s=1 satisfies A*x+s=b.
  np.testing.assert_array_equal(problem.a @ np.array([-2e6]) + [1.0], problem.b)
  stats = {}
  with np.errstate(over='ignore', invalid='ignore', under='ignore'):
    status = problem._check_termination(
        np.array([0.0]),
        np.array([scale]),
        1e-10,
        np.array([1 / scale]),
        1.0,
        1.0,
        1.0,
        stats,
        collect_stats,
    )
  assert stats['pinfeas'] == pytest.approx(1e-6)
  assert status is qtqp.SolutionStatus.UNFINISHED
  assert math.isfinite(stats['norm_y'])


@pytest.mark.parametrize('scale', [1.0, 1e155, 1e160])
def test_bounded_problem_rejects_invalid_dual_certificate_at_every_ray_scale(
    scale,
):
  # min -x subject to 1e-6*x <= 1 is bounded at x=1e6.
  problem = _problem([[1e-6]], [1.0], [-1.0])
  stats = {}
  with np.errstate(over='ignore', invalid='ignore', under='ignore'):
    status = problem._check_termination(
        np.array([scale]),
        np.array([1.0]),
        1e-10,
        np.array([1.0]),
        1.0,
        1.0,
        1.0,
        stats,
        False,
    )
  assert stats['dinfeas_a'] > problem.tol_infeas_rel
  assert status is qtqp.SolutionStatus.UNFINISHED
  assert math.isfinite(stats['norm_x'])


@pytest.mark.parametrize('scale', [1.0, 1e-200])
def test_nonzero_tiny_residuals_are_not_reported_as_exact_solutions(scale):
  problem = _problem([[1.0]], [scale], [0.0], z=1)
  problem.tol_feas = scale / 10
  stats = {}
  status = problem._check_termination(
      np.array([0.0]),
      np.array([0.0]),
      1.0,
      np.array([0.0]),
      1.0,
      0.0,
      0.0,
      stats,
      False,
  )
  assert stats['pres'] == pytest.approx(scale, abs=0.0)
  assert status is qtqp.SolutionStatus.UNFINISHED


@pytest.mark.parametrize('scale', [1.0, 1e155, 1e160])
def test_valid_primal_certificate_remains_accepted(scale):
  # Inconsistent x<=0 and x>=1, with dual witness y=(1,1).
  problem = _problem([[1.0], [-1.0]], [0.0, -1.0], [0.0])
  stats = {}
  with np.errstate(over='ignore', invalid='ignore', under='ignore'):
    status = problem._check_termination(
        np.array([0.0]),
        np.array([scale, scale]),
        1e-10,
        np.array([1 / scale, 1 / scale]),
        1.0,
        1.0,
        1.0,
        stats,
        False,
    )
  assert status is qtqp.SolutionStatus.INFEASIBLE
  assert stats['pinfeas'] == 0.0


@pytest.mark.parametrize('equilibration', list(qtqp.EquilibrationStrategy))
@pytest.mark.parametrize(
    'backend',
    [
        qtqp.LinearSolver.SCIPY,
        qtqp.LinearSolver.SCIPY_DENSE,
        qtqp.LinearSolver.QDLDL,
    ],
)
def test_public_solver_returns_the_known_bounded_solution(
    equilibration, backend
):
  if backend is qtqp.LinearSolver.QDLDL:
    pytest.importorskip('qdldl')
  problem = qtqp.QTQP(
      a=sparse.csc_matrix([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
      b=np.array([2.0, 3.0, 0.0, 0.0]),
      c=np.array([-1.0, -2.0]),
      z=0,
      p=sparse.eye(2, format='csc'),
  )
  original_a = problem.a.copy()
  solution = problem.solve(
      verbose=False,
      linear_solver=backend,
      equilibration_strategy=equilibration,
      collect_stats=True,
  )
  assert solution.status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, [1.0, 2.0], atol=1e-7)
  np.testing.assert_allclose(
      problem.a @ solution.x + solution.s, problem.b, atol=1e-7
  )
  np.testing.assert_allclose(
      problem.p @ solution.x + problem.a.T @ solution.y + problem.c,
      [0.0, 0.0],
      atol=1e-7,
  )
  np.testing.assert_array_equal(problem.a.toarray(), original_a.toarray())


def test_empty_zero_and_nonfinite_norm_controls():
  assert qtqp._norm(np.array([])) == 0.0
  assert qtqp._norm(np.zeros(4)) == 0.0
  assert math.isnan(qtqp._norm(np.array([np.nan])))
  assert math.isinf(qtqp._norm(np.array([np.inf])))
