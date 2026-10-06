# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Residual normalizers must not overflow before a representable division."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _problem(a, b, c, z=0):
  problem = qtqp.QTQP(a=sparse.csc_matrix(a), b=np.array(b), c=np.array(c), z=z)
  problem.solve(verbose=False, max_iter=1, linear_solver=qtqp.LinearSolver.SCIPY,
                equilibration_strategy=qtqp.EquilibrationStrategy.NONE)
  return problem


def _check(problem, x, y, s, tau=1.):
  stats = {}
  with np.errstate(over='ignore', invalid='ignore', under='ignore'):
    status = problem._check_termination(np.array(x), np.array(y), tau,
        np.array(s), 1., 0., 0., stats, False)
  return status, stats


@pytest.mark.parametrize('scale', [1., 1e150, 1e307, 1e308])
def test_infeasible_primal_iterate_cannot_pass_an_overflowed_normalizer(scale):
  problem = _problem([[.1]], [0.], [0.])
  status, stats = _check(problem, [scale], [.1 / scale], [scale])
  expected = 1.1 / max(1. / scale, 2.)
  assert stats['res_primal'] == pytest.approx(expected)
  assert status is qtqp.SolutionStatus.UNFINISHED
  # x=0 is feasible; the supplied x is strictly outside the half-space.
  assert (problem.a @ np.array([scale]))[0] > problem.b[0]


@pytest.mark.parametrize('scale', [1., 1e150, 1e307, 1e308])
def test_nonstationary_iterate_cannot_pass_an_overflowed_dual_normalizer(scale):
  problem = _problem([[0., .1]], [0.], [0., 0.], z=1)
  status, stats = _check(problem, [scale, 0.], [scale], [0.])
  assert stats['res_primal'] == 0.
  assert stats['res_dual'] == pytest.approx(.1 / max(1. / scale, 2.))
  assert status is qtqp.SolutionStatus.UNFINISHED


@pytest.mark.parametrize('scale', [1., 1e150, 1e307, 1e308])
def test_bounded_problem_does_not_accept_an_overflowed_recession_test(scale):
  # min -x, .1*x <= 1 is bounded at x=10, with dual optimum y=10.
  problem = _problem([[.1]], [1.], [-1.])
  status, stats = _check(problem, [scale], [1. / scale], [scale], tau=1e-10)
  assert stats['dinfeas_a'] == pytest.approx(.55)
  assert status is qtqp.SolutionStatus.UNFINISHED


def test_exact_stationary_solution_still_passes():
  problem = _problem([[.1]], [1.], [-1.])
  status, stats = _check(problem, [10.], [10.], [0.])
  assert status is qtqp.SolutionStatus.SOLVED
  assert stats['res_primal'] == stats['res_dual'] == 0.
