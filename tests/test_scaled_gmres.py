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
"""GMRES must preserve finite solves across right-hand-side scales."""

import numpy as np
import pytest
import scipy.sparse as sp

import qtqp
from qtqp import direct, solvers_dense, solvers_sparse


@pytest.mark.parametrize("backend", [solvers_sparse.ScipySolver,
                                    solvers_dense.ScipyDenseSolver,
                                    solvers_sparse.QdldlSolver])
@pytest.mark.parametrize("scale", [1.0, 1e160, 1e-200])
@pytest.mark.parametrize("warm", [False, True])
def test_scaled_rhs_matches_independent_solve(backend, scale, warm):
  if backend is solvers_sparse.QdldlSolver:
    pytest.importorskip("qdldl")
  a = np.array([[1.0, 0.5], [-0.5, 1.0]])
  p = np.array([[2.0, 0.25], [0.25, 1.0]])
  solver = direct.DirectKktSolver(
      a=sp.csc_matrix(a), p=sp.csc_matrix(p), z=0,
      # Make the preconditioner inexact so multiple Arnoldi vectors are used.
      min_static_regularization=3.0,
      max_iterative_refinement_steps=8, atol=0.0, rtol=1e-12,
      solver=backend(), refinement_strategy=direct.RefinementStrategy.GMRES,
      gmres_restart=4)
  try:
    for mu in (0.1, 0.3):
      solver.update(mu=mu, s=np.array([1.0, 2.0]), y=np.ones(2))
      kkt = np.block([[p + mu * np.eye(2), a.T],
                      [a, -np.diag([1.0 + mu, 2.0 + mu])]])
      rhs_unit = np.array([2.0, -1.0, 3.0, 0.5])
      rhs = scale * rhs_unit
      start = scale * np.array([0.1, -0.2, 0.3, -0.4]) if warm else np.zeros(4)
      original_start = start.copy()
      result, stats = solver.solve(rhs, start)
      expected = np.linalg.solve(kkt, rhs_unit * [1.0, 1.0, -1.0, -1.0])
      # Scaling before comparison avoids an absolute tolerance hiding underflow.
      np.testing.assert_allclose(result / scale, expected, rtol=1e-11, atol=1e-12)
      np.testing.assert_array_equal(start, original_start)
      assert stats["converged"]
      assert 1 < stats["solves"] <= 8
      assert np.isfinite(stats["final_residual_norm"])
      residual = rhs_unit * [1.0, 1.0, -1.0, -1.0] - kkt @ (result / scale)
      assert np.linalg.norm(residual, np.inf) <= 3e-12
  finally:
    solver.free()


def test_zero_rhs_still_needs_no_refinement():
  solver = direct.DirectKktSolver(
      a=sp.eye(1, format="csc"), p=sp.eye(1, format="csc"), z=0,
      min_static_regularization=1e-8, max_iterative_refinement_steps=4,
      atol=0.0, rtol=1e-12, solver=solvers_sparse.ScipySolver(),
      refinement_strategy=direct.RefinementStrategy.GMRES)
  try:
    solver.update(mu=0.1, s=np.ones(1), y=np.ones(1))
    result, stats = solver.solve(np.zeros(2), np.zeros(2))
    np.testing.assert_array_equal(result, 0.0)
    assert stats["converged"] and stats["solves"] == 0
  finally:
    solver.free()


def test_extreme_scaling_does_not_forge_an_unbounded_certificate():
  """A bounded, feasible QP must not come back as an unboundedness claim.

  This is the public-API consequence of the overflowing two-norm. At 1e160
  the GMRES residual norm evaluated to infinity, the cycle bailed with the
  iterate unchanged, and the homogeneous embedding read the stalled result
  as an unboundedness certificate -- for a QP whose unconstrained minimum,
  1e160 * [1, 1], is feasible. Richardson refinement reported a numeric
  failure on the same data, so the two smoothers disagreed on whether the
  problem had a certificate at all.
  """
  scale = 1e160
  problem = qtqp.QTQP(
      a=sp.csc_matrix([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
      b=scale * np.array([1.0, 1.0, 0.0]),
      c=scale * np.array([-1.0, -1.0]),
      p=sp.eye(2, format="csc"), z=0,
  )
  for equilibration in qtqp.EquilibrationStrategy:
    statuses = {
        strategy: problem.solve(
            verbose=False, equilibration_strategy=equilibration,
            refinement_strategy=strategy,
        ).status
        for strategy in qtqp.RefinementStrategy
    }
    gmres = statuses[qtqp.RefinementStrategy.GMRES]
    # A certificate for a problem that has none is worse than no answer.
    assert gmres is not qtqp.SolutionStatus.UNBOUNDED
    assert gmres is not qtqp.SolutionStatus.INFEASIBLE
    # And GMRES must not part company with the classical smoother here.
    assert gmres is statuses[qtqp.RefinementStrategy.RICHARDSON]
