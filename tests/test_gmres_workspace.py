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

"""GMRES work arrays must fit the refinement budget that can consume them."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _solver(restart, budget, strategy=qtqp.RefinementStrategy.GMRES,
            min_static_regularization=1e-8):
  return qtqp.direct.DirectKktSolver(
      a=sparse.csc_matrix([[1., 0.], [0., 1.], [-1., -1.]]),
      p=sparse.eye(2, format="csc"), z=0,
      min_static_regularization=min_static_regularization,
      max_iterative_refinement_steps=budget, atol=1e-12, rtol=1e-12,
      solver=qtqp.LinearSolver.SCIPY.value(),
      refinement_strategy=strategy, gmres_restart=restart,
  )


@pytest.mark.parametrize("restart,budget", [(20, 1), (100, 2), (20, 3),
                                          (8, 8), (2, 8), (1, 20)])
def test_workspace_does_not_exceed_reachable_cycle(restart, budget):
  solver = _solver(restart, budget)
  k, dim = min(restart, budget), solver.n + solver.m
  assert solver._gm_v.shape == (k + 1, dim)
  assert solver._gm_z.shape == (k, dim)
  assert solver._gm_h.shape == (k + 1, k)
  assert solver._gm_cs.shape == solver._gm_sn.shape == (k,)
  assert solver._gm_g.shape == (k + 1,)
  # The requested setting remains available; only allocation is bounded.
  assert solver.gmres_restart == restart


@pytest.mark.parametrize("restart,budget", [(20, 1), (100, 2), (20, 3)])
def test_bounded_workspace_preserves_solves_and_is_reused(restart, budget):
  solver = _solver(restart, budget)
  reference = _solver(min(restart, budget), budget)
  buffers = tuple(getattr(solver, name) for name in
                  ("_gm_v", "_gm_z", "_gm_h", "_gm_cs", "_gm_sn", "_gm_g"))
  rhs = np.array([1., -2., 3., -4., 5.])
  try:
    for mu in (0.1, 0.01):
      for item in (solver, reference):
        item.update(mu=mu, s=np.ones(3), y=np.ones(3))
      actual, stats = solver.solve(rhs, np.zeros(5))
      expected, expected_stats = reference.solve(rhs, np.zeros(5))
      np.testing.assert_array_equal(actual, expected)
      assert stats == expected_stats
      assert stats["solves"] <= budget
      a = np.array([[1., 0.], [0., 1.], [-1., -1.]])
      kkt = np.block([[(1. + mu) * np.eye(2), a.T],
                      [a, -(1. + mu) * np.eye(3)]])
      signed_rhs = rhs * np.array([1., 1., -1., -1., -1.])
      np.testing.assert_allclose(kkt @ actual, signed_rhs, atol=1e-11)
      for name, buffer in zip(
          ("_gm_v", "_gm_z", "_gm_h", "_gm_cs", "_gm_sn", "_gm_g"), buffers):
        assert getattr(solver, name) is buffer
  finally:
    solver.free()
    reference.free()


@pytest.mark.parametrize("restart,budget", [(20, 2), (20, 3), (100, 4)])
def test_capped_workspace_spans_a_budget_consuming_refinement(restart, budget):
  """The cap must hold when refinement writes to every row it reserves.

  With the regularization clamp inactive the preconditioner inverts the
  operator exactly, so a cycle converges after one Arnoldi step and touches
  only the first row of each work array -- a workspace of one row would pass
  the shape-free checks above. Raising min_static_regularization above the
  true cone diagonal (s/y + mu) makes the preconditioner inexact, so the
  cycle consumes the whole budget and exercises the full allocation.
  """
  clamped = dict(min_static_regularization=1e-2)
  solver = _solver(restart, budget, **clamped)
  reference = _solver(min(restart, budget), budget, **clamped)
  rhs = np.array([1., -2., 3., -4., 5.])
  try:
    for item in (solver, reference):
      item.update(mu=1e-10, s=np.full(3, 1e-12), y=np.ones(3))
    actual, stats = solver.solve(rhs, np.zeros(5))
    expected, expected_stats = reference.solve(rhs, np.zeros(5))
    # The point of the fixture: refinement really does use every apply.
    assert stats["solves"] == budget
    np.testing.assert_array_equal(actual, expected)
    assert stats == expected_stats
  finally:
    solver.free()
    reference.free()


def test_richardson_does_not_allocate_gmres_workspace():
  solver = _solver(0, 2, qtqp.RefinementStrategy.RICHARDSON)
  assert not hasattr(solver, "_gm_v")
