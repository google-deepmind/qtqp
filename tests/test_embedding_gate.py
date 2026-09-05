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

"""The embedding gate uses current complementarity in its operating frame."""

import importlib
import math

import numpy as np
import pytest
from scipy import sparse

import qtqp


try:
  importlib.import_module("qdldl")
except (ImportError, OSError):
  _QDLDL_AVAILABLE = False
else:
  _QDLDL_AVAILABLE = True

_BACKENDS = [
    qtqp.LinearSolver.SCIPY,
    qtqp.LinearSolver.SCIPY_DENSE,
    pytest.param(qtqp.LinearSolver.QDLDL, marks=pytest.mark.skipif(
        not _QDLDL_AVAILABLE, reason="Optional qdldl backend is unavailable",
    )),
]


def _bounded_qp():
  # Strictly convex and feasible: exact minimizer x = -c/P = 1e10.
  return qtqp.QTQP(
      a=sparse.csc_matrix([[0.0]]), b=np.array([1000.0]),
      c=np.array([-10000.0]), p=sparse.csc_matrix([[1e-6]]), z=0,
  )


@pytest.mark.parametrize("equilibration", list(qtqp.EquilibrationStrategy))
@pytest.mark.parametrize("backend", _BACKENDS)
def test_bounded_qp_does_not_become_an_unbounded_certificate(
    monkeypatch, equilibration, backend,
):
  qp = _bounded_qp()
  initial = []
  original = qp._check_termination

  def observe(x, y, tau, s, alpha, mu, sigma, stats, collect_stats):
    working_ratio = float(y @ s) / ((qp.m - qp.z) * tau * tau)
    status = original(x, y, tau, s, alpha, mu, sigma, stats, collect_stats)
    if not initial:
      initial.append((working_ratio, stats.copy(), status))
    return status

  monkeypatch.setattr(qp, "_check_termination", observe)
  solution = qp.solve(
      verbose=False, collect_stats=True, linear_solver=backend,
      equilibration_strategy=equilibration,
  )

  assert solution.status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(solution.x, [1e10], rtol=1e-8)
  np.testing.assert_allclose(solution.s, [1000.0], rtol=1e-6)
  if equilibration is qtqp.EquilibrationStrategy.RUIZ:
    # The scaled problem is P=1, c=-100, b=1. Its initial y=s=tau=1
    # gives ratio 1, preserved by homogeneous normalization. Returning
    # to the original frame multiplies complementarity by 1e10.
    ratio, stats, status = initial[0]
    assert qp.sigma_eq == pytest.approx(1e-3)
    assert qp.gamma_eq == pytest.approx(1e-4)
    assert ratio == pytest.approx(1.0)
    assert stats["ktratio"] == pytest.approx(1.0)
    assert stats["complementarity"] == pytest.approx(1e10)
    assert status is qtqp.SolutionStatus.UNFINISHED


@pytest.mark.parametrize(
    ("returned_y", "expected_ratio", "expected_status", "track_almost"),
    [(0.1, 1e-8, qtqp.SolutionStatus.SOLVED, True),
     (1000.0, 1e-4, qtqp.SolutionStatus.UNFINISHED, True),
     (1e7, 1.0, qtqp.SolutionStatus.UNFINISHED, False)],
)
def test_solution_certificate_and_almost_gates_share_working_ratio(
    returned_y, expected_ratio, expected_status, track_almost,
):
  qp = _bounded_qp()
  qp.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)
  qp._best_almost_score = math.inf
  qp._best_almost_iterate = None
  x, y, s = qp._equilibrate_iterates(
      np.array([1e10]), np.array([returned_y]), np.array([1000.0]),
  )
  stats = {}
  # A stale, large preceding mu must not control this iterate's gate.
  status = qp._check_termination(
      x, y, 1.0, s, 0.0, 1e12, 0.0, stats, True,
  )

  assert status is expected_status
  assert stats["ktratio"] == pytest.approx(expected_ratio)
  # These diagnostics intentionally remain in the returned frame.
  assert stats["mu"] == pytest.approx(1000.0 * returned_y)
  assert stats["complementarity"] == pytest.approx(1000.0 * returned_y)
  assert (qp._best_almost_iterate is not None) is track_almost
  if track_almost:
    assert qp._best_almost_score < 1.0
  else:
    # This point passes the numerical recession tests. Only the correct
    # gate prevents its original-frame ratio of 1e10 from certifying it.
    assert stats["ctx"] < -qp.tol_infeas_abs
    assert stats["dinfeas"] < qp.tol_infeas_rel


def test_gate_does_not_floor_current_complementarity():
  qp = qtqp.QTQP(
      a=sparse.csc_matrix([[0.0]]), b=np.array([1.0]),
      c=np.array([0.0]), z=0,
  )
  qp.solve(
      verbose=False, linear_solver=qtqp.LinearSolver.SCIPY,
      equilibration_strategy=qtqp.EquilibrationStrategy.NONE,
  )
  # Returned point x=0, y=1e-10, s=1 is feasible with gap 1e-10.
  # Homogeneous tau=1e-8 gives raw mu=1e-26 and kappa/tau=1e-10.
  # Flooring mu to 1e-14 would instead produce ratio 100.
  stats = {}
  status = qp._check_termination(
      np.array([0.0]), np.array([1e-18]), 1e-8, np.array([1e-8]),
      0.0, 1e-14, 0.0, stats, True,
  )
  assert status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(stats["ktratio"], 1e-10, rtol=1e-12, atol=0.0)
  np.testing.assert_allclose(stats["mu"], 1e-26, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("equilibration", list(qtqp.EquilibrationStrategy))
@pytest.mark.parametrize("infeasible", [False, True])
def test_true_certificates_survive_the_frame_correction(equilibration, infeasible):
  if infeasible:
    a = sparse.csc_matrix([[1.0], [-1.0]])
    b, c, z = np.array([0.0, -1.0]), np.array([0.0]), 0
  else:
    a = sparse.csc_matrix([[1.0, 2.0], [0.0, -1.0]])
    b, c, z = np.zeros(2), np.array([2.0, -1.0]), 1
  qp = qtqp.QTQP(a=a, b=b, c=c, z=z)
  solution = qp.solve(
      verbose=False, collect_stats=True, linear_solver=qtqp.LinearSolver.SCIPY,
      equilibration_strategy=equilibration,
  )

  assert solution.stats[-1]["ktratio"] > qp.certificate_ktratio
  if infeasible:
    assert solution.status is qtqp.SolutionStatus.INFEASIBLE
    assert b @ solution.y == pytest.approx(-1.0)
    assert np.linalg.norm(a.T @ solution.y) < 1e-8
    assert np.all(solution.y >= 0.0)
  else:
    assert solution.status is qtqp.SolutionStatus.UNBOUNDED
    assert c @ solution.x == pytest.approx(-1.0)
    assert np.linalg.norm(a @ solution.x + solution.s) < 1e-8
    assert solution.s[0] == 0.0
    assert solution.s[1] >= 0.0
