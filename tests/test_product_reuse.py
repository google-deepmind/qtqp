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
"""Product reuse preserves the local metric and Newton tau equation."""

from collections import defaultdict

import numpy as np
import pytest
from scipy import sparse

import qtqp


class _CountingMatrix:
  """Count actual matrix applications without changing their arithmetic."""

  def __init__(self, matrix):
    self.matrix = sparse.csc_matrix(matrix)
    self.nnz = self.matrix.nnz
    self.applies = 0

  def __matmul__(self, vector):
    self.applies += 1
    return self.matrix @ vector


def test_warm_local_metric_uses_one_p_product():
  a = np.array([[1.0, -0.2], [0.3, 1.2], [-0.7, 0.5]])
  p = np.array([[2.0, 0.3], [0.3, 1.5]])
  b, c = np.array([0.4, 1.0, -0.2]), np.array([0.7, -0.1])
  x, y = np.array([0.3, -0.7]), np.array([-0.4, 1.2, 0.8])
  s, tau = np.array([0.0, 0.7, 1.1]), 0.9
  solver = qtqp.QTQP(
      a=sparse.csc_matrix(a), b=b, c=c, p=sparse.csc_matrix(p), z=1
  )
  counted_p = _CountingMatrix(p)
  actual = solver._lambda_local(x, y, s, tau, solver.a, counted_p, b, c)

  # Evaluate ||T_mu||_{H^-1} independently as a dense diagonal-metric
  # solve, including the equality row and the quadratic perspective term.
  mu = y @ s / 2
  u = np.concatenate([x, y, [tau]])
  barrier_gradient = np.array([0.0, 0.0, 0.0, -1 / y[1], -1 / y[2], -1 / tau])
  embedding = np.concatenate([
      p @ x + a.T @ y + c * tau,
      -a @ x + b * tau,
      [-c @ x - b @ y - (x @ p @ x) / tau],
  ])
  residual = embedding + mu * (u + barrier_gradient)
  metric = mu * np.diag([1.0, 1.0, 1.0,
                         1 + 1 / y[1] ** 2, 1 + 1 / y[2] ** 2,
                         1 + 1 / tau ** 2])
  expected = np.sqrt(residual @ np.linalg.solve(metric, residual))
  assert actual == pytest.approx(expected, rel=1e-13)
  assert counted_p.applies == 1


def test_tau_products_shared_through_gondzio_trials(monkeypatch):
  rng = np.random.default_rng(2)
  n, m, z = 8, 20, 3
  a = rng.normal(size=(m, n))
  factor = rng.normal(size=(n, n))
  p = 0.1 * (factor.T @ factor) + 0.05 * np.eye(n)
  x_star, y_star = rng.normal(size=n), rng.normal(size=m)
  s_star = np.zeros(m)
  y_star[z:] = rng.uniform(0.1, 2.0, size=m - z)
  inactive = rng.uniform(size=m - z) < 0.5
  y_star[z:][inactive] = 0.0
  s_star[z:][inactive] = rng.uniform(0.1, 2.0, size=inactive.sum())
  b, c = a @ x_star + s_star, -p @ x_star - a.T @ y_star
  solver = qtqp.QTQP(
      a=sparse.csc_matrix(a), b=b, c=c, p=sparse.csc_matrix(p), z=z
  )
  invariants = {}
  newton_calls = defaultdict(list)
  original_invariants = solver._tau_invariants
  original_newton = solver._newton_step

  def record_invariants(p, mu):
    assert solver.it not in invariants, "tau products rebuilt within an iteration"
    data = original_invariants(p, mu)
    invariants[solver.it] = (mu, data)
    return data

  def record_newton(**kwargs):
    data = kwargs["tau_data"]
    assert data is invariants[solver.it][1]
    newton_calls[solver.it].append(data)
    return original_newton(**kwargs)

  monkeypatch.setattr(solver, "_tau_invariants", record_invariants)
  monkeypatch.setattr(solver, "_newton_step", record_newton)
  result = solver.solve(
      verbose=False, linear_solver=qtqp.LinearSolver.SCIPY_DENSE,
      max_centrality_correctors=3,
  )
  assert result.status == qtqp.SolutionStatus.SOLVED
  assert invariants.keys() == newton_calls.keys()
  assert len({mu for mu, _ in invariants.values()}) > 1
  # The first two calls are predictor/corrector; a third is a real
  # Gondzio trial, so this checks every production consumer of the cache.
  assert max(map(len, newton_calls.values())) >= 3
  np.testing.assert_allclose(a @ result.x + result.s, b, atol=1e-6, rtol=0)
  np.testing.assert_allclose(p @ result.x + a.T @ result.y + c, 0, atol=1e-6)
  assert c @ result.x + 0.5 * result.x @ p @ result.x == pytest.approx(
      c @ x_star + 0.5 * x_star @ p @ x_star, abs=1e-6
  )


def test_cached_tau_root_tracks_changed_system():
  a = np.array([[1.0, 0.2], [-0.3, 1.0]])
  b, c = np.array([0.2, 1.0]), np.array([0.4, -0.1])
  solver = qtqp.QTQP(a=sparse.csc_matrix(a), b=b, c=c, z=1)
  rhs = np.array([0.8, -0.3, 0.1, 0.5])
  mu_target, r_tau = 0.03, 0.2

  for mu, cost_scale, p_scale in [(0.2, 1.0, 1.0), (0.07, 0.8, 1.7)]:
    p = p_scale * np.array([[2.0, 0.3], [0.3, 1.5]])
    kkt = np.block([
        [p + mu * np.eye(2), a.T],
        [-a, np.diag([mu, 0.7 / 1.2 + mu])],
    ])
    solver.q = np.concatenate([cost_scale * c, b])
    solver.kinv_q = np.linalg.solve(kkt, solver.q)
    kinv_r = np.linalg.solve(kkt, rhs)
    counted_p = _CountingMatrix(p)
    tau_data = solver._tau_invariants(counted_p, mu)
    assert counted_p.applies == 1
    cached = solver._solve_for_tau(
        counted_p, kinv_r, mu, mu_target, r_tau, tau_data=tau_data
    )
    assert counted_p.applies == 2  # only the varying P K^-1 r product
    standalone = solver._solve_for_tau(
        sparse.csc_matrix(p), kinv_r, mu, mu_target, r_tau
    )

    # Form the perspective equation independently and select its positive
    # polynomial root. Changing both the system and q detects stale data.
    v, w = solver.kinv_q, kinv_r
    coefficients = [
        mu + v @ solver.q - v[:2] @ p @ v[:2],
        -r_tau - w @ solver.q + 2 * w[:2] @ p @ v[:2],
        -mu_target - w[:2] @ p @ w[:2],
    ]
    expected = max(np.roots(coefficients))
    assert cached == pytest.approx(expected, rel=1e-13)
    assert standalone == pytest.approx(expected, rel=1e-13)
