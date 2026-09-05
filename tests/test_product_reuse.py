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


def _dense_local_metric(x, y, s, tau, a, p, b, c, z):
  """Independent dense evaluation on ordinary-scale, strictly interior data."""
  mu = y @ s / (len(y) - z)
  barrier = np.zeros(len(x) + len(y) + 1)
  barrier[len(x) + z:-1] = -1 / y[z:]
  barrier[-1] = -1 / tau
  embedding = np.concatenate([
      p @ x + a.T @ y + c * tau,
      -a @ x + b * tau,
      [-c @ x - b @ y - x @ p @ x / tau],
  ])
  residual = embedding + mu * (np.concatenate([x, y, [tau]]) + barrier)
  diagonal = np.full(len(residual), mu)
  diagonal[len(x) + z:-1] += mu / y[z:] ** 2
  diagonal[-1] += mu / tau ** 2
  return np.sqrt(residual @ np.linalg.solve(np.diag(diagonal), residual))


@pytest.mark.parametrize('quadratic', [False, True])
@pytest.mark.parametrize('cached', ['ax', 'px', 'both'])
def test_cached_warm_metric_matches_standalone(quadratic, cached):
  a = np.array([[1.0, -0.2], [0.3, 1.2], [-0.7, 0.5]])
  p = np.array([[2.0, 0.3], [0.3, 1.5]]) if quadratic else np.zeros((2, 2))
  b, c = np.array([0.4, 1.0, -0.2]), np.array([0.7, -0.1])
  solver = qtqp.QTQP(
      a=sparse.csc_matrix(a), b=b, c=c, p=sparse.csc_matrix(p), z=1
  )
  # Different scales and shifts catch a cache that accidentally retains the
  # unnormalized product or a product from the preceding candidate.
  for scale, shift in [(0.3, 1e-4), (1.2, 0.1)]:
    x = scale * np.array([0.3, -0.7])
    y = scale * np.array([-0.4, shift, 0.8])
    s, tau = scale * np.array([0.0, 0.7, shift]), scale
    kwargs = {}
    if cached in ('ax', 'both'):
      kwargs['ax'] = a @ x
    if cached in ('px', 'both'):
      kwargs['px'] = p @ x
    counted_p = _CountingMatrix(p)
    actual = solver._lambda_local(
        x, y, s, tau, solver.a, counted_p, b, c, **kwargs
    )
    standalone = solver._lambda_local(x, y, s, tau, solver.a, solver.p, b, c)
    expected = _dense_local_metric(x, y, s, tau, a, p, b, c, 1)
    assert actual == pytest.approx(expected, rel=2e-13)
    assert standalone == pytest.approx(expected, rel=2e-13)
    assert counted_p.applies == (0 if 'px' in kwargs else 1)


@pytest.mark.parametrize('equilibration', list(qtqp.EquilibrationStrategy))
@pytest.mark.parametrize('quadratic, mode', [
    (True, 'accepted'), (False, 'accepted'),
    (True, 'vetoed'), (False, 'cold'),
])
def test_warm_screen_reuses_products_on_public_solve(
    monkeypatch, equilibration, quadratic, mode,
):
  a = np.array([[1.0, -0.3], [1.0, 0.0], [-1.0, 0.0],
                [0.0, 1.0], [0.0, -1.0]])
  p = np.array([[2.0, 0.2], [0.2, 1.5]]) if quadratic else np.zeros((2, 2))
  x_star = np.array([0.2, -0.4])
  y_star = np.array([0.3, 1.0, 0.0, 0.7, 0.0])
  s_star = np.array([0.0, 0.0, 2.0, 0.0, 3.0])
  b, c = a @ x_star + s_star, -p @ x_star - a.T @ y_star
  solver = qtqp.QTQP(
      a=sparse.csc_matrix(a), b=b, c=c, p=sparse.csc_matrix(p), z=1
  )
  options = dict(verbose=False, collect_stats=True,
                 linear_solver=qtqp.LinearSolver.SCIPY,
                 equilibration_strategy=equilibration)
  base = solver.solve(**options)
  assert base.status == qtqp.SolutionStatus.SOLVED

  state = {'screen': False, 'products': [], 'metrics': [], 'init_calls': 0}
  original_validate = solver._validate_warm_start
  original_metric = solver._lambda_local
  original_init = solver._init_variables
  original_check = solver._check_termination

  def validate(*args):
    result = original_validate(*args)
    state['screen'] = result is not None
    return result

  def metric(x, y, s, tau, a, p, b, c, **kwargs):
    dense_a, dense_p = a.toarray(), p.toarray()
    expected = _dense_local_metric(x, y, s, tau, dense_a, dense_p, b, c, 1)
    if state['screen']:
      if kwargs.get('ax') is not None:
        np.testing.assert_allclose(kwargs['ax'], dense_a @ x, rtol=2e-13, atol=1e-14)
      if kwargs.get('px') is not None:
        np.testing.assert_allclose(kwargs['px'], dense_p @ x, rtol=2e-13, atol=1e-14)
    actual = original_metric(x, y, s, tau, a, p, b, c, **kwargs)
    assert actual == pytest.approx(expected, rel=2e-12, abs=1e-12)
    state['metrics'].append((state['screen'], actual))
    return actual

  def init(*args):
    state['screen'] = False
    state['init_calls'] += 1
    return original_init(*args)

  def check(*args):
    state['screen'] = False
    return original_check(*args)

  # Observe actual sparse applications in the production screening window,
  # including the precomputation before _lambda_local is called. A.T is CSR.
  def count_products(original):
    def matmul(matrix, vector):
      if state['screen']:
        state['products'].append(matrix.shape)
      return original(matrix, vector)
    return matmul

  for matrix_type in (sparse.csc_matrix, sparse.csr_matrix):
    monkeypatch.setattr(matrix_type, '__matmul__', count_products(matrix_type.__matmul__))
  monkeypatch.setattr(solver, '_validate_warm_start', validate)
  monkeypatch.setattr(solver, '_lambda_local', metric)
  monkeypatch.setattr(solver, '_init_variables', init)
  monkeypatch.setattr(solver, '_check_termination', check)

  previous_score = None
  for perturbation in (0.0, 0.02):
    state.update(screen=False, products=[], metrics=[], init_calls=0)
    warm_start = None if mode == 'cold' else (
        base.x + perturbation * np.array([1.0, -0.5]),
        base.y * (1 + perturbation), base.s + perturbation,
    )
    threshold = 1e-20 if mode == 'vetoed' else 100.0
    result = solver.solve(
        **options, warm_start=warm_start, warm_start_threshold=threshold
    )
    assert result.status == qtqp.SolutionStatus.SOLVED
    np.testing.assert_allclose(a @ result.x + result.s, b, atol=1e-7, rtol=0)
    np.testing.assert_allclose(p @ result.x + a.T @ result.y + c, 0, atol=1e-7)
    np.testing.assert_allclose(result.x, x_star, atol=1e-7, rtol=0)

    screen_scores = [score for screened, score in state['metrics'] if screened]
    cold_scores = [score for screened, score in state['metrics'] if not screened]
    if mode == 'cold':
      assert state['products'] == []
      assert screen_scores == []
      assert solver.warm_lambda is None
    else:
      assert state['products'].count(a.shape) == 1
      assert state['products'].count(p.shape) == 1
      assert state['products'].count(a.T.shape) == 4
      assert len(state['products']) == 6
      assert len(screen_scores) == 4
      assert solver.warm_lambda == min(screen_scores)
      if previous_score is not None:
        assert solver.warm_lambda != previous_score
      previous_score = solver.warm_lambda
    assert solver.warm_accepted == (mode == 'accepted')
    if mode == 'accepted':
      assert state['init_calls'] == 0
      assert cold_scores == []
      assert solver.lambda_init == solver.warm_lambda
    else:
      assert state['init_calls'] == 1
      assert len(cold_scores) == 1
      assert solver.lambda_init == cold_scores[0]


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
