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
"""CPU checks for combined factor-solve/matvec routing and buffer ownership."""

import numpy as np
import pytest
from scipy import sparse

from qtqp import direct


class _SharedBufferBackend(direct.LinearSolver):
  """A real dense solve and matvec deliberately overwrite the same buffer."""

  def set_kkt(self, kkt):
    super().set_kkt(kkt)
    self.buffer = np.empty(kkt.shape[0])
    self.solve_calls = 0

  def factorize(self):
    triangle = self._kkt.toarray()
    self.matrix = triangle + triangle.T - np.diag(np.diag(triangle))

  def solve(self, rhs):
    self.solve_calls += 1
    np.copyto(self.buffer, np.linalg.solve(self.matrix, rhs))
    return self.buffer

  def __matmul__(self, vector):
    np.copyto(self.buffer, self.matrix @ vector)
    return self.buffer

  def format(self):
    return 'csc'


@pytest.mark.parametrize('add', [False, True])
def test_fallback_copies_before_shared_buffer_matvec(add):
  matrix = np.array([[3.0, 1.0], [1.0, -2.0]])
  backend = _SharedBufferBackend()
  backend.set_kkt(sparse.csc_matrix(np.triu(matrix)))
  backend.factorize()
  rhs = np.array([0.3, -0.7])
  base = np.array([0.2, -0.1]) if add else None
  rhs_before, base_before = rhs.copy(), None if base is None else base.copy()
  storage = np.full((2, 2), -99.0)
  out = storage[1]  # Like a caller-owned row of the GMRES basis.
  expected = np.linalg.solve(matrix, rhs) + (base if add else 0)

  product = backend.solve_and_matvec(rhs, out=out, add_to=base)

  assert np.shares_memory(product, backend.buffer)
  assert not np.shares_memory(out, backend.buffer)
  np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)
  np.testing.assert_allclose(product, matrix @ expected, rtol=0, atol=1e-15)
  np.testing.assert_array_equal(storage[0], [-99.0, -99.0])
  np.testing.assert_array_equal(rhs, rhs_before)
  if add:
    np.testing.assert_array_equal(base, base_before)
  # A later backend operation may destroy its product, but not the
  # preconditioned vector already saved in caller-owned storage.
  backend @ np.zeros(2)
  np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)


class _ObservedCombinedBackend(_SharedBufferBackend):
  def __init__(self, corrupt_second=False):
    self.combined_calls = 0
    self.inside_combined = False
    self.outs = []
    self.added = []
    self.corrupt_second = corrupt_second

  def solve_and_matvec(self, rhs, *, out, add_to=None):
    self.combined_calls += 1
    assert not np.shares_memory(out, self.buffer)
    assert not np.shares_memory(out, rhs)
    if add_to is not None:
      assert not np.shares_memory(out, add_to)
    self.outs.append(out)
    self.added.append(add_to is not None)
    self.inside_combined = True
    try:
      return super().solve_and_matvec(rhs, out=out, add_to=add_to)
    finally:
      self.inside_combined = False

  def solve(self, rhs):
    assert self.inside_combined, 'refinement bypassed solve_and_matvec'
    result = super().solve(rhs)
    if self.corrupt_second and self.solve_calls == 2:
      result += 100.0
    return result


@pytest.mark.parametrize('strategy, warm_dtype', [
    (direct.RefinementStrategy.GMRES, np.float64),
    (direct.RefinementStrategy.RICHARDSON, np.float64),
    # The new Richardson output allocation must preserve the promotion
    # previously supplied by adding a float64 correction to the warm start.
    (direct.RefinementStrategy.RICHARDSON, np.float32),
    (direct.RefinementStrategy.RICHARDSON, np.int64),
])
def test_refinement_routes_through_combined_primitive(strategy, warm_dtype):
  a = np.array([[1.0, -0.4], [0.2, 0.8]])
  p = np.array([[2.0, 0.1], [0.1, 1.5]])
  mu, s, y = 0.25, np.array([0.0, 0.75]), np.array([1.0, 1.25])
  true_matrix = np.block([
      [p + mu * np.eye(2), a.T],
      [a, -np.diag([mu, mu + s[1] / y[1]])],
  ])
  expected = np.array([-0.3, 0.7, -0.2, 0.4])
  true_rhs = true_matrix @ expected
  rhs = true_rhs.copy()
  rhs[2:] *= -1
  warm = np.array([1, -2, 3, -1], dtype=warm_dtype)
  rhs_before, warm_before = rhs.copy(), warm.copy()
  backend = _ObservedCombinedBackend()
  solver = direct.DirectKktSolver(
      a=sparse.csc_matrix(a), p=sparse.csc_matrix(p), z=1,
      min_static_regularization=0.4, max_iterative_refinement_steps=30,
      atol=1e-12, rtol=1e-12, solver=backend,
      refinement_strategy=strategy, gmres_restart=4,
  )
  solver.update(mu=mu, s=s, y=y)
  actual, stats = solver.solve(rhs=rhs, warm_start=warm)

  assert stats['converged']
  assert stats['status'] == 'converged'
  assert backend.combined_calls == backend.solve_calls == stats['solves']
  assert backend.combined_calls >= 2  # The static clamp requires refinement.
  assert actual.dtype == np.float64
  np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-11)
  true_residual = np.linalg.norm(true_rhs - true_matrix @ actual, np.inf)
  assert true_residual <= 2 * stats['tolerance']
  assert stats['final_residual_norm'] == pytest.approx(true_residual, abs=5e-15)
  np.testing.assert_array_equal(rhs, rhs_before)
  np.testing.assert_array_equal(warm, warm_before)
  assert warm.dtype == warm_dtype
  if strategy is direct.RefinementStrategy.GMRES:
    assert not any(backend.added)
    assert all(np.shares_memory(out, solver._gm_z) for out in backend.outs)
  else:
    assert all(backend.added)
    assert all(not np.shares_memory(left, right)
               for i, left in enumerate(backend.outs)
               for right in backend.outs[i + 1:])


def test_richardson_rollback_preserves_true_iterate_with_combined_buffer():
  backend = _ObservedCombinedBackend(corrupt_second=True)
  solver = direct.DirectKktSolver(
      a=sparse.csc_matrix((1, 1)), p=sparse.eye(1, format='csc'), z=1,
      min_static_regularization=1.0, max_iterative_refinement_steps=10,
      atol=0.0, rtol=0.0, solver=backend,
      refinement_strategy=direct.RefinementStrategy.RICHARDSON,
  )
  solver.update(mu=0.5, s=np.zeros(1), y=np.ones(1))
  true_matrix = np.diag([1.5, -0.5])
  regularized_matrix = np.diag([1.5, -1.0])
  true_rhs, rhs = np.array([3.0, 2.0]), np.array([3.0, -2.0])
  warm = np.array([0.5, -0.25])
  expected = warm + np.linalg.solve(regularized_matrix, true_rhs - true_matrix @ warm)
  actual, stats = solver.solve(rhs=rhs, warm_start=warm)

  assert backend.combined_calls == backend.solve_calls == stats['solves'] == 2
  assert stats['status'] == 'stalled'
  assert not stats['converged']
  np.testing.assert_array_equal(actual, expected)
  assert not np.shares_memory(backend.outs[0], backend.outs[1])
  residual = np.linalg.norm(true_rhs - true_matrix @ actual, np.inf)
  assert residual == stats['final_residual_norm'] == 0.9375
  np.testing.assert_array_equal(warm, [0.5, -0.25])
  np.testing.assert_array_equal(rhs, [3.0, -2.0])
  backend @ np.zeros(2)
  np.testing.assert_array_equal(actual, expected)
