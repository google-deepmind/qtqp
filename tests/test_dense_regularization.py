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
"""Dense equality-diagonal validation preserves supported zero-shift solves."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _dense_backend(name):
  if name == "scipy":
    return qtqp.direct.ScipyDenseSolver()
  # Initialization/setup use only NumPy-compatible allocation and transfer
  # primitives. The invalid diagonal must raise before any device operation.
  backend = qtqp.direct.CupyDenseSolver.__new__(qtqp.direct.CupyDenseSolver)
  backend._cp = np
  return backend


def _direct_solver(backend):
  return qtqp.direct.DirectKktSolver(
      a=sparse.csc_matrix([[1.0]]), p=sparse.csc_matrix([[1.0]]), z=1,
      min_static_regularization=0.0, max_iterative_refinement_steps=20,
      atol=1e-12, rtol=1e-12, solver=backend,
  )


@pytest.mark.parametrize("backend_name", ["scipy", "cupy"])
@pytest.mark.parametrize("zero_row", [0, 1])
def test_zero_equality_diagonal_rejected_before_backend_work(backend_name, zero_row):
  backend = _dense_backend(backend_name)
  backend.set_dims(n=1, m=3, z=2)
  if backend_name == "scipy":
    primal, dual = backend._R_x, backend._R_y
  else:
    primal, dual = backend._R_x_gpu, backend._R_y_gpu
    backend._cp = None  # validation must not need the GPU namespace
  primal.fill(17.0)
  dual.fill(19.0)
  diag = np.array([2.0, -0.1, -0.2, -1.0])
  diag[1 + zero_row] = 0.0
  with np.errstate(divide="raise", invalid="raise"):
    with pytest.raises(ValueError, match="min_static_regularization > 0"):
      backend.update_diag(diag)
  np.testing.assert_array_equal(primal, 17.0)
  np.testing.assert_array_equal(dual, 19.0)


@pytest.mark.parametrize("backend_name", ["scipy", "cupy"])
@pytest.mark.parametrize("z", [1, 2])
def test_public_dense_zero_shift_error_frees_backend(monkeypatch, backend_name, z):
  backend = _dense_backend(backend_name)
  freed = []
  monkeypatch.setattr(backend, "free", lambda: freed.append(True))
  requested = (qtqp.LinearSolver.SCIPY_DENSE if backend_name == "scipy"
               else qtqp.LinearSolver.CUPY_DENSE)
  monkeypatch.setattr(qtqp, "_resolve_linear_solver", lambda _: (requested, backend))
  solver = qtqp.QTQP(
      a=sparse.csc_matrix([[1.0], [2.0]]), b=np.array([1.0, 2.0]),
      p=sparse.csc_matrix([[1.0]]), c=np.zeros(1), z=z,
  )
  with np.errstate(divide="raise", invalid="raise"):
    with pytest.raises(ValueError, match="min_static_regularization > 0"):
      solver.solve(
          verbose=False, linear_solver=requested, min_static_regularization=0.0
      )
  assert freed == [True]
  assert solver._linear_solver is None


@pytest.mark.parametrize("backend_name", ["scipy", "cupy"])
@pytest.mark.parametrize("initialization", [True, False])
def test_standalone_dense_solver_rejects_zero_equality_shift(
    backend_name, initialization
):
  solver = _direct_solver(_dense_backend(backend_name))
  try:
    with np.errstate(divide="raise", invalid="raise"):
      with pytest.raises(ValueError, match="min_static_regularization > 0"):
        if initialization:
          solver.update_init()
        else:
          solver.update(mu=0.0, s=np.zeros(1), y=np.ones(1))
  finally:
    solver.free()


def test_dense_positive_mu_allows_zero_static_regularization():
  solver = _direct_solver(qtqp.direct.ScipyDenseSolver())
  try:
    mu = 0.2
    solver.update(mu=mu, s=np.zeros(1), y=np.ones(1))
    rhs = np.array([1.0, 1.0])
    actual, stats = solver.solve(rhs=rhs, warm_start=np.zeros(2))
    expected = np.linalg.solve(np.array([[1 + mu, 1.0], [-1.0, mu]]), rhs)
    assert stats["converged"]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
  finally:
    solver.free()


@pytest.mark.parametrize("backend,z,regularization", [
    (qtqp.LinearSolver.SCIPY_DENSE, 0, 0.0),
    (qtqp.LinearSolver.SCIPY, 1, 0.0),
    (qtqp.LinearSolver.SCIPY_DENSE, 1, 1e-8),
])
def test_supported_regularization_cases_still_solve(backend, z, regularization):
  result = qtqp.QTQP(
      a=sparse.csc_matrix([[1.0]]), b=np.ones(1), c=np.zeros(1),
      p=sparse.csc_matrix([[1.0]]), z=z,
  ).solve(
      verbose=False, linear_solver=backend,
      min_static_regularization=regularization,
  )
  assert result.status == qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(result.x, [float(z)], atol=1e-7, rtol=0)
