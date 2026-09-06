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
"""Real-GPU coverage of combined solves and returned-vector residuals."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


@pytest.fixture(params=[qtqp.LinearSolver.CUDSS, qtqp.LinearSolver.CUPY_DENSE])
def gpu_solver(request):
  cp = pytest.importorskip("cupy")
  if request.param is qtqp.LinearSolver.CUDSS:
    pytest.importorskip("nvmath")
  try:
    devices = cp.cuda.runtime.getDeviceCount()
  except cp.cuda.runtime.CUDARuntimeError as error:
    pytest.skip(f"CUDA device unavailable: {error}")
  if not devices:
    pytest.skip("No CUDA device available")
  return request.param


@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("nondefault_stream", [False, True])
def test_combined_gpu_operation_keeps_solution_on_device(
    monkeypatch, gpu_solver, correct, nondefault_stream
):
  import cupy as cp

  stream = cp.cuda.Stream(non_blocking=True) if nondefault_stream else cp.cuda.Stream.null
  with stream:
    backend = gpu_solver.value()
    try:
      backend.set_dims(n=2, m=1, z=1)
      triangle = sparse.coo_matrix([
          [4.0, 1.0, 2.0], [0.0, 5.0, -1.0], [0.0, 0.0, -3.0]
      ]).asformat(backend.format())
      backend.set_kkt(triangle)
      rhs = np.array([0.25, -0.5, 2.0])
      rhs_before = rhs.copy()
      add_to = np.array([0.2, -0.3, 0.4]) if correct else None
      add_before = None if add_to is None else add_to.copy()
      # A contiguous row view, as used for the preallocated GMRES basis.
      out = np.empty((2, 3))[1]
      device_solutions = []
      solve_gpu = backend._solve_gpu
      matvec_gpu = backend._matvec_gpu

      def solve(vector):
        result = solve_gpu(vector)
        device_solutions.append(result)
        return result

      def matvec(vector):
        # No download/re-upload (or device copy) between the two operations.
        assert vector is device_solutions[-1]
        return matvec_gpu(vector)

      monkeypatch.setattr(backend, "_solve_gpu", solve)
      monkeypatch.setattr(backend, "_matvec_gpu", matvec)
      for diagonal in ([4.0, 5.0, -3.0], [6.0, 7.0, -2.0]):
        backend.update_diag(np.array(diagonal))
        backend.factorize()
        matrix = triangle.toarray() + triangle.toarray().T
        np.fill_diagonal(matrix, diagonal)
        expected = np.linalg.solve(matrix, rhs)
        if correct:
          expected += add_to
        product = backend.solve_and_matvec(rhs, out=out, add_to=add_to)
        np.testing.assert_allclose(out, expected, rtol=1e-11, atol=1e-11)
        # Judge the actual returned vector, not the RHS or the reference solve.
        np.testing.assert_allclose(product, matrix @ out, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(rhs, rhs_before)
        if correct:
          np.testing.assert_array_equal(add_to, add_before)
      assert len(device_solutions) == 2
    finally:
      backend.free()


@pytest.mark.parametrize("strategy", list(qtqp.RefinementStrategy))
@pytest.mark.parametrize("equilibration", list(qtqp.EquilibrationStrategy))
def test_combined_gpu_refinement_matches_host_roundtrip(
    monkeypatch, gpu_solver, strategy, equilibration
):
  a = np.array([[1.0, -0.2], [1.0, 0.0], [-1.0, 0.0],
                [0.0, 1.0], [0.0, -1.0]])
  p = np.array([[2.0, 0.3], [0.3, 1.5]])
  x_star = np.array([0.2, -0.4])
  y_star = np.array([0.3, 1.0, 0.0, 0.7, 0.0])
  s_star = np.array([0.0, 0.0, 2.0, 0.0, 3.0])
  b, c = a @ x_star + s_star, -p @ x_star - a.T @ y_star
  options = dict(
      verbose=False, collect_stats=True, linear_solver=gpu_solver,
      refinement_strategy=strategy, equilibration_strategy=equilibration,
  )

  def run():
    return qtqp.QTQP(
        a=sparse.csc_matrix(a), b=b, c=c, p=sparse.csc_matrix(p), z=1
    ).solve(**options)

  combined = run()
  # Exercise the same factors/matvecs with the original host round trip.
  with monkeypatch.context() as patch:
    patch.setattr(gpu_solver.value, "solve_and_matvec", qtqp.direct.LinearSolver.solve_and_matvec)
    separate = run()

  for result in (combined, separate):
    assert result.status is qtqp.SolutionStatus.SOLVED
    np.testing.assert_allclose(result.x, x_star, atol=1e-7, rtol=0)
    np.testing.assert_allclose(a @ result.x + result.s, b, atol=1e-7, rtol=0)
    np.testing.assert_allclose(p @ result.x + a.T @ result.y + c, 0, atol=1e-7)
    assert abs(result.y @ result.s) < 1e-7
  assert combined.iterations == separate.iterations
  np.testing.assert_allclose(combined.x, separate.x, rtol=1e-10, atol=1e-10)
