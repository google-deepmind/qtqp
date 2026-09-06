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
"""GPU KKT solver backends (CUDA)."""

import gc
import logging
from typing import Literal

import numpy as np
import scipy.sparse as sp

from .direct import LinearSolver


class _GpuSolver(LinearSolver):
  """Host interface around device solves and non-destructive device matvecs."""

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    self._x_gpu.set(x)
    return self._matvec_gpu(self._x_gpu).get()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self._solve_gpu(rhs).get()

  def solve_and_matvec(
      self, rhs: np.ndarray, *, out: np.ndarray, add_to: np.ndarray | None = None
  ) -> np.ndarray:
    # Form the complete vector on-device, then evaluate its actual matvec.
    # Queue both operations before either download; never reconstruct the
    # residual from the RHS or a recurrence that omits factorization error.
    solution = self._solve_gpu(rhs)
    if add_to is not None:
      self._x_gpu.set(add_to)
      self._cp.add(self._x_gpu, solution, out=solution)
    product = self._matvec_gpu(solution)
    solution.get(out=out)
    return product.get()


class CuDssSolver(_GpuSolver):
  """Wrapper around Nvidia's CuDSS for GPU-accelerated solving.

  Maintains a single GPU sparse matrix used for both nvmath (factorize/solve)
  and cupy matvec.  On the first call, the CPU matrix is converted to a cupy
  GPU sparse matrix which is passed to nvmath's DirectSolver.  On subsequent
  calls the GPU data array is updated in-place via .set(); nvmath wraps the
  data pointer so it sees the new values without needing reset_operands
  (which would invalidate the plan).
  """

  def __init__(self):
    import cupy  # pylint: disable=g-import-not-at-top
    import cupyx.scipy.sparse  # pylint: disable=g-import-not-at-top
    import nvmath  # pylint: disable=g-import-not-at-top

    self._cp = cupy
    self._cp_sparse = cupyx.scipy.sparse
    self.nvmath = nvmath
    self._solver: nvmath.sparse.advanced.DirectSolver | None = None
    # Single GPU sparse matrix for both nvmath and matvec.
    self._kkt_gpu = None
    self._x_gpu: cupy.ndarray | None = None
    self._rhs_gpu: cupy.ndarray | None = None

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    """Transfers KKT to GPU; called once at init time."""
    super().set_kkt(kkt)
    self._kkt_gpu = self._cp_sparse.csr_matrix(kkt)
    # The transpose shares data, so diagonal updates keep both views current.
    self._kkt_gpu_t = self._kkt_gpu.T
    self._kkt_diag_gpu = self._cp.asarray(self._kkt_diag)
    self._kkt_diag_idxs_gpu = self._cp.asarray(self._kkt_diag_idxs)

  def update_diag(self, diag: np.ndarray) -> None:
    self._kkt_diag_gpu.set(diag)
    self._kkt_gpu.data[self._kkt_diag_idxs_gpu] = self._kkt_diag_gpu

  def factorize(self):
    cp = self._cp
    if self._solver is None:
      sparse_system_type = (
          self.nvmath.sparse.advanced.DirectSolverMatrixType.SYMMETRIC
      )
      sparse_system_view = (
          self.nvmath.sparse.advanced.DirectSolverMatrixViewType.UPPER
      )
      # Turn off annoying logs by default.
      logger = logging.getLogger("null")
      logger.disabled = True
      options = self.nvmath.sparse.advanced.DirectSolverOptions(
          sparse_system_type=sparse_system_type,
          sparse_system_view=sparse_system_view,
          logger=logger,
      )
      n = self._kkt_gpu.shape[1]
      self._x_gpu = cp.empty(n, dtype=cp.float64)
      self._rhs_gpu = cp.empty(n, order="F", dtype=cp.float64)
      self._solver = self.nvmath.sparse.advanced.DirectSolver(
          self._kkt_gpu, self._rhs_gpu, options=options
      )
      self._solver.plan()
    # No reset_operands: nvmath wraps _kkt_gpu's data pointer, so
    # in-place updates via .set() in set_kkt are visible directly.

    self._solver.factorize()

  def _matvec_gpu(self, x):
    return (
        self._kkt_gpu @ x
        + self._kkt_gpu_t @ x
        - self._kkt_diag_gpu * x
    )

  def _solve_gpu(self, rhs):
    self._rhs_gpu.set(rhs)
    return self._solver.solve(stream=self._cp.cuda.get_current_stream())

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self._solver is not None:
      self._solver.free()
      self._solver = None
      # Force clean up any 'zombie' references, in order to avoid cuda errors.
      gc.collect(0)  # Run GC only on the youngest generation.


class CupyDenseSolver(_GpuSolver):
  """GPU Cholesky solver via Gram/Schur-complement reduction (cupy).

  GPU counterpart of ScipyDenseSolver.  See that class's docstring for the
  Gram derivation.  Forms G = H + A' D^{-1} A on the GPU and factorizes
  with Cholesky via cupy.linalg.cholesky.
  """

  def __init__(self):
    import cupy  # pylint: disable=g-import-not-at-top
    import cupyx.scipy.linalg  # pylint: disable=g-import-not-at-top

    self._cp = cupy
    self._cupyx_linalg = cupyx.scipy.linalg
    self._n = 0
    self._m = 0

  def set_dims(self, n: int, m: int, z: int) -> None:
    cp = self._cp
    self._n = n
    self._m = m
    self._z = z
    self._A_gpu = None
    self._P_offdiag_gpu = None
    self._R_x_gpu = cp.empty(n, dtype=cp.float64)
    self._R_y_gpu = cp.empty(m, dtype=cp.float64)
    self._inv_R_y_gpu = cp.empty(m, dtype=cp.float64)
    self._inv_sqrt_R_y_gpu = cp.empty(m, dtype=cp.float64)
    self._G_gpu = cp.empty((n, n), dtype=cp.float64)
    self._diag_idx = cp.arange(n)
    self._result_gpu = cp.empty(n + m, dtype=cp.float64)
    # Keep the solve result intact while computing its device matvec.
    self._matvec_result_gpu = cp.empty(n + m, dtype=cp.float64)
    self._x_gpu = cp.empty(n + m, dtype=cp.float64)
    self._rhs_gpu = cp.empty(n + m, dtype=cp.float64)
    self._g_gpu = cp.empty(n, dtype=cp.float64)
    self._L = None  # Lower-triangular Cholesky factor

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    super().set_kkt(kkt)
    cp = self._cp
    n = self._n
    # Extract before densifying, avoiding a full (n + m)^2 host allocation
    # for a backend that only needs the m x n and n x n blocks.
    self._A_gpu = cp.asarray(
        kkt[:n, n:].T.toarray(order="F"), dtype=cp.float64
    )
    P_block = kkt[:n, :n].toarray()
    P_block = P_block + P_block.T - np.diag(np.diag(P_block))
    np.fill_diagonal(P_block, 0.0)
    self._P_offdiag_gpu = cp.asarray(P_block, dtype=cp.float64)

  def update_diag(self, diag: np.ndarray) -> None:
    if self._z and np.any(diag[self._n : self._n + self._z] == 0.0):
      raise ValueError(
          "Dense Gram elimination requires positive regularization on equality "
          "rows. Set min_static_regularization > 0 for initialization."
      )
    cp = self._cp
    self._R_x_gpu.set(diag[:self._n])
    self._R_y_gpu.set(-diag[self._n:])
    cp.divide(1.0, self._R_y_gpu, out=self._inv_R_y_gpu)
    cp.sqrt(self._inv_R_y_gpu, out=self._inv_sqrt_R_y_gpu)

  def factorize(self) -> None:
    cp = self._cp
    idx = self._diag_idx
    cp.copyto(self._G_gpu, self._P_offdiag_gpu)
    self._G_gpu[idx, idx] += self._R_x_gpu
    A_scaled = self._A_gpu * self._inv_sqrt_R_y_gpu[:, None]
    self._G_gpu += A_scaled.T @ A_scaled
    # Same numerical perturbation as ScipyDenseSolver.factorize.
    self._G_gpu[idx, idx] += 1e-14 * cp.max(self._G_gpu[idx, idx])
    self._L = cp.linalg.cholesky(self._G_gpu)

  def _matvec_gpu(self, x):
    cp = self._cp
    n = self._n
    x_x, x_y = x[:n], x[n:]
    result = self._matvec_result_gpu
    cp.dot(self._P_offdiag_gpu, x_x, out=result[:n])
    cp.multiply(self._R_x_gpu, x_x, out=self._g_gpu)
    result[:n] += self._g_gpu
    cp.dot(self._A_gpu.T, x_y, out=self._g_gpu)
    result[:n] += self._g_gpu
    cp.dot(self._A_gpu, x_x, out=result[n:])
    result[n:] -= self._R_y_gpu * x_y
    return result

  def _solve_gpu(self, rhs):
    cp = self._cp
    n = self._n
    self._rhs_gpu.set(rhs)
    inv_R_y = self._inv_R_y_gpu
    cp.multiply(inv_R_y, self._rhs_gpu[n:], out=self._result_gpu[n:])
    cp.dot(self._A_gpu.T, self._result_gpu[n:], out=self._g_gpu)
    self._g_gpu += self._rhs_gpu[:n]
    # Solve L L' x = g via triangular solves.
    x = self._cupyx_linalg.solve_triangular(self._L, self._g_gpu, lower=True)
    x = self._cupyx_linalg.solve_triangular(self._L, x, lower=True, trans='C')
    result = self._result_gpu
    result[:n] = x
    cp.dot(self._A_gpu, x, out=result[n:])
    result[n:] -= self._rhs_gpu[n:]
    result[n:] *= inv_R_y
    return result

  def format(self) -> Literal["csr"]:
    return "csr"
