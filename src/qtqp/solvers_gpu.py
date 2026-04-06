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


class CuDssSolver(LinearSolver):
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
    """Transfers KKT data to GPU; does not retain the CPU matrix."""
    if self._kkt_gpu is None:
      self._kkt_gpu = self._cp_sparse.csr_matrix(kkt)
    else:
      self._kkt_gpu.data.set(kkt.data)

  def factorize(self):
    cp = self._cp
    if self._solver is None:
      sparse_system_type = (
          self.nvmath.sparse.advanced.DirectSolverMatrixType.SYMMETRIC
      )
      # Turn off annoying logs by default.
      logger = logging.getLogger("null")
      logger.disabled = True
      options = self.nvmath.sparse.advanced.DirectSolverOptions(
          sparse_system_type=sparse_system_type, logger=logger
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

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    self._x_gpu.set(x)
    return (self._kkt_gpu @ self._x_gpu).get()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    self._rhs_gpu.set(rhs)
    result = self._solver.solve()
    return self._cp.asnumpy(result)

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self._solver is not None:
      self._solver.free()
      self._solver = None
      # Force clean up any 'zombie' references, in order to avoid cuda errors.
      gc.collect(0)  # Run GC only on the youngest generation.


class CupyDenseSolver(LinearSolver):
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
    self._A_gpu = None
    self._P_offdiag_gpu = None
    self._R_x_gpu = cp.empty(n, dtype=cp.float64)
    self._R_y_gpu = cp.empty(m, dtype=cp.float64)
    self._G_gpu = cp.empty((n, n), dtype=cp.float64)
    self._diag_idx = cp.arange(n)
    self._result_gpu = cp.empty(n + m, dtype=cp.float64)
    self._L = None  # Lower-triangular Cholesky factor

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    cp = self._cp
    n, m = self._n, self._m
    if self._A_gpu is None:
      kkt_dense = kkt.toarray()
      self._A_gpu = cp.asarray(kkt_dense[n:, :n], dtype=cp.float64)
      P_block = kkt_dense[:n, :n].copy()
      np.fill_diagonal(P_block, 0.0)
      self._P_offdiag_gpu = cp.asarray(P_block, dtype=cp.float64)
    diag = kkt.diagonal()
    self._R_x_gpu.set(diag[:n])
    self._R_y_gpu.set(-diag[n:])

  def factorize(self) -> None:
    cp = self._cp
    idx = self._diag_idx
    cp.copyto(self._G_gpu, self._P_offdiag_gpu)
    self._G_gpu[idx, idx] += self._R_x_gpu
    A_scaled = self._A_gpu * (1.0 / cp.sqrt(self._R_y_gpu))[:, None]
    self._G_gpu += A_scaled.T @ A_scaled
    # Same numerical perturbation as ScipyDenseSolver.factorize.
    self._G_gpu[idx, idx] += 1e-14 * cp.max(self._G_gpu[idx, idx])
    self._L = cp.linalg.cholesky(self._G_gpu)

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    cp = self._cp
    n = self._n
    x_gpu = cp.asarray(x)
    x_x, x_y = x_gpu[:n], x_gpu[n:]
    result = self._result_gpu
    result[:n] = self._P_offdiag_gpu @ x_x + self._R_x_gpu * x_x + self._A_gpu.T @ x_y
    result[n:] = self._A_gpu @ x_x - self._R_y_gpu * x_y
    return cp.asnumpy(result)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    cp = self._cp
    n = self._n
    rhs_gpu = cp.asarray(rhs)
    inv_R_y = 1.0 / self._R_y_gpu
    g = rhs_gpu[:n] + self._A_gpu.T @ (inv_R_y * rhs_gpu[n:])
    # Solve L L' x = g via triangular solves.
    x = self._cupyx_linalg.solve_triangular(self._L, g, lower=True)
    x = self._cupyx_linalg.solve_triangular(self._L, x, lower=True, trans='C')
    result = self._result_gpu
    result[:n] = x
    cp.multiply(inv_R_y, self._A_gpu @ x - rhs_gpu[n:], out=result[n:])
    return cp.asnumpy(result)

  def format(self) -> Literal["csr"]:
    return "csr"
