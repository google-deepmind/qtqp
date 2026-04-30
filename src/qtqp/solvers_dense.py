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
"""Dense KKT solver backends using Gram/Schur-complement reduction."""

from typing import Literal

import numpy as np
import scipy.sparse as sp

from .direct import LinearSolver


class ScipyDenseSolver(LinearSolver):
  """Dense Cholesky solver via Gram/Schur-complement reduction.

  Instead of factorizing the full (n+m)x(n+m) KKT system, eliminates y to
  obtain an n x n SPD system solved by Cholesky (dpotrf).  Reduces
  factorization cost from O((n+m)^3) to O(n^3), a large win when m >> n
  (typical for QPs).

  **Derivation.**  The KKT system is:

      [ H   A'] [x]   [r_x]
      [ A  -D ] [y] = [r_y]

  where H = P + diag(R_x) (n x n, SPD) and D = diag(R_y) (m x m, D > 0).
  From the second row:  y = D^{-1} (A x - r_y).
  Substituting into the first row gives the Gram (normal-equations) system:

      G x = r_x + A' D^{-1} r_y,   where  G = H + A' D^{-1} A.

  G is SPD (H is SPD from regularization, A' D^{-1} A is PSD).

  **Implementation.**  The KKT diagonal mixes P's diagonal with R_x (the
  regularization), so we store P_offdiag (P with diagonal zeroed) once and
  read the combined diagonal each iteration:

      G = P_offdiag + diag(kkt_diag[:n]) + A' D^{-1} A
        = P_offdiag + diag(P_diag + R_x)  + A' D^{-1} A
        = P + diag(R_x) + A' D^{-1} A     = H + A' D^{-1} A.

  The A' D^{-1} A term is formed via dsyrk on A_scaled = diag(1/sqrt(R_y)) A
  so that A_scaled' A_scaled = A' D^{-1} A, exploiting BLAS3 symmetry.
  """

  def __init__(self):
    from scipy.linalg import lapack, blas  # pylint: disable=g-import-not-at-top

    self._dpotrf = lapack.dpotrf
    self._dpotrs = lapack.dpotrs
    self._dsyrk = blas.dsyrk
    self._dsymv = blas.dsymv
    self._n = 0
    self._m = 0

  def set_dims(self, n: int, m: int, z: int) -> None:
    self._n = n
    self._m = m
    # Blocks extracted once from the KKT scaffold (populated in first set_kkt).
    self._A: np.ndarray | None = None         # (m, n) dense, C-order
    self._P_offdiag: np.ndarray | None = None  # (n, n) P with diagonal zeroed, F-order

    # Pre-allocate all per-iteration buffers.
    self._R_x = np.empty(n, dtype=np.float64)
    self._R_y = np.empty(m, dtype=np.float64)
    self._G = np.empty((n, n), dtype=np.float64, order="F")
    self._chol = np.empty((n, n), dtype=np.float64, order="F")
    self._A_scaled = np.empty((m, n), dtype=np.float64, order="F")
    self._diag_idx = np.diag_indices(n)
    self._result = np.empty(n + m, dtype=np.float64)
    self._g = np.empty(n, dtype=np.float64)

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    super().set_kkt(kkt)
    n = self._n
    kkt_dense = kkt.toarray()
    self._A = np.ascontiguousarray(kkt_dense[:n, n:].T, dtype=np.float64)
    P_block = kkt_dense[:n, :n]
    P_block = P_block + P_block.T - np.diag(np.diag(P_block))
    np.fill_diagonal(P_block, 0.0)
    self._P_offdiag = np.asfortranarray(P_block)

  def update_diag(self, diag: np.ndarray) -> None:
    np.copyto(self._R_x, diag[:self._n])
    np.negative(diag[self._n:], out=self._R_y)

  def factorize(self) -> None:
    # G = P_offdiag + diag(R_x) + A' diag(1/R_y) A
    np.copyto(self._G, self._P_offdiag)
    self._G[self._diag_idx] += self._R_x
    np.multiply(self._A, (1.0 / np.sqrt(self._R_y))[:, None], out=self._A_scaled)
    # Symmetric rank-k update: G += A_scaled.T @ A_scaled via BLAS dsyrk.
    self._dsyrk(1.0, self._A_scaled, beta=1.0, c=self._G, trans=1,
                lower=1, overwrite_c=True)
    # G is theoretically SPD but the rank-k update can introduce roundoff
    # that makes it very slightly indefinite (eigenvalue ~ -1e-8) when 1/R_y
    # spans many orders of magnitude.  A tiny relative perturbation fixes
    # this; iterative refinement (which uses the exact block matvec in
    # __matmul__) corrects for any factorization-level perturbation.
    self._G[self._diag_idx] += 1e-14 * np.max(self._G[self._diag_idx])
    np.copyto(self._chol, self._G)
    self._chol, info = self._dpotrf(self._chol, lower=True, overwrite_a=True)
    if info != 0:
      raise np.linalg.LinAlgError(f"Cholesky failed (dpotrf info={info})")

  def rescale_off_diagonals(self, r: np.ndarray) -> None:
    super().rescale_off_diagonals(r)
    n = self._n
    r_x = r[:n]
    r_y = r[n:]
    # _A is (m, n): A[i, j] *= r_y[i] * r_x[j].
    self._A *= r_y[:, None]
    self._A *= r_x[None, :]
    # _P_offdiag is (n, n) symmetric: P[i, j] *= r_x[i] * r_x[j].
    self._P_offdiag *= r_x[:, None]
    self._P_offdiag *= r_x[None, :]

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    n = self._n
    x_x, x_y = x[:n], x[n:]
    result = self._result
    # result[:n] = P_offdiag @ x_x + R_x * x_x + A.T @ x_y
    self._dsymv(1.0, self._P_offdiag, x_x, 0.0, result[:n], lower=1,
                overwrite_y=1)
    result[:n] += self._R_x * x_x
    result[:n] += self._A.T @ x_y
    result[n:] = self._A @ x_x - self._R_y * x_y
    return result

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    n = self._n
    inv_R_y = 1.0 / self._R_y
    # Reduced RHS: g = rhs_x + A' (R_y^{-1} rhs_y)
    g = self._g
    np.copyto(g, rhs[:n])
    g += self._A.T @ (inv_R_y * rhs[n:])
    x, _ = self._dpotrs(self._chol, g, lower=True)
    # Back-substitute: y = R_y^{-1} (A x - rhs_y)
    result = self._result
    result[:n] = x
    np.multiply(inv_R_y, self._A @ x - rhs[n:], out=result[n:])
    return result

  def format(self) -> Literal["csr"]:
    return "csr"
