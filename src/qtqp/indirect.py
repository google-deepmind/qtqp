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
"""Indirect (iterative) KKT solver with swappable backends.

IndirectKktSolver owns the KKT matrix, preconditioning, and adaptive tolerance,
delegating the iterative solve to swappable IterativeSolver backends. It exposes
the same interface as DirectKktSolver (.update, .solve, .free) so the IPM loop
can use either transparently.

Key differences from DirectKktSolver:
  - No static regularization or diagonal correction: the iterative solver uses
    the true KKT matrix. QTQP's mu*I terms already ensure bounded condition
    number O(1/mu) on each diagonal block.
  - No iterative refinement loop: the iterative solver itself is the iterative
    process.
  - Adaptive tolerance scaled with mu: early IPM iterations (large mu) use
    loose tolerances; accuracy tightens as mu shrinks.

Solver backends:
  ScipyMinresSolver — scipy.sparse.linalg.minres (no extra deps)
  PetscFieldSplitSolver — petsc4py KSP MINRES with fieldsplit preconditioner
"""

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg


class IterativeSolver:
  """Base class for iterative KKT solver backends."""

  def solve(
      self,
      kkt: sp.spmatrix,
      rhs: np.ndarray,
      x0: np.ndarray,
      rtol: float,
  ) -> tuple[np.ndarray, int]:
    """Solves kkt @ x = rhs iteratively.

    Args:
      kkt: The KKT matrix (symmetric indefinite).
      rhs: Right-hand side vector.
      x0: Warm-start initial guess.
      rtol: Relative convergence tolerance.

    Returns:
      (solution, iteration_count): iteration_count is 0 on convergence,
      >0 otherwise (scipy convention).
    """
    raise NotImplementedError

  def free(self) -> None:
    pass


class ScipyMinresSolver(IterativeSolver):
  """MINRES solver via scipy.sparse.linalg.minres."""

  def solve(self, kkt, rhs, x0, rtol, **kwargs):
    sol, info = scipy.sparse.linalg.minres(kkt, rhs, x0=x0, rtol=rtol)
    return sol, info


class PetscFieldSplitSolver(IterativeSolver):
  """GMRES solver with ILU preconditioner via petsc4py.

  Uses two PETSc matrices:
    - amat: the true (unregularized) KKT matrix for GMRES matvec
    - pmat: a regularized copy for the ILU preconditioner

  The regularization prevents zero pivots in ILU factorization (the equality
  constraint diagonal entries are -mu, which approaches zero as mu shrinks).
  GMRES still converges to the solution of the true system since amat is used
  for the matrix-vector products.
  """

  _MIN_REG = 1e-8

  def __init__(self, n: int, m: int):
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top
    self._PETSc = petsc4py.PETSc
    self._n = n
    self._m = m
    self._amat = None
    self._pmat = None
    self._ksp = None
    self._b = None
    self._x = None

  def _setup(self, kkt, kkt_reg):
    """Build PETSc Mats and KSP on first call."""
    PETSc = self._PETSc
    n, m = self._n, self._m
    dim = n + m

    # Convert to CSR for PETSc AIJ format.
    self._amat_csr = kkt.tocsr()
    self._pmat_csr = kkt_reg.tocsr()

    self._amat = PETSc.Mat().createAIJWithArrays(
        self._amat_csr.shape,
        (self._amat_csr.indptr, self._amat_csr.indices, self._amat_csr.data),
    )
    self._amat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
    self._amat.assemble()

    self._pmat = PETSc.Mat().createAIJWithArrays(
        self._pmat_csr.shape,
        (self._pmat_csr.indptr, self._pmat_csr.indices, self._pmat_csr.data),
    )
    self._pmat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
    self._pmat.assemble()

    # Define index sets for the two fields.
    self._is0 = PETSc.IS().createGeneral(np.arange(n, dtype=np.int32))
    self._is1 = PETSc.IS().createGeneral(np.arange(n, n + m, dtype=np.int32))

    # GMRES with block-diagonal fieldsplit preconditioner.
    # The (1,1) block P+mu*I is SPD -> ICC gives a good approximation.
    # The (2,2) block -(D+mu*I) is diagonal -> Jacobi is exact.
    # GMRES handles the non-SPD preconditioner (no symmetry requirement).
    self._ksp = PETSc.KSP().create()
    self._ksp.setType(PETSc.KSP.Type.GMRES)
    self._ksp.setOperators(self._amat, self._pmat)

    pc = self._ksp.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    pc.setFieldSplitIS(("0", self._is0), ("1", self._is1))
    pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    # Use the true (unpreconditioned) residual norm for convergence.
    self._ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    # Disable the divergence tolerance (dtol) check. Warm-starting from
    # the previous IPM iteration can give a large initial residual when mu
    # changes significantly, triggering KSP_DIVERGED_DTOL at iteration 0.
    self._ksp.setTolerances(divtol=1e30)

    self._ksp.setUp()
    pc.setUp()

    # Configure sub-KSPs: ICC on (1,1) block, Jacobi on (2,2) block.
    sub_ksps = pc.getFieldSplitSubKSP()
    for sub_ksp in sub_ksps:
      sub_ksp.setType(PETSc.KSP.Type.PREONLY)
    sub_ksps[0].getPC().setType(PETSc.PC.Type.ICC)
    sub_ksps[1].getPC().setType(PETSc.PC.Type.JACOBI)
    for sub_ksp in sub_ksps:
      sub_ksp.setUp()

    # Pre-allocate RHS and solution PETSc vectors.
    self._b = self._amat.createVecRight()
    self._x = self._amat.createVecRight()
    self._sol = np.empty(dim, dtype=np.float64)

  def solve(self, kkt, rhs, x0, rtol, kkt_reg=None):
    if self._amat is None:
      self._setup(kkt, kkt_reg)
    else:
      # Update both PETSc matrices from the scipy CSC matrices.
      amat_csr = kkt.tocsr()
      self._amat_csr.data[:] = amat_csr.data
      self._amat.stateIncrease()

      pmat_csr = kkt_reg.tocsr()
      self._pmat_csr.data[:] = pmat_csr.data
      self._pmat.stateIncrease()

      self._ksp.setOperators(self._amat, self._pmat)
      self._ksp.setUp()

    self._ksp.setTolerances(rtol=rtol)
    self._b.array[:] = rhs
    self._x.array[:] = x0
    self._ksp.setInitialGuessNonzero(True)
    self._ksp.solve(self._b, self._x)

    np.copyto(self._sol, self._x.array)
    reason = self._ksp.getConvergedReason()
    iters = self._ksp.getIterationNumber()
    rnorm = self._ksp.getResidualNorm()
    logging.debug(
        "PETSc KSP: reason=%d, iters=%d, rnorm=%e, rtol=%e",
        reason, iters, rnorm, rtol,
    )
    # PETSc convention: positive reason = converged, negative = diverged.
    info = 0 if reason > 0 else -1
    # Return a copy: the caller (IndirectKktSolver) stores the result and
    # expects it to remain valid across multiple solve calls. Returning
    # the pre-allocated buffer directly would alias it, causing subsequent
    # solves to overwrite the previous result.
    return self._sol.copy(), info

  def free(self):
    if self._ksp is not None:
      self._ksp.destroy()
      self._ksp = None
    for attr in ('_amat', '_pmat', '_is0', '_is1'):
      obj = getattr(self, attr, None)
      if obj is not None:
        obj.destroy()
        setattr(self, attr, None)
    self._b = None
    self._x = None


class IndirectKktSolver:
  """Iterative KKT solver with adaptive tolerance.

  Mirrors DirectKktSolver's interface (.update, .solve, .free) but uses an
  iterative solver instead of a direct factorization.

  Maintains two KKT matrices:
    - _kkt: the true (unregularized) matrix used for the iterative matvec
    - _kkt_reg: a regularized copy used for the preconditioner factorization

  The regularization prevents zero pivots in the preconditioner (equality
  constraint diagonal entries are -mu, approaching 0 as mu shrinks). The
  iterative solver still converges to the true system's solution since the
  matvec uses the unregularized matrix.
  """

  _MIN_REG = 1e-8

  def __init__(
      self,
      *,
      a: sp.spmatrix,
      p: sp.spmatrix,
      z: int,
      atol: float,
      rtol: float,
      solver: IterativeSolver,
  ):
    self.m, self.n = a.shape
    self.z = z
    self._p_diags = p.diagonal()
    self._atol = atol
    self._rtol = rtol
    self._solver = solver

    # Build KKT scaffold with NaN sentinels on diagonals (same as DirectKktSolver).
    n_nans = sp.diags(np.full(self.n, np.nan, dtype=np.float64), format="csc")
    m_nans = sp.diags(np.full(self.m, np.nan, dtype=np.float64), format="csc")
    self._kkt = sp.bmat(
        [[p + n_nans, a.T], [a, m_nans]],
        format="csc",
        dtype=np.float64,
    )
    self._kkt_nan_idxs = np.isnan(self._kkt.data)

    # Regularized copy for preconditioning (same scaffold, separate data).
    self._kkt_reg = self._kkt.copy()

    # Pre-allocate buffers.
    self._true_diags = np.empty(self.n + self.m, dtype=np.float64)
    self._reg_diags = np.empty(self.n + self.m, dtype=np.float64)
    self._kkt_rhs = np.empty(self.n + self.m, dtype=np.float64)

  def update(self, mu: float, s: np.ndarray, y: np.ndarray):
    """Forms the KKT diagonals and injects into both matrices."""
    self._mu = mu
    # True diagonals (before sign flip): [p_diags + mu, h + mu].
    self._true_diags[: self.n] = self._p_diags + mu
    self._true_diags[self.n : self.n + self.z] = mu
    self._true_diags[self.n + self.z :] = s[self.z :] / y[self.z :] + mu

    # Regularized diagonals: clamp small values for stable factorization.
    np.maximum(self._true_diags, self._MIN_REG, out=self._reg_diags)

    # Flip sign of cone block for both.
    self._true_diags[self.n :] *= -1.0
    self._reg_diags[self.n :] *= -1.0

    self._kkt.data[self._kkt_nan_idxs] = self._true_diags
    self._kkt_reg.data[self._kkt_nan_idxs] = self._reg_diags

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the KKT system iteratively."""
    # Negate second block of RHS (same convention as DirectKktSolver).
    np.copyto(self._kkt_rhs, rhs)
    self._kkt_rhs[self.n :] *= -1.0

    # Adaptive tolerance: scale with mu so early iterations are loose.
    # IPM convergence requires solve error = o(mu).
    rhs_norm = np.linalg.norm(self._kkt_rhs, np.inf)
    abs_tol = self._atol + self._rtol * rhs_norm
    # MINRES rtol means ||r||/||b|| <= rtol, so convert our absolute
    # tolerance target to a relative one.
    rtol = abs_tol / max(rhs_norm, 1e-30)

    sol, info = self._solver.solve(
        self._kkt, self._kkt_rhs, x0=warm_start, rtol=rtol,
        kkt_reg=self._kkt_reg,
    )

    if np.any(np.isnan(sol)):
      raise ValueError("Iterative solver returned NaNs.")

    residual_norm = np.linalg.norm(self._kkt @ sol - self._kkt_rhs, np.inf)
    status = "converged" if info == 0 else "non-converged"
    logging.debug(
        "Indirect KKT solve: status=%s, info=%d, res=%e",
        status, info, residual_norm,
    )

    return sol, {
        "solves": 1,
        "final_residual_norm": residual_norm,
        "status": status,
    }

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._kkt @ x

  def free(self):
    self._solver.free()


class CgNormalEqSolver:
  """KKT solver via CG on the normal equations (reduced system).

  Reduces the (n+m)x(n+m) indefinite KKT system:
      [H    A^T] [x]   [r1]
      [A   -D  ] [y] = [r2]
  where H = P + mu*I and D = diag(s/y) + mu*I (for inequalities),

  to an n x n SPD system by eliminating y:
      N @ x = r1 + A^T @ (D^{-1} @ r2)
  where N = H + A^T @ D^{-1} @ A.

  After solving for x, y is recovered: y = D^{-1} @ (A @ x - r2).

  N is applied matrix-free (never formed explicitly) via:
      N @ v = H @ v + A^T @ (D^{-1} @ (A @ v))

  Preconditioner: diag(N) = diag(P) + mu + colsum(A^2 / d_vec).
  """

  def __init__(
      self,
      *,
      a: sp.spmatrix,
      p: sp.spmatrix,
      z: int,
      atol: float,
      rtol: float,
  ):
    self.m, self.n = a.shape
    self.z = z
    self._a = a
    self._at = a.T.tocsc()
    self._p = p
    self._p_diags = p.diagonal()
    self._atol = atol
    self._rtol = rtol

    # Pre-compute A^2 column sums for diagonal preconditioner.
    # diag(A^T diag(1/d) A) = sum_j A_ji^2 / d_j for each column i.
    self._a_sq = a.multiply(a)  # Element-wise A^2, same sparsity.

    # Pre-allocate buffers.
    self._d_vec = np.empty(self.m, dtype=np.float64)  # D diagonal: d_i + mu
    self._d_inv = np.empty(self.m, dtype=np.float64)  # 1 / d_vec
    self._precond = np.empty(self.n, dtype=np.float64)  # diag(N)^{-1}
    self._rhs_x = np.empty(self.n, dtype=np.float64)
    self._sol = np.empty(self.n + self.m, dtype=np.float64)

    # KKT scaffold for residual checking (same as IndirectKktSolver).
    n_nans = sp.diags(np.full(self.n, np.nan, dtype=np.float64), format="csc")
    m_nans = sp.diags(np.full(self.m, np.nan, dtype=np.float64), format="csc")
    self._kkt = sp.bmat(
        [[p + n_nans, a.T], [a, m_nans]],
        format="csc",
        dtype=np.float64,
    )
    self._kkt_nan_idxs = np.isnan(self._kkt.data)
    self._true_diags = np.empty(self.n + self.m, dtype=np.float64)
    self._kkt_rhs = np.empty(self.n + self.m, dtype=np.float64)

  def update(self, mu: float, s: np.ndarray, y: np.ndarray):
    """Computes D diagonal and preconditioner for current iteration."""
    self._mu = mu
    # D diagonal: equality block = mu, inequality block = s/y + mu.
    self._d_vec[:self.z] = mu
    self._d_vec[self.z:] = s[self.z:] / y[self.z:] + mu
    np.reciprocal(self._d_vec, out=self._d_inv)

    # Diagonal preconditioner: diag(N) = diag(P) + mu + colsum(A^2 / d_vec).
    # self._a_sq is A^2 (element-wise), so A^T diag(1/d) A's diagonal is
    # the column sum of A^2 scaled by 1/d_vec.
    self._precond[:] = self._p_diags + mu
    self._precond += self._a_sq.T.dot(self._d_inv)
    np.reciprocal(self._precond, out=self._precond)  # invert for preconditioning

    # Also update the KKT scaffold for residual checking.
    self._true_diags[:self.n] = self._p_diags + mu
    self._true_diags[self.n:self.n + self.z] = mu
    self._true_diags[self.n + self.z:] = s[self.z:] / y[self.z:] + mu
    self._true_diags[self.n:] *= -1.0
    self._kkt.data[self._kkt_nan_idxs] = self._true_diags

  def _normal_eq_matvec(self, v: np.ndarray) -> np.ndarray:
    """Computes N @ v = (P + mu*I) @ v + A^T @ (D^{-1} @ (A @ v))."""
    av = self._a @ v
    av *= self._d_inv
    return self._p @ v + self._mu * v + self._at @ av

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the KKT system via CG on the normal equations."""
    n, m = self.n, self.m

    # Negate second block of RHS (same convention as DirectKktSolver).
    np.copyto(self._kkt_rhs, rhs)
    self._kkt_rhs[n:] *= -1.0
    r1, r2 = self._kkt_rhs[:n], self._kkt_rhs[n:]

    # Reduced RHS: r1 + A^T @ (D^{-1} @ r2).
    self._rhs_x[:] = r1 + self._at @ (self._d_inv * r2)

    # CG tolerance: atol + rtol * ||rhs||, converted to a relative tolerance
    # for scipy.sparse.linalg.cg.
    rhs_norm = np.linalg.norm(self._rhs_x, np.inf)
    abs_tol = self._atol + self._rtol * rhs_norm
    rtol = abs_tol / max(rhs_norm, 1e-30)

    # Matrix-free N operator and diagonal preconditioner.
    N_op = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=self._normal_eq_matvec
    )
    M_op = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=lambda v: self._precond * v
    )

    x_sol, info = scipy.sparse.linalg.cg(
        N_op, self._rhs_x, x0=warm_start[:n], M=M_op, rtol=rtol,
    )

    # Recover y: y = D^{-1} @ (A @ x - r2).
    y_sol = self._d_inv * (self._a @ x_sol - r2)

    self._sol[:n] = x_sol
    self._sol[n:] = y_sol

    # Check residual against full KKT system.
    residual_norm = np.linalg.norm(
        self._kkt @ self._sol - self._kkt_rhs, np.inf
    )
    status = "converged" if info == 0 else "non-converged"
    logging.debug(
        "CG normal eq solve: status=%s, info=%d, res=%e",
        status, info, residual_norm,
    )

    return self._sol.copy(), {
        "solves": 1,
        "final_residual_norm": residual_norm,
        "status": status,
    }

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._kkt @ x

  def free(self):
    pass


class MinresSolver:
  """KKT solver via MINRES with diagonal scaling and iterative refinement.

  Solves the full (n+m)x(n+m) symmetric indefinite KKT system:
      [H    A^T] [x]   [r1]
      [A   -D  ] [y] = [r2]

  Explicitly scales the system as (S @ KKT @ S) @ z = S @ rhs where
  S = diag(1/sqrt(|diag(KKT)|)), then uses iterative refinement on the true
  (unscaled) residual to achieve the target accuracy. MINRES in the scaled
  space converges quickly due to the reduced condition number, while iterative
  refinement ensures the unscaled residual meets the tolerance.
  """

  _MAX_REFINEMENT_STEPS = 10

  def __init__(
      self,
      *,
      a: sp.spmatrix,
      p: sp.spmatrix,
      z: int,
      atol: float,
      rtol: float,
  ):
    self.m, self.n = a.shape
    self.z = z
    self._p_diags = p.diagonal()
    self._atol = atol
    self._rtol = rtol

    # Build KKT scaffold with NaN sentinels on diagonals.
    n_nans = sp.diags(np.full(self.n, np.nan, dtype=np.float64), format="csc")
    m_nans = sp.diags(np.full(self.m, np.nan, dtype=np.float64), format="csc")
    self._kkt = sp.bmat(
        [[p + n_nans, a.T], [a, m_nans]],
        format="csc",
        dtype=np.float64,
    )
    self._kkt_nan_idxs = np.isnan(self._kkt.data)

    # Pre-allocate buffers.
    dim = self.n + self.m
    self._true_diags = np.empty(dim, dtype=np.float64)
    self._scale = np.empty(dim, dtype=np.float64)  # S = 1/sqrt(|diag|)
    self._kkt_rhs = np.empty(dim, dtype=np.float64)
    self._scaled_rhs = np.empty(dim, dtype=np.float64)

  def update(self, mu: float, s: np.ndarray, y: np.ndarray):
    """Forms the KKT diagonals and scaling vector."""
    # Diagonal magnitudes (before sign flip).
    self._true_diags[:self.n] = self._p_diags + mu
    self._true_diags[self.n:self.n + self.z] = mu
    self._true_diags[self.n + self.z:] = s[self.z:] / y[self.z:] + mu

    # Scaling: S = 1/sqrt(|diag|).
    np.sqrt(self._true_diags, out=self._scale)
    np.reciprocal(self._scale, out=self._scale)

    # Flip sign of cone block and inject into KKT.
    self._true_diags[self.n:] *= -1.0
    self._kkt.data[self._kkt_nan_idxs] = self._true_diags

    # Build the LinearOperator once per update (reused across refinement steps).
    dim = self.n + self.m
    self._A_op = scipy.sparse.linalg.LinearOperator(
        (dim, dim), matvec=self._scaled_matvec
    )

  def _scaled_matvec(self, v: np.ndarray) -> np.ndarray:
    """Computes (S @ KKT @ S) @ v."""
    return self._scale * (self._kkt @ (self._scale * v))

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the KKT system via MINRES with iterative refinement.

    First solves the diagonally-scaled system with MINRES, then refines on the
    true (unscaled) residual until it meets the tolerance. Each refinement step
    re-solves the scaled system on the residual, so MINRES sees a small RHS and
    converges quickly (typically in very few iterations).
    """
    n = self.n

    # Negate second block of RHS (same convention as DirectKktSolver).
    np.copyto(self._kkt_rhs, rhs)
    self._kkt_rhs[n:] *= -1.0

    # Tolerance on the true (unscaled) system.
    rhs_norm = np.linalg.norm(self._kkt_rhs, np.inf)
    tolerance = self._atol + self._rtol * rhs_norm

    # Initial MINRES solve on scaled system.
    sol = warm_start.copy()
    total_solves = 0
    status = "non-converged"

    for step in range(self._MAX_REFINEMENT_STEPS):
      # Compute true residual: r = rhs - KKT @ sol.
      residual = self._kkt_rhs - self._kkt @ sol
      residual_norm = np.linalg.norm(residual, np.inf)

      if residual_norm < tolerance:
        status = "converged"
        break

      # Check for stalling.
      if step > 0 and residual_norm >= old_residual_norm:
        logging.debug(
            "MINRES refinement stalled at step %d. Old: %e, New: %e",
            step, old_residual_norm, residual_norm,
        )
        status = "stalled"
        break
      old_residual_norm = residual_norm

      # Scale the residual and solve the correction in the scaled space.
      scaled_residual = self._scale * residual
      scaled_x0 = np.zeros_like(scaled_residual)

      # MINRES rtol for this refinement step.
      scaled_rhs_norm = np.linalg.norm(scaled_residual, np.inf)
      abs_tol = self._atol + self._rtol * scaled_rhs_norm
      rtol = abs_tol / max(scaled_rhs_norm, 1e-30)

      z_correction, info = scipy.sparse.linalg.minres(
          self._A_op, scaled_residual, x0=scaled_x0, rtol=rtol,
      )
      total_solves += 1

      # Unscale and apply correction.
      sol += self._scale * z_correction

    else:
      logging.debug(
          "MINRES refinement did not converge after %d steps. res=%e > tol=%e",
          self._MAX_REFINEMENT_STEPS, residual_norm, tolerance,
      )

    if np.any(np.isnan(sol)):
      raise ValueError("MINRES solver returned NaNs.")

    logging.debug(
        "MINRES solve: status=%s, solves=%d, res=%e",
        status, total_solves, residual_norm,
    )

    return sol.copy(), {
        "solves": total_solves,
        "final_residual_norm": residual_norm,
        "status": status,
    }

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._kkt @ x

  def free(self):
    pass
