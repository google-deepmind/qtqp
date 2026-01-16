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
"""Indirect KKT linear system solvers."""

import logging
from typing import Any, Literal, Protocol

from . import LinearSolver

import numpy as np
import scipy.sparse as sp


class PetscGMRES(LinearSolver):
  """MinRes solver using PETSc."""
  def __init__(self) -> None:
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top
    self.module = petsc4py.PETSc

    self.n: int | None = None
    self.m: int | None = None

    # self.ksp = self.module.KSP().create()
    self.warm_start: np.ndarray | None = None

    # Use MINRES for symmetric indefinite systems
    # self.ksp.setType("gmres")
    self.ksp = self.module.KSP().create()
    self.ksp.setType("fgmres")

    # Mat/Vec handles
    self.kkt_wrapper: self.module.Mat | None = None
    self.rhs: self.module.Vec | None = None
    self.sol: self.module.Vec | None = None
    self.is_x: self.module.IS | None = None
    self.is_y: self.module.IS | None = None

    # Cached sub-blocks for preconditioner
    self.Kxx: self.module.Mat | None = None       # = G
    self.Kyy_pos: self.module.Mat | None = None   # = -Kyy = W (SPD)

    self._setup_done = False

  def _setup(self, kkt: sp.spmatrix) -> None:

    N = int(self.n + self.m)
    self.kkt = self.module.Mat().createAIJ(size=(N, N), comm=self.module.COMM_SELF)
    self.kkt.setPreallocationCSR((kkt.indptr, kkt.indices))  # pattern only
    # forbid pattern changes (debug/safety); you can turn this off later
    self.kkt.setOption(self.module.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
    self.kkt.setOption(self.module.Mat.Option.SYMMETRIC, True)
    self.kkt.setOption(self.module.Mat.Option.SPD, False)
    self.kkt.setUp()
    self.kkt.assemble()
    self.ksp.setOperators(self.kkt) 

    self._block_schur_preconditioner()

    self._setup_done = True

  def update(self, kkt: sp.spmatrix) -> None:
    if not self._setup_done:
      self._setup(kkt)
    
    # overwrite values (fast)
    self.kkt.setValuesCSR(
      kkt.indptr, kkt.indices, kkt.data, 
      addv=self.module.InsertMode.INSERT_VALUES
    ) 
    self.kkt.assemblyBegin()
    self.kkt.assemblyEnd()

  def solve(self, rhs: np.ndarray, warm_start: np.ndarray) -> np.ndarray: 
    """Solves the linear system.""" 
    if self.rhs is None: 
      self.rhs = self.module.Vec().createWithArray(rhs.ravel()) 
      self.sol = self.module.Vec().createSeq(rhs.size) 
    self.rhs.setArray(rhs) 
    
    self.sol.setArray(warm_start) 
    
    self.ksp.solve(self.rhs, self.sol) 
    
    # eig = self.ksp.computeEigenvalues()
    # logging.warning(f"min: {np.min(eig)}, max: {np.max(eig)}")
    iters = self.ksp.getIterationNumber() 
    reason = self.ksp.getConvergedReason() 
    if reason == 4:
      logging.warning("Indirect solver hit max iterations")
    elif reason < 0:
      logging.warning(f"Indirect solver diverged, reason: {reason}")
    # else:
      # logging.warning(f"Indirect solver converged in {iters}/10000 iterations, reason: {reason}")
    # print(f"Reason: {reason} with iters: {iters}") 
    
    x = self.sol.getArray().copy().real.reshape(rhs.shape) 
    # self.ksp.getPC().destroy()
    # self.ksp.destroy()
    return x, 0

  def _none_PC(self):
    self.ksp.getPC().setType("none")

  def _block_jacobi_preconditioner(self):
    pc = self.ksp.getPC() 

    # Define the two fields by index sets: [0..1] and [2..6]
    is_u = self.module.IS().createStride(self.n, first=0, step=1)
    is_l = self.module.IS().createStride(self.m, first=self.n, step=1)

    pc.setType("fieldsplit")
    # IMPORTANT: lambda first -> makes A00 = H
    pc.setFieldSplitIS(("lambda", is_l), ("u", is_u))
    pc.setFieldSplitType(self.module.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE)

    self.ksp.setFromOptions()
    self.ksp.setTolerances(rtol=1e-8, atol=0, max_it=100)
    # self.ksp.setComputeEigenvalues(True)
    self.ksp.setUp()

    ksp_l, ksp_s = pc.getFieldSplitSubKSP()   # ksp_l for H-block, ksp_s for Schur on u
    ksp_l.setType("preonly")
    ksp_l.getPC().setType("jacobi")   # exact for diagonal H

    ksp_s.setType("preonly")                       # Schur is SPD
    ksp_s.getPC().setType("cholesky")              # approximate inverse of SPD Schur
    # ksp_s.setTolerances(rtol=1e-6, atol=0, max_it=1000)
    # ksp_s.getPC().setFactorLevels(2)

  def _block_schur_preconditioner(self):
    pc = self.ksp.getPC() 

    # Define the two fields by index sets: [0..1] and [2..6]
    is_u = self.module.IS().createStride(self.n, first=0, step=1)
    is_l = self.module.IS().createStride(self.m, first=self.n, step=1)

    pc.setType("fieldsplit")
    # IMPORTANT: lambda first -> makes A00 = H
    pc.setFieldSplitIS(("lambda", is_l), ("u", is_u))

    pc.setFieldSplitType(self.module.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(self.module.PC.FieldSplitSchurFactType.LOWER)

    # Since A00 is diagonal H, SELFP becomes very strong (often exact Schur)
    pc.setFieldSplitSchurPreType(self.module.PC.FieldSplitSchurPreType.SELFP)

    self.ksp.setFromOptions()
    self.ksp.setTolerances(rtol=1e-10, atol=0, max_it=100)
    # self.ksp.setComputeEigenvalues(True)
    self.ksp.setUp()

    ksp_l, ksp_s = pc.getFieldSplitSubKSP()   # ksp_l for H-block, ksp_s for Schur on u
    ksp_l.setType("preonly")
    ksp_l.getPC().setType("jacobi")   # exact for diagonal H

    ksp_s.setType("preonly")                       # Schur is SPD
    ksp_s.getPC().setType("cholesky")              # approximate inverse of SPD Schur
    ksp_s.getPC().setFactorPivot(1e-8)

  def _custom_PC(self):
    """ILU preconditioner on full K (often good for GMRES, not SPD)."""
    pc_ilu = self.ksp.getPC()
    pc_ilu.setType("ilu")
    pc_ilu.setFactorPivot(1e-8, True)  # enable pivoting
    pc_ilu.setFactorLevels(5)

    self.ksp.setFromOptions()
    self.ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=10000)
    # self.ksp.setComputeEigenvalues(True)
    self.ksp.setUp()

  def format(self) -> Literal["csr"]:
    """Returns the expected sparse matrix format."""
    return "csr"

  def type(self) -> Literal["indirect"]:
    """Returns the type of the solver."""
    return "indirect"


class ScipyMinres(LinearSolver):
  """MinRes solver."""
  def __init__(self) -> None:
    self.warm_start: np.ndarray | None = None
    self.solver = sp.linalg.minres
    self.kkt: sp.spmatrix | None = None

    # cached blocks / sizes
    self.p: sp.spmatrix | None = None   # will store G = K[:n,:n]
    self.a: sp.spmatrix | None = None   # will store A = K[:n,n:]
    self._n: int | None = None
    self._m: int | None = None

    # PRECONDITIONER (this will be M^{-1} as a LinearOperator)
    self.M: sp.linalg.LinearOperator | None = None

  def update(self, kkt: sp.spmatrix, s: np.ndarray) -> None:
    """Updates the preconditioner.

    Assumes len(s) equals the size of the dual/constraint block (m),
    so KKT dimension is (n+m) x (n+m) with n = N - m.
    """
    self.kkt = kkt

    N = kkt.shape[0]
    m = int(np.asarray(s).size)
    n = N - m
    if n <= 0:
      raise ValueError(f"Bad partition inferred from s: N={N}, m={m} => n={n}")

    self._n, self._m = n, m

    # Extract blocks
    G = kkt[:n, :n].tocsc()
    A = kkt[:n, n:].tocsc()
    K22 = kkt[n:, n:].tocsc()  # this is -(H+mu I) = -W

    self.p = G
    self.a = A

    # Diagonals
    diagG = np.asarray(G.diagonal()).ravel()
    # W should be positive diagonal: W = -K22
    diagW = -np.asarray(K22.diagonal()).ravel()

    # Stabilize (avoid division by zero / negative due to numerical noise)
    epsG = 1e-12 + 1e-9 * (np.mean(np.abs(diagG)) + 1.0)
    epsS = 1e-12 + 1e-9 * (np.mean(np.abs(diagW)) + 1.0)

    diagG_safe = np.where(diagG > epsG, diagG, epsG)
    inv_diagG = 1.0 / diagG_safe

    # diag(A^T diag(G)^{-1} A) = (A.^2)^T * inv_diagG
    # (A.multiply(A)) squares entrywise but stays sparse
    A2 = A.multiply(A)
    diag_AtDinvA = np.asarray(A2.T @ inv_diagG).ravel()

    diagS = diagW + diag_AtDinvA
    diagS_safe = np.where(diagS > epsS, diagS, epsS)
    inv_diagS = 1.0 / diagS_safe

    # Build M^{-1} as an SPD LinearOperator
    def matvec(z: np.ndarray) -> np.ndarray:
      z1 = z[:n]
      z2 = z[n:]
      y1 = inv_diagG * z1
      y2 = inv_diagS * z2
      return np.concatenate([y1, y2])

    self.M = sp.linalg.LinearOperator(shape=(N, N), matvec=matvec, dtype=kkt.dtype)

    # keep warm-start length consistent
    if self.warm_start is None or self.warm_start.size != N:
      self.warm_start = np.zeros(N, dtype=float)

  def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, int]:
    """Solves the linear system."""
    if self.kkt is None:
      raise RuntimeError("Call update(kkt, s) before solve(rhs).")
    if self.M is None:
      raise RuntimeError("Preconditioner not built; call update(kkt, s) first.")
    if self.warm_start is None:
      self.warm_start = np.zeros(self.kkt.shape[1])

    x, exitflag = self.solver(
        A=self.kkt,
        b=rhs,
        M=self.M,          # <-- preconditioner (approx inverse), SPD
        x0=self.warm_start,
        rtol=1e-10,
        # show=True,
    )
    self.warm_start = x
    return x, exitflag

  def format(self) -> Literal["csc"]:
    return "csc"

  def type(self) -> Literal["indirect"]:
    """Returns the type of the solver."""
    return "indirect"

class IndirectKktSolver:
  """Indirect KKT linear system solver.

  Solves a quasidefinite KKT system:
      [ P + mu * I       A.T     ] [ x ]   [ rhs_x ]
      [     A      -(D + mu * I) ] [ y ] = [ -rhs_y ]
  where D is a diagonal matrix derived from slacks and duals.
  """

  def __init__(
      self,
      *,
      a: sp.spmatrix,
      p: sp.spmatrix,
      z: int,
      min_static_regularization: float,
      max_iterative_refinement_steps: int,
      atol: float,
      rtol: float,
      solver: LinearSolver,
  ):
    """Initializes the DirectKktSolver.

    Args:
      a: The constraint matrix from the QP.
      p: The quadratic cost matrix from the QP.
      z: The number of zero elements in the diagonal of the barrier term.
      min_static_regularization: Minimum static regularization to add to the
        diagonal of the KKT matrix.
      max_iterative_refinement_steps: Maximum number of iterative refinement
        steps to perform.
      atol: Absolute tolerance for iterative refinement.
      rtol: Relative tolerance for iterative refinement.
      solver: An instance of a direct solver class (e.g., MklPardisoSolver).
    """
    # Create KKT scaffold with NaNs where we will update values each iteration.
    self.m, self.n = a.shape
    self.z = z
    self.p = p
    self.p_diags = p.diagonal()
    self.min_static_regularization = min_static_regularization
    self.atol = atol
    self.rtol = rtol
    self.solver = PetscGMRES()
    # self.solver = ScipyMinres()

    self.solver.a = a
    self.solver.p = p

    # Pre-allocate KKT scaffold. We use NaNs to mark mutable diagonals.
    n_nans = sp.diags(np.full(self.n, np.nan, dtype=np.float64), format=self.solver.format())
    m_nans = sp.diags(np.full(self.m, np.nan, dtype=np.float64), format=self.solver.format())

    # Construct the sparse block matrices once.
    self.kkt = sp.bmat(
        [[p + n_nans, a.T], [a, m_nans]],
        format=self.solver.format(),
        dtype=np.float64,
    )

    # Cache indices of the diagonal elements for fast updates.
    self.kkt_nan_idxs = np.isnan(self.kkt.data)

  def update(self, mu: float, s: np.ndarray, y: np.ndarray) -> None:
    """Forms the KKT matrix diagonals and preconditioner.

    Args:
      mu: The barrier parameter.
      s: The slack variables.
      y: The dual variables for the conic constraints.
    """
    # Calculate the dynamic diagonal block D = s / y for inequality rows.
    # For equality rows (first z), the diagonal is 0.
    h = np.concatenate([np.zeros(self.z), s[self.z :] / y[self.z :]])
    # "True" diagonals for accurate residual calculation (no regularization).
    # KKT form: [P+mu*I, A'; A, -(D+mu*I)]
    true_diags = np.concatenate([self.p_diags, h]) + mu
    # "Regularized" diagonals for stable factorization.
    reg_diags = np.maximum(true_diags, self.min_static_regularization)
    # Flip the sign of the cone variables.
    true_diags[self.n :] *= -1.0
    reg_diags[self.n :] *= -1.0
  
    # 1. Inject regularized values for the linear solve.
    self.kkt.data[self.kkt_nan_idxs] = reg_diags
    self.solver.update(self.kkt, s)
    
    # 2. Restore true values for subsequent residual checks in `solve()`.
    self.kkt.data[self.kkt_nan_idxs] = true_diags

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the linear system with the given preconditioner and data"""
    rhs = rhs.copy()
    rhs[self.n :] *= -1.0
    # self.solver.warm_start = warm_start 
    x, exitflag = self.solver.solve(rhs, warm_start)
    res_norm = np.linalg.norm(self.kkt @ x - rhs, np.inf)
    return x, {
        "solves": 1,
        "final_residual_norm": res_norm,
        "status": "converged" if exitflag == 0 else "non-converged"
    }
      