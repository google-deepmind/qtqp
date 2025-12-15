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

import numpy as np
import scipy.sparse as sp

class LinearSolver(Protocol):
  """Protocol defining the interface for linear solvers."""

  def update(self, kkt: sp.spmatrix) -> None:
    """Factorizes or refactorizes the KKT matrix."""
    ...

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system."""
    ...

  def format(self) -> str:
    """Returns the expected sparse matrix format (eg, 'csc' or 'csr')."""
    ...

class PetscMinres(LinearSolver):
  """MinRes solver using PETSc."""
  def __init__(self) -> None:
    """Initializes the MinRes solver."""
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top

    self.module = petsc4py.PETSc
    self.ksp = self.module.KSP().create()

    self.warm_start : np.ndarray | None = None

    # Configure as a direct solver (apply preconditioner only)
    self.ksp.setType("minres")
    self.ksp.getPC().setType("none")

    # self.ksp.getPC().setType("fieldsplit")
    # self.ksp.getPC().setFieldSplitType(self.module.PC.CompositeType.SCHUR)

    # Allow command-line customization (eg, -mat_mumps_icntl_14 20).
    

    # Confgure kkt matrix and solutions
    self.kkt_wrapper : self.module.Mat | None = None
    self.rhs : self.module.Vec | None = None
    self.sol : self.module.Vec | None = None

  def update(self, kkt: sp.spmatrix) -> None:
    """Updates the preconditioner."""
    # TODO: Need to use IS() for updating wrapper instead of creating a new one
    self.kkt_wrapper = self.module.Mat().createAIJ(
        size=kkt.shape, csr=(kkt.indptr, kkt.indices, kkt.data)
    )
    self.kkt_wrapper.setOption(self.module.Mat.Option.SYMMETRIC, True)
    self.kkt_wrapper.setOption(self.module.Mat.Option.SPD, False)
    self.kkt_wrapper.assemble()

    # update preconditioner
    self.ksp.setOperators(self.kkt_wrapper)
    self.ksp.setFromOptions()
    self.ksp.setTolerances(rtol=1e-8, atol=1e-8, max_it=100000)
    # self.ksp.SetUp()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system."""
    if self.rhs is None:
        self.rhs = self.module.Vec().createWithArray(rhs.ravel())
        self.sol = self.module.Vec().createSeq(rhs.size)
    # else:
    self.rhs.setArray(rhs)
    if self.warm_start is None:
        logging.warning("No warm start provided to indirect solver, using zero vector")
        self.warm_start = np.zeros(rhs.size)

    self.sol.setArray(self.warm_start)
    self.ksp.solve(self.rhs, self.sol)

    iters = self.ksp.getIterationNumber()
    # print(f"Iterations: {iters}")

    x = self.sol.getArray().copy().real.reshape(rhs.shape)
    return x, 0

  def format(self) -> Literal["csr"]:
    """Returns the expected sparse matrix format (eg, 'csc' or 'csr')."""
    return "csr"

class ScipyMinres(LinearSolver):
  """MinRes solver."""
  def __init__(self) -> None:
    """Initializes the MinRes solver."""
    self.warm_start : np.ndarray | None = None
    self.solver = sp.linalg.minres
    self.kkt : sp.spmatrix | None = None

  def update(self, kkt: sp.spmatrix) -> None:
    """Updates the preconditioner."""
    if self.kkt is None:
      self.kkt = kkt

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system."""
    if self.warm_start is None:
      self.warm_start = np.zeros(self.kkt.shape[1])
    x, exitflag = self.solver(
        A=self.kkt, 
        b=rhs, 
        x0=self.warm_start,
        rtol=1e-8
    )
    return x, exitflag

  def format(self) -> str:
    """Returns the expected sparse matrix format (eg, 'csc' or 'csr')."""
    return "csc"

class IndirectKktSolver:
  """Indirect KKT linear system solver.

  Solves a quasidefinite KKT system:
      [ P + mu * I       A.T     ] [ x ]   [ rhs_x ]
      [     A      -(D + mu * I) ] [ y ] = [ -rhs_y ]
  where D is a diagonal matrix derived from slacks and duals.
  """

  def __init__(
      self,
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
    self.solver = PetscMinres()

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
    self.solver.update(self.kkt)
    
    # 2. Restore true values for subsequent residual checks in `solve()`.
    self.kkt.data[self.kkt_nan_idxs] = true_diags

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the linear system with the given preconditioner and data"""
    rhs = rhs.copy()
    rhs[self.n :] *= -1.0
    self.solver.warm_start = warm_start 
    x, exitflag = self.solver.solve(rhs)
    res_norm = np.linalg.norm(self.kkt @ x - rhs, np.inf)
    return x, {
        "solves": 1,
        "final_residual_norm": res_norm,
        "status": "converged" if exitflag == 0 else "non-converged"
    }
      