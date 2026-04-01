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
"""Direct KKT linear system solvers."""

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

  def free(self):
    pass


class MklPardisoSolver(LinearSolver):
  """Wrapper around pydiso.mkl_solver.MKLPardisoSolver."""

  def __init__(self):
    import pydiso.mkl_solver  # pylint: disable=g-import-not-at-top

    self.mkl_solver = pydiso.mkl_solver
    self.factorization: pydiso.mkl_solver.MKLPardisoSolver | None = None

  def update(self, kkt: sp.spmatrix):
    if self.factorization is None:
      self.factorization = self.mkl_solver.MKLPardisoSolver(
          kkt, matrix_type="real_symmetric_indefinite"
      )
      # Recommended iparms for IPMs from Pardiso docs.
      # These only affect the analysis step so should be set before __init__,
      # but this is not currently possible with the current interface.
      self.factorization.set_iparm(10, 1)
      self.factorization.set_iparm(12, 1)
    else:
      self.factorization.refactor(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    try:
      return self.factorization.solve(rhs)
    except self.mkl_solver.PardisoError as e:
      logging.warning("PardisoError: %s", e)
      logging.warning("Performing analysis and factorization steps again.")
      self.factorization._analyze()  # pylint: disable=protected-access
      self.factorization._factor()  # pylint: disable=protected-access
      return self.factorization.solve(rhs)

  def format(self) -> Literal["csr"]:
    return "csr"


class QdldlSolver(LinearSolver):
  """Wrapper around qdldl.Solver for quasi-definite LDL factorization."""

  def __init__(self):
    import qdldl  # pylint: disable=g-import-not-at-top

    self.qdldl = qdldl
    self.factorization: qdldl.Solver | None = None

  def update(self, kkt: sp.spmatrix):
    if self.factorization is None:
      self.factorization = self.qdldl.Solver(kkt)
    else:
      self.factorization.update(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class ScipySolver(LinearSolver):
  """Wrapper around scipy.linalg.factorized."""

  def __init__(self):
    self.factorization = None

  def update(self, kkt: sp.spmatrix):
    # Use to_csc() to ensure correct format, though usually it's a cheap view.
    self.factorization = sp.linalg.factorized(kkt.tocsc())

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class CholModSolver(LinearSolver):
  """Wrapper around sksparse.cholmod for Cholesky LDLt factorization."""

  def __init__(self):
    import sksparse.cholmod  # pylint: disable=g-import-not-at-top

    self.cholmod = sksparse.cholmod
    self.factorization: sksparse.cholmod.CholeskyFactor | None = None

  def update(self, kkt: sp.spmatrix):
    if self.factorization is None:
      self.factorization = self.cholmod.cholesky(kkt, mode="simplicial")
    else:
      self.factorization.cholesky_inplace(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class EigenSolver(LinearSolver):
  """Wrapper around Eigen Simplicial LDL^T."""

  def __init__(self):
    import nanoeigenpy  # pylint: disable=g-import-not-at-top

    self.nanoeigenpy = nanoeigenpy
    self.solver: nanoeigenpy.SimplicialLDLT | None = None

  def update(self, kkt: sp.spmatrix):
    if self.solver is None:
      self.solver = self.nanoeigenpy.SimplicialLDLT()
      self.solver.analyzePattern(kkt)

    self.solver.factorize(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.solver.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class MumpsSolver(LinearSolver):
  """Wrapper for MUMPS solver (via petsc4py)."""

  def __init__(self):
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top

    self.PETSc = petsc4py.PETSc  # pylint: disable=invalid-name
    self.ksp = self.PETSc.KSP().create()

    # Configure as a direct solver (apply preconditioner only)
    self.ksp.setType(self.PETSc.KSP.Type.PREONLY)
    self.ksp.getPC().setType(self.PETSc.PC.Type.LU)
    self.ksp.getPC().setFactorSolverType("mumps")

    # Allow command-line customization (eg, -mat_mumps_icntl_14 20).
    self.ksp.setFromOptions()

  def update(self, kkt: sp.spmatrix):
    kkt_wrapper = self.PETSc.Mat().createAIJ(
        size=kkt.shape, csr=(kkt.indptr, kkt.indices, kkt.data)
    )
    kkt_wrapper.setOption(self.PETSc.Mat.Option.SYMMETRIC, True)
    kkt_wrapper.setOption(self.PETSc.Mat.Option.SPD, False)
    kkt_wrapper.assemble()

    # Check if KSP already has a matrix defined to determine the flag
    already_factorized = self.ksp.getOperators()[0] is not None
    if already_factorized:
      flag = self.PETSc.Mat.Structure.SAME_NONZERO_PATTERN
    else:
      flag = self.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN

    try:
      self.ksp.setOperators(kkt_wrapper, kkt_wrapper, flag)
    except TypeError:
      # Fallback for older petsc4py API; usually auto-detects reuse
      self.ksp.setOperators(kkt_wrapper, kkt_wrapper)

    # Force factorization (symbolic first time, numeric every time)
    self.ksp.setUp()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    b = self.PETSc.Vec().createWithArray(rhs.ravel())
    x = self.PETSc.Vec().createSeq(rhs.size)
    self.ksp.solve(b, x)
    return x.getArray().real.reshape(rhs.shape)

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self.ksp is not None:
      self.ksp.destroy()
      self.ksp = None


class CuDssSolver(LinearSolver):
  """Wrapper around Nvidia's CuDSS for GPU-accelerated solving."""

  def __init__(self):
    import nvmath  # pylint: disable=g-import-not-at-top

    self.nvmath = nvmath
    self.solver: nvmath.sparse.advanced.DirectSolver | None = None

  def update(self, kkt: sp.spmatrix):
    if self.solver is None:
      sparse_system_type = (
          self.nvmath.sparse.advanced.DirectSolverMatrixType.SYMMETRIC
      )
      # Turn off annoying logs by default.
      logger = logging.getLogger("null")
      logger.disabled = True
      options = self.nvmath.sparse.advanced.DirectSolverOptions(
          sparse_system_type=sparse_system_type, logger=logger
      )
      # RHS must be in column major order (Fortran) for cuDSS.
      dummy_rhs = np.empty(kkt.shape[1], order="F", dtype=np.float64)
      self.solver = self.nvmath.sparse.advanced.DirectSolver(
          kkt, dummy_rhs, options=options
      )
      self.solver.plan()
    else:
      self.solver.reset_operands(a=kkt)

    self.solver.factorize()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    # Ensure RHS is Fortran contiguous for cuDSS expected input format
    rhs_fortran = np.asfortranarray(rhs, dtype=np.float64)
    self.solver.reset_operands(b=rhs_fortran)
    return self.solver.solve()

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self.solver is not None:
      self.solver.free()
      self.solver = None
      # Force clean up any 'zombie' references, in order to avoid cuda errors.
      import gc  # pylint: disable=g-import-not-at-top
      gc.collect(0)  # Run GC only on the youngest generation.


class DirectKktSolver:
  """Direct KKT linear system solver with iterative refinement.

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
    self.p_diags = p.diagonal()
    self.min_static_regularization = min_static_regularization
    self.max_iterative_refinement_steps = max_iterative_refinement_steps
    self.atol = atol
    self.rtol = rtol
    self.solver = solver

    # Pre-allocate KKT scaffold. We use NaNs to mark mutable diagonals.
    n_nans = sp.diags(np.full(self.n, np.nan, dtype=np.float64), format="csc")
    m_nans = sp.diags(np.full(self.m, np.nan, dtype=np.float64), format="csc")

    # Construct the sparse block matrix once.
    self.kkt = sp.bmat(
        [[p + n_nans, a.T], [a, m_nans]],
        format=self.solver.format(),
        dtype=np.float64,
    )
    # Cache indices of the diagonal elements for fast updates.
    self.kkt_nan_idxs = np.isnan(self.kkt.data)

    # Pre-allocate reusable buffers to avoid per-call allocations.
    self.kkt_diags = np.empty(self.n + self.m, dtype=np.float64)  # [p_diags+mu, s/y+mu]
    self._true_diags = np.empty(self.n + self.m, dtype=np.float64)
    self._reg_diags = np.empty(self.n + self.m, dtype=np.float64)
    self.kkt_rhs = np.empty(self.n + self.m, dtype=np.float64)    # RHS with cone block negated

  def update(self, mu: float, s: np.ndarray, y: np.ndarray):
    """Forms the KKT matrix diagonals and factorizes it.

    This method employs an optimization to avoid copying the full sparse KKT
    matrix. It temporarily injects the regularized diagonals for the solver,
    then immediately restores the true diagonals for residual calculation.

    Args:
      mu: The barrier parameter.
      s: The slack variables.
      y: The dual variables for the conic constraints.
    """
    # Fill KKT diagonals: [p_diags + mu, h + mu] where h = [0]*z ++ s/y.
    # KKT form: [P+mu*I, A'; A, -(D+mu*I)]
    kkt_diags = self.kkt_diags
    kkt_diags[: self.n] = self.p_diags
    kkt_diags[self.n : self.n + self.z] = 0.0
    kkt_diags[self.n + self.z :] = s[self.z :] / y[self.z :]
    kkt_diags += mu

    # "True" diagonals for accurate residual calculation (no regularization).
    np.copyto(self._true_diags, kkt_diags)
    # "Regularized" diagonals for stable factorization.
    np.maximum(self._true_diags, self.min_static_regularization, out=self._reg_diags)
    # Flip the sign of the cone variables.
    self._true_diags[self.n :] *= -1.0
    self._reg_diags[self.n :] *= -1.0

    # Inject regularized diagonals so the factorization is numerically stable,
    # then immediately restore the true (unregularized) diagonals. This means
    # the stored kkt matrix reflects the exact problem, so residuals computed
    # during iterative refinement (kkt_rhs - kkt @ sol) measure the true error.
    # Refinement then implicitly corrects for the regularization bias introduced
    # by the factorization, converging to the unregularized solution.
    self.kkt.data[self.kkt_nan_idxs] = self._reg_diags
    self.solver.update(self.kkt)
    self.kkt.data[self.kkt_nan_idxs] = self._true_diags

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the linear system with the given factorization.

    Performs iterative refinement to improve the solution accuracy.

    Args:
      rhs: The right-hand side of the linear system.
      warm_start: A warm-start for the solution.

    Returns:
      A tuple containing:
        - sol: The solution vector.
        - A dictionary with solve statistics including:
          - "solves": The number of linear solves performed.
          - "final_residual_norm": The final infinity norm of the residual.
          - "status": The status of the iterative refinement ("converged",
            "non-converged", or "stalled").

    Raises:
      ValueError: If the solution contains NaN values.
    """
    # Adjust RHS to match the quasidefinite KKT form (second block negated).
    # Use pre-allocated buffer to avoid a copy allocation on every call.
    np.copyto(self.kkt_rhs, rhs)
    self.kkt_rhs[self.n :] *= -1.0
    tolerance = self.atol + self.rtol * np.linalg.norm(self.kkt_rhs, np.inf)

    # Initial sol and residual.
    sol = warm_start.copy()
    residual = self.kkt_rhs - self.kkt @ sol
    residual_norm = np.linalg.norm(residual, np.inf)

    # Iterative refinement loop.
    status, solves = "non-converged", 0
    # max_iterative_refinement_steps >= 1 so we always do at least one solve.
    for solves in range(1, self.max_iterative_refinement_steps + 1):
      # Perform correction step using the linear system solver.
      old_residual_norm = residual_norm
      sol += self.solver.solve(residual)
      residual = self.kkt_rhs - self.kkt @ sol
      residual_norm = np.linalg.norm(residual, np.inf)

      # Check for convergence.
      if residual_norm < tolerance:
        status = "converged"
        break

      # Check for stalling (residual not improving).
      if residual_norm >= old_residual_norm:
        logging.debug(
            "Iterative refinement stalled at step %d. Old res: %e, New res: %e",
            solves,
            old_residual_norm,
            residual_norm,
        )
        status = "stalled"
        break
    else:
      logging.debug(
          "Iterative refinement did not converge after %d solves."
          " Final residual: %e > tolerance: %e",
          solves,
          residual_norm,
          tolerance,
      )

    if np.any(np.isnan(sol)):
      raise ValueError("Linear solver returned NaNs.")

    logging.debug(
        "KKT solve: status=%s, solves=%d, res=%e", status, solves, residual_norm
    )

    return sol, {
        "solves": solves,
        "final_residual_norm": residual_norm,
        "status": status,
    }

  def free(self):
    """Frees the solver resources."""
    self.solver.free()
