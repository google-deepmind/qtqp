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
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp


class MklPardisoSolver:
  """Wrapper around pydiso.mkl_solver.MKLPardisoSolver.

  This class provides an interface to the MKL Pardiso solver for
  symmetric indefinite matrices.
  """

  def __init__(self):
    """Initializes the MklPardisoSolver."""
    import pydiso.mkl_solver  # pylint: disable=g-import-not-at-top

    self.module = pydiso.mkl_solver
    self.factorization: self.module.MKLPardisoSolver | None = None

  def update(self, kkt: sp.spmatrix):
    """Factorizes or refactorizes the KKT matrix.

    Args:
      kkt: The sparse KKT matrix to be factorized.
    """
    if self.factorization is None:
      self.factorization = self.module.MKLPardisoSolver(
          kkt, matrix_type="real_symmetric_indefinite"
      )
    else:
      self.factorization.refactor(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system using the factorized KKT matrix.

    Args:
      rhs: The right-hand side vector.

    Returns:
      The solution vector.
    """
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csr"]:
    """Returns the sparse matrix format expected by the solver."""
    return "csr"


class QdldlSolver:
  """Wrapper around qdldl.Solver.

  This class provides an interface to the QDLDL solver.
  """

  def __init__(self):
    """Initializes the QdldlSolver."""
    import qdldl  # pylint: disable=g-import-not-at-top

    self.module = qdldl
    self.factorization: self.module.Solver | None = None

  def update(self, kkt: sp.spmatrix):
    """Factorizes or updates the factorization of the KKT matrix.

    Args:
      kkt: The sparse KKT matrix.
    """
    if self.factorization is None:
      self.factorization = self.module.Solver(kkt)
    else:
      self.factorization.update(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system using the factorized KKT matrix.

    Args:
      rhs: The right-hand side vector.

    Returns:
      The solution vector.
    """
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csc"]:
    """Returns the sparse matrix format expected by the solver."""
    return "csc"


class ScipySolver:
  """Wrapper around scipy.linalg.factorized.

  This class uses `scipy.linalg.factorized` for solving linear systems.
  """

  def update(self, kkt: sp.spmatrix):
    """Factorizes the KKT matrix.

    Args:
      kkt: The sparse KKT matrix to be factorized.
    """
    self.factorization = sp.linalg.factorized(kkt.tocsc())

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system using the factorized KKT matrix.

    Args:
      rhs: The right-hand side vector.

    Returns:
      The solution vector.
    """
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    """Returns the sparse matrix format expected by the solver."""
    return "csc"


class CholModSolver:
  """Wrapper around sksparse.cholmod.

  This class provides an interface to the CHOLMOD sparse Cholesky factorization
  solver.
  """

  def __init__(self):
    """Initializes the CholModSolver."""
    from sksparse import cholmod  # pylint: disable=g-import-not-at-top

    self.module = cholmod
    self.factorization: self.module.CholeskyFactor | None = None

  def update(self, kkt: sp.spmatrix):
    """Factorizes or updates the factorization of the KKT matrix.

    Args:
      kkt: The sparse KKT matrix.
    """
    if self.factorization is None:
      self.factorization = self.module.cholesky(kkt, mode="simplicial")
    else:
      self.factorization.cholesky_inplace(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system using the factorized KKT matrix.

    Args:
      rhs: The right-hand side vector.

    Returns:
      The solution vector.
    """
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    """Returns the sparse matrix format expected by the solver."""
    return "csc"


class CuDssSolver:
  """Wrapper around Nvidia's CuDSS for GPUs.

  This class provides an interface to the NVIDIA cuDSS library for solving
  sparse linear systems on GPUs.
  """

  def __init__(self):
    """Initializes the CuDssSolver."""
    import nvmath.sparse  # pylint: disable=g-import-not-at-top

    self.module = nvmath.sparse
    self.solver: self.module.advanced.DirectSolver | None = None

  def update(self, kkt: sp.spmatrix):
    """Factorizes the KKT matrix and stores the factorization.

    Args:
      kkt: The sparse KKT matrix.
    """
    if self.solver is None:
      sparse_system_type = self.module.advanced.DirectSolverMatrixType.SYMMETRIC
      # Turn off annoying logs.
      logger = logging.getLogger("null")
      logger.disabled = True
      options = self.module.advanced.DirectSolverOptions(
          sparse_system_type=sparse_system_type, logger=logger
      )
      # RHS must be in column major order (Fortran).
      dummy = np.ones(kkt.shape[1], order="F")
      self.solver = self.module.advanced.DirectSolver(
          kkt, dummy, options=options
      )
      self.solver.plan()
    else:
      self.solver.reset_operands(a=kkt)

    self.solver.factorize()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system using the factorized KKT matrix.

    Args:
      rhs: The right-hand side vector.

    Returns:
      The solution vector.
    """
    self.solver.reset_operands(b=np.asfortranarray(rhs))
    return self.solver.solve()

  def format(self) -> Literal["csr"]:
    """Returns the sparse matrix format expected by the solver."""
    return "csr"

  def __del__(self):
    """Frees the solver resources."""
    if self.solver is not None:
      self.solver.free()


class DirectKktSolver:
  """Direct KKT linear system solver.

  This class constructs and solves KKT linear systems arising in interior-point
  methods for quadratic programming. It supports iterative refinement.
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
      solver: Any,
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

    # P and A are in csc so these should be csc.
    n_nans = sp.diags(np.nan * np.ones(self.n), format="csc")
    m_nans = sp.diags(np.nan * np.ones(self.m), format="csc")
    self.kkt = sp.bmat(
        [
            [p + n_nans, a.T],
            [a, m_nans],
        ],
        format=self.solver.format(),
    )
    self.kkt_nan_idxs = np.isnan(self.kkt.data)

  def update(self, mu: float, s: np.ndarray, y: np.ndarray):
    """Forms the KKT matrix and factorizes it.

    Args:
      mu: The barrier parameter.
      s: The slack variables.
      y: The dual variables for the conic constraints.
    """
    h = np.concatenate([np.zeros(self.z), s[self.z :] / y[self.z :]])
    true_diags = np.concatenate([self.p_diags, h]) + mu
    # Add regularization to the diagonal.
    reg_diags = np.maximum(true_diags, self.min_static_regularization)
    # Flip the sign of the cone variables.
    true_diags[self.n :] *= -1
    # Update the matrix with the true diagonal values.
    self.kkt.data[self.kkt_nan_idxs] = true_diags
    # Flip the sign of the regularization.
    reg_diags[self.n :] *= -1
    regularized_kkt = self.kkt.copy()
    regularized_kkt.data[self.kkt_nan_idxs] = reg_diags
    # Refactor the *regularized* KKT matrix.
    self.solver.update(regularized_kkt)

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
    rhs = rhs.copy()
    rhs[self.n :] *= -1
    sol = warm_start.copy()
    residual = rhs - self.kkt @ sol
    residual_norm = np.linalg.norm(residual, np.inf)
    i = 0
    # Enter the iterative refinement loop.
    while residual_norm > self.atol + self.rtol * np.linalg.norm(rhs, np.inf):
      logging.debug("lin sys solves=%d, residual: %s", i, residual_norm)
      if i > self.max_iterative_refinement_steps:  # Always do 1 solve.
        logging.debug(
            "Iterative refinement did not converge after %d steps. This may be"
            " caused by poor numerical conditioning.",
            self.max_iterative_refinement_steps,
        )
        status = "non-converged"
        break
      i += 1

      # Perform refinement step
      correction = self.solver.solve(residual)
      new_sol = sol + correction
      new_residual = rhs - self.kkt @ new_sol
      new_residual_norm = np.linalg.norm(new_residual, np.inf)

      # Check for stalling
      if new_residual_norm >= residual_norm:
        logging.debug(
            "Iterative refinement stalled at step %d. This may be caused by"
            " poor numerical conditioning. Previous residual: %s, new: %s.",
            i + 1,
            residual_norm,
            new_residual_norm,
        )
        status = "stalled"
        break

      sol = new_sol
      residual = new_residual
      residual_norm = new_residual_norm
    else:
      status = "converged"

    if np.any(np.isnan(sol)):
      raise ValueError(f"NaN in sol: {sol=}")

    logging.debug("lin sys solves=%d, residual: %s", i, residual_norm)

    return sol, {
        "solves": i,
        "final_residual_norm": residual_norm,
        "status": status,
    }
