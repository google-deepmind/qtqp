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
"""KKT solver and utilities."""

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qtqp.linear import LinearSolver


class KKTSolver:
  """Unified KKT linear system solver.

  Solves a quasidefinite KKT system:
      [ P + mu * I       A.T     ] [ x ]   [ rhs_x ]
      [     A      -(D + mu * I) ] [ y ] = [ -rhs_y ]
  where D is a diagonal matrix derived from slacks and duals.

  Automatically applies iterative refinement for direct solvers, while
  indirect (iterative) solvers handle convergence internally.
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
    """Initializes the KKTSolver.

    Args:
      a: The constraint matrix from the QP.
      p: The quadratic cost matrix from the QP.
      z: The number of zero elements in the diagonal of the barrier term.
      min_static_regularization: Minimum static regularization to add to the
        diagonal of the KKT matrix.
      max_iterative_refinement_steps: Maximum number of iterative refinement
        steps to perform (only used for direct solvers).
      atol: Absolute tolerance for iterative refinement.
      rtol: Relative tolerance for iterative refinement.
      solver: An instance of a linear solver (direct or indirect).
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

  def update(self, mu: float, s: np.ndarray, y: np.ndarray) -> None:
    """Forms the KKT matrix diagonals and updates the solver.

    This method employs an optimization to avoid copying the full sparse KKT
    matrix. It temporarily injects the regularized diagonals for the solver,
    then immediately restores the true diagonals for residual calculation.

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

    # 1. Inject regularized values for the solver update.
    self.kkt.data[self.kkt_nan_idxs] = reg_diags

    if self.solver.type() == 'indirect' and self.solver.n is None:
      self.solver.n = self.n
      self.solver.m = self.m
    self.solver.update(self.kkt)

    # Estimate spectral norm for GMRES monitoring.
    # σ_min estimation via ARPACK fails on quasi-definite matrices, so we
    # report σ_max and the diagonal condition (max/min abs diagonal) as a proxy.
    try:
      sigma_max = spla.svds(
          self.kkt, k=1, which="LM", return_singular_vectors=False
      )[0]
      diag_abs = np.abs(reg_diags)
      diag_cond = diag_abs.max() / diag_abs.min() if diag_abs.min() > 0 else np.inf
      # logging.warning(
      #     "KKT σ_max=%.2e, diag_cond=%.2e (diag_max=%.2e, diag_min=%.2e)",
      #     sigma_max, diag_cond, diag_abs.max(), diag_abs.min(),
      # )
    except Exception as e:
      # logging.warning("Failed to estimate KKT spectral norm: %s", e)
      pass

    # 2. Restore true values for subsequent residual checks in `solve()`.
    self.kkt.data[self.kkt_nan_idxs] = true_diags

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the linear system.

    For direct solvers, performs iterative refinement to improve accuracy.
    For indirect solvers, delegates directly to the iterative solver.

    Args:
      rhs: The right-hand side of the linear system.
      warm_start: A warm-start for the solution.

    Returns:
      A tuple containing:
        - sol: The solution vector.
        - A dictionary with solve statistics including:
          - "solves": The number of linear solves performed.
          - "final_residual_norm": The final infinity norm of the residual.
          - "status": The status of the solve ("converged", "non-converged",
            or "stalled").

    Raises:
      ValueError: If the solution contains NaN values.
    """
    # Adjust RHS to match the quasidefinite KKT form (second block negated).
    rhs = rhs.copy()
    rhs[self.n :] *= -1.0

    if self.solver.type() == "direct":
      return self._solve_with_refinement(rhs, warm_start)
    else:
      return self._solve_indirect(rhs, warm_start)

  def _solve_with_refinement(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves using iterative refinement (for direct solvers)."""
    tolerance = self.atol + self.rtol * np.linalg.norm(rhs, np.inf)

    # Initial sol and residual.
    sol = warm_start.copy()
    residual = rhs - self.kkt @ sol
    residual_norm = np.linalg.norm(residual, np.inf)

    # Iterative refinement loop.
    status, solves = "non-converged", 0
    # max_iterative_refinement_steps >= 1 so we always do at least one solve.
    for solves in range(1, self.max_iterative_refinement_steps + 1):
      # Perform correction step using the linear system solver.
      old_residual_norm = residual_norm
      sol += self.solver.solve(residual)
      residual = rhs - self.kkt @ sol
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

  def _solve_indirect(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves using an indirect (iterative) solver."""
    sol, exitflag = self.solver.solve(rhs, warm_start)
    residual_norm = np.linalg.norm(self.kkt @ sol - rhs, np.inf)

    if np.any(np.isnan(sol)):
      raise ValueError("Linear solver returned NaNs.")

    return sol, {
        "solves": 1,
        "final_residual_norm": residual_norm,
        "status": "converged" if exitflag == 0 else "non-converged",
    }