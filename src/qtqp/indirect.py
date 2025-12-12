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

class MinResSolver(LinearSolver):
  """MinRes solver."""
  def __init__(self, kkt: sp.spmatrix) -> None:
    """Initializes the MinRes solver."""
    ...

  def update(self, kkt: sp.spmatrix) -> None:
    """Factorizes or refactorizes the KKT matrix."""
    ...

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system."""
    ...

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
    """Forms the KKT matrix diagonals and preconditioner.

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
    # 2. Restore true values for subsequent residual checks in `solve()`.
    self.kkt.data[self.kkt_nan_idxs] = true_diags

    def _check_preconditioner(self) -> None:
      """Checks the preconditioner."""
      M_dense = self.M.toarray()
      lam = scipy.linalg.eigh(self.kkt.toarray(), M_dense, eigvals_only=True)
      cond_number_K = np.linalg.cond(self.kkt.toarray())
      cond_number_MK = np.max(np.abs(lam)) / np.min(np.abs(lam))
      print(f"K: {cond_number_K:.2e}, MK: {cond_number_MK:.2e}")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
      """Solves the linear system with the given preconditioner and data"""
      ...
      