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
"""Direct KKT linear system solver with iterative refinement.

Solver backends live in separate modules:
  solvers_sparse  — ScipySolver, MklPardisoSolver, QdldlSolver, etc.
  solvers_dense   — ScipyDenseSolver (Gram/Cholesky CPU)
  solvers_gpu     — CuDssSolver, CupyDenseSolver (CUDA)

All backend classes are re-exported here so that ``from . import direct``
followed by ``direct.ScipySolver`` continues to work.
"""

import logging
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp


def diag_data_indices(mat: sp.spmatrix) -> np.ndarray:
  """Return the data-array index of each diagonal entry in a CSC/CSR matrix.

  Works identically for CSC (col k, find row k) and CSR (row k, find col k).
  """
  mat.sort_indices()
  dim = mat.shape[0]
  idxs = np.empty(dim, dtype=np.intp)
  for k in range(dim):
    start = mat.indptr[k]
    idxs[k] = start + np.searchsorted(
        mat.indices[start:mat.indptr[k + 1]], k
    )
  return idxs


class LinearSolver:
  """Base class for KKT linear system solvers.

  To add a new solver, subclass this and implement factorize, solve, and format.
  set_kkt, __matmul__, and free are provided; override __matmul__ for a more
  efficient matvec (e.g. a pre-allocated dense array).
  """

  def set_dims(self, n: int, m: int, z: int) -> None:
    """Stores problem dimensions; called once before set_kkt.

    Dense solvers override this to pre-allocate buffers sized by n and m.
    Sparse solvers ignore it (the default is a no-op).
    """
    pass

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    """Stores the upper-triangular KKT matrix; called once at init time.

    Subclasses that rely on the base sparse storage or the default
    update_diag/__matmul__ helpers should call super().set_kkt(kkt) so
    _kkt, _kkt_diag, and _kkt_diag_idxs are populated.
    """
    self._kkt = kkt
    self._kkt_diag = kkt.diagonal()
    self._kkt_diag_idxs = diag_data_indices(kkt)

  def update_diag(self, diag: np.ndarray) -> None:
    """Updates the KKT diagonal in place; called each iteration before factorize.

    The default writes the diagonal into the stored sparse matrix and updates
    _kkt_diag.  Backends with private copies (dense, GPU, LU-from-upper)
    override this to update their own storage.
    """
    self._kkt.data[self._kkt_diag_idxs] = diag
    np.copyto(self._kkt_diag, diag)

  def factorize(self) -> None:
    """Factorizes the stored KKT matrix (with regularized diagonals).

    DirectKktSolver applies a diagonal correction during iterative refinement
    to account for the difference between regularized and true diagonals,
    so implementations only need to factorize kkt as given.
    """
    raise NotImplementedError

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    """Solves the factorized system for the given right-hand side."""
    raise NotImplementedError

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._kkt @ x + self._kkt.T @ x - self._kkt_diag * x

  def format(self) -> str:
    """Preferred sparse format for the KKT scaffold ('csc' or 'csr')."""
    raise NotImplementedError

  def free(self) -> None:
    pass


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
    # Store only the diagonal of P. The off-diagonal elements of P are constant
    # and are baked into the KKT scaffold permanently (see kkt construction
    # below). Only the diagonal changes each iteration (to add mu * I), so we
    # cache it here for fast in-place updates without touching the full matrix.
    self._p_diags = p.diagonal()
    self.min_static_regularization = min_static_regularization
    self.max_iterative_refinement_steps = max_iterative_refinement_steps
    self._atol = atol
    self._rtol = rtol
    self._solver = solver
    self._solver.set_dims(n=self.n, m=self.m, z=self.z)

    # Build the upper-triangular KKT scaffold once. Placeholder ones on the
    # diagonal ensure those positions exist in the sparse structure; they are
    # overwritten each iteration with the actual values (which depend on mu,
    # s, y).
    n_ones = sp.eye(self.n, format="csc", dtype=np.float64)
    m_ones = sp.eye(self.m, format="csc", dtype=np.float64)
    p_triu = sp.triu(p, format="csc")

    self._kkt = sp.bmat(
        [[p_triu + n_ones, a.T], [None, m_ones]],
        format=self._solver.format(),
        dtype=np.float64,
    )
    self._solver.set_kkt(self._kkt)

    # Pre-allocate reusable buffers to avoid per-call allocations.
    self._true_diags = np.empty(self.n + self.m, dtype=np.float64)
    self._reg_diags = np.empty(self.n + self.m, dtype=np.float64)
    self._kkt_rhs = np.empty(self.n + self.m, dtype=np.float64)    # RHS with cone block negated
    self._diag_correction = np.zeros(self.n + self.m, dtype=np.float64)  # reg - true

  def update(self, mu: float, s: np.ndarray, y: np.ndarray):
    """Forms the KKT matrix diagonals and factorizes it.

    Computes regularized diagonals (clamped to min_static_regularization) for
    numerical stability, and stores the difference from the true diagonals as
    diag_correction. This correction is applied during iterative refinement so
    that the solver converges to the solution of the true (unregularized) system.

    Args:
      mu: The barrier parameter.
      s: The slack variables.
      y: The dual variables for the conic constraints.
    """
    # Fill true diagonals: [p_diags + mu, h + mu] where h = [[0]*z; s/y].
    # KKT form: [P+mu*I, A'; A, -(D+mu*I)]
    self._true_diags[: self.n] = self._p_diags + mu
    self._true_diags[self.n : self.n + self.z] = mu
    self._true_diags[self.n + self.z :] = s[self.z :] / y[self.z :] + mu

    # "Regularized" diagonals for stable factorization.
    np.maximum(self._true_diags, self.min_static_regularization, out=self._reg_diags)
    # Flip the sign of the cone variables.
    self._true_diags[self.n :] *= -1.0
    self._reg_diags[self.n :] *= -1.0

    # Inject regularized diagonals and factorize. The solver sees one consistent
    # matrix throughout. During iterative refinement, DirectKktSolver adds
    # diag_correction * sol to the residual to account for the difference
    # between the regularized matrix (used for factorization and matvec) and
    # the true matrix (whose solution we seek), converging to the exact answer.
    self._solver.update_diag(self._reg_diags)
    self._solver.factorize()
    np.subtract(self._reg_diags, self._true_diags, out=self._diag_correction)

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
    np.copyto(self._kkt_rhs, rhs)
    self._kkt_rhs[self.n :] *= -1.0
    tolerance = self._atol + self._rtol * np.linalg.norm(self._kkt_rhs, np.inf)

    # Initial sol and residual.
    # The true residual is kkt_rhs - kkt_true @ sol. We split the matvec as:
    #   kkt_true @ sol = kkt_reg @ sol - diag_correction @ sol
    # so residual = kkt_rhs - kkt_reg @ sol + diag_correction * sol.
    # self._solver @ sol computes kkt_reg @ sol (using the factorized matrix).
    sol = warm_start.copy()
    residual = self._kkt_rhs - self._solver @ sol + self._diag_correction * sol
    residual_norm = np.linalg.norm(residual, np.inf)

    # Iterative refinement loop.
    status, solves = "non-converged", 0
    # max_iterative_refinement_steps >= 1 so we always do at least one solve.
    for solves in range(1, self.max_iterative_refinement_steps + 1):
      # Perform correction step using the linear system solver.
      old_residual_norm = residual_norm
      sol += self._solver.solve(residual)
      residual = self._kkt_rhs - self._solver @ sol + self._diag_correction * sol
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
    self._solver.free()


# Re-export all backend solver classes so that ``direct.ScipySolver`` etc.
# continue to work without changing __init__.py or test imports.
from .solvers_sparse import AccelerateSolver
from .solvers_sparse import CholModSolver
from .solvers_sparse import EigenSolver
from .solvers_sparse import MklPardisoSolver
from .solvers_sparse import MumpsSolver
from .solvers_sparse import QdldlSolver
from .solvers_sparse import ScipySolver
from .solvers_sparse import UmfpackSolver
from .solvers_dense import ScipyDenseSolver
from .solvers_gpu import CuDssSolver
from .solvers_gpu import CupyDenseSolver
