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

import enum
import logging
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular


class RefinementStrategy(enum.Enum):
  """Available iterative refinement strategies for DirectKktSolver.

  RICHARDSON:
    Classical iterative refinement, i.e. preconditioned Richardson iteration

        x_{k+1} = x_k + M^{-1} (b - A_true x_k)

    where M is the factorized regularized KKT matrix and A_true is the
    unregularized KKT matrix. Each iteration costs one preconditioner apply
    (a factor-solve) and one matvec. Stops when the inf-norm of the residual
    no longer improves or falls below atol + rtol * ||b||_inf.

  GMRES:
    Right-preconditioned restarted GMRES on A_true x = b, using the same
    factorization M as the preconditioner. Each inner Arnoldi step costs
    one factor-solve plus one matvec; each restart cycle then does one
    final factor-solve to map the Krylov-space least-squares solution back
    to a primal-space correction. Right preconditioning is the key
    difference vs. a stock left-preconditioned GMRES: the least-squares
    problem minimizes the *true* residual ||b - A_true x||, not the
    preconditioned residual ||M^{-1}(b - A_true x)||, so convergence isn't
    fooled when M distorts the residual norm (which is exactly the regime
    where pure IR stalls). Restarts every gmres_restart inner steps; the
    apply budget is capped at max_iterative_refinement_steps total.
  """

  RICHARDSON = "richardson"
  GMRES = "gmres"


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

    Subclasses that override this should call super().set_kkt(kkt) to ensure
    the base-class _kkt, _kkt_diag, and _kkt_diag_idxs are populated.
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
    res = self._kkt @ x
    res += self._kkt.T @ x
    res -= self._kkt_diag * x
    return res

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
      refinement_strategy: RefinementStrategy = RefinementStrategy.RICHARDSON,
      gmres_restart: int = 10,
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
      refinement_strategy: Which iterative-refinement scheme to drive the KKT
        solve with. See RefinementStrategy for descriptions.
      gmres_restart: Krylov dimension per restart cycle (inner Arnoldi steps
        before restart). Each cycle uses gmres_restart + 1 factor-solves
        (Arnoldi steps + one final M^{-1} apply to convert the Krylov-space
        solution back). Ignored when refinement_strategy is RICHARDSON.
    """
    if gmres_restart < 1:
      raise ValueError("gmres_restart must be >= 1.")
    self.refinement_strategy = refinement_strategy
    self.gmres_restart = gmres_restart
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
    self._true_diags[: self.n] = self._p_diags
    self._true_diags[: self.n] += mu
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

    Performs iterative refinement (Richardson or GMRES, see
    RefinementStrategy) to improve solution accuracy.

    Args:
      rhs: The right-hand side of the linear system.
      warm_start: A warm-start for the solution.

    Returns:
      A tuple (sol, stats) where stats has the keys
        - "solves": number of preconditioner applications performed.
        - "final_residual_norm": ||r||_inf after the last step.
        - "rhs_norm": ||rhs||_inf.
        - "tolerance": atol + rtol * rhs_norm.
        - "converged": bool.
        - "status": "converged", "stalled", or "non-converged".

    Raises:
      ValueError: If the solution contains NaN values.
    """
    # Adjust RHS to match the quasidefinite KKT form (second block negated).
    # Use pre-allocated buffer to avoid a copy allocation on every call.
    np.copyto(self._kkt_rhs, rhs)
    self._kkt_rhs[self.n :] *= -1.0
    rhs_norm = np.linalg.norm(self._kkt_rhs, np.inf)
    tolerance = self._atol + self._rtol * rhs_norm

    if self.refinement_strategy is RefinementStrategy.RICHARDSON:
      sol, stats = self._solve_richardson(rhs_norm, tolerance, warm_start)
    elif self.refinement_strategy is RefinementStrategy.GMRES:
      sol, stats = self._solve_gmres(rhs_norm, tolerance, warm_start)
    else:
      raise ValueError(
          f"Unknown refinement strategy: {self.refinement_strategy}"
      )

    if np.any(np.isnan(sol)):
      raise ValueError("Linear solver returned NaNs.")

    logging.debug(
        "KKT solve: strategy=%s, status=%s, solves=%d, res=%e",
        self.refinement_strategy.name,
        stats["status"],
        stats["solves"],
        stats["final_residual_norm"],
    )
    return sol, stats

  def _solve_richardson(self, rhs_norm, tolerance, warm_start):
    """Classical iterative refinement: preconditioned Richardson iteration."""
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

    return sol, {
        "solves": solves,
        "final_residual_norm": residual_norm,
        "rhs_norm": rhs_norm,
        "tolerance": tolerance,
        "converged": status == "converged",
        "status": status,
    }

  def _solve_gmres(self, rhs_norm, tolerance, warm_start):
    """Restarted right-preconditioned GMRES.

    Operator      A x  = self._solver @ x - self._diag_correction * x  (=
                          kkt_true x).
    Preconditioner M^-1 x = self._solver.solve(x)  (the direct factor).

    Each inner Arnoldi step costs one factor-solve plus one matvec; each
    cycle does one additional factor-solve at the end to convert the
    Krylov-space solution back to a primal-space correction. Total
    preconditioner applies are bounded by max_iterative_refinement_steps.
    The best iterate seen across restarts is tracked and returned, so a
    stalled refinement never returns a worse solution than the warm start.
    """
    sol = warm_start.copy()
    residual = (
        self._kkt_rhs - self._solver @ sol + self._diag_correction * sol
    )
    residual_norm = float(np.linalg.norm(residual, np.inf))

    best_sol = sol.copy()
    best_residual_norm = residual_norm
    status = "non-converged"
    solves = 0
    prev_ratio_high = False  # consecutive sluggish-step tracker

    if residual_norm < tolerance:
      status = "converged"

    while (
        status == "non-converged"
        and self.max_iterative_refinement_steps - solves >= 2
    ):
      # Each cycle uses (inner + 1) factor-solves: inner Arnoldi steps plus
      # one final M^{-1} apply. Cap inner so total cycle cost <= remaining.
      remaining = self.max_iterative_refinement_steps - solves
      max_inner = min(self.gmres_restart, remaining - 1)
      old_residual_norm = residual_norm
      used = self._gmres_cycle(sol, residual, max_inner, tolerance)
      solves += used

      # Recompute the exact inf-norm residual; the per-cycle running check
      # is in 2-norm.
      residual = (
          self._kkt_rhs - self._solver @ sol + self._diag_correction * sol
      )
      residual_norm = float(np.linalg.norm(residual, np.inf))

      if residual_norm < best_residual_norm:
        best_residual_norm = residual_norm
        np.copyto(best_sol, sol)

      if residual_norm < tolerance:
        status = "converged"
        break

      # Lucky breakdown (cycle exited early via Arnoldi breakdown) means
      # GMRES exhausted the Krylov subspace; further restarts cannot help.
      if used < max_inner + 1:
        status = "stalled"
        break

      # Stall check: bail on residual blow-up vs best, or on two consecutive
      # cycles with ratio > 0.95 (i.e. crawling at <5% improvement per cycle).
      if residual_norm > 2.0 * best_residual_norm:
        status = "stalled"
        break
      ratio_high = (
          old_residual_norm > 0.0
          and (residual_norm / old_residual_norm) > 0.95
      )
      if ratio_high and prev_ratio_high:
        logging.debug(
            "GMRES stalled after %d solves. Old res: %e, New res: %e",
            solves, old_residual_norm, residual_norm,
        )
        status = "stalled"
        break
      prev_ratio_high = ratio_high
    else:
      if status == "non-converged":
        logging.debug(
            "GMRES did not converge after %d solves."
            " Final residual: %e > tolerance: %e",
            solves, residual_norm, tolerance,
        )

    # Roll back to the best iterate if a later cycle worsened the residual.
    if best_residual_norm < residual_norm:
      np.copyto(sol, best_sol)
      residual_norm = best_residual_norm

    return sol, {
        "solves": solves,
        "final_residual_norm": residual_norm,
        "rhs_norm": rhs_norm,
        "tolerance": tolerance,
        "converged": status == "converged",
        "status": status,
    }

  def _gmres_cycle(
      self,
      sol: np.ndarray,
      residual: np.ndarray,
      max_inner: int,
      tolerance: float,
  ) -> int:
    """One restart cycle of right-preconditioned GMRES.

    Updates ``sol`` in place. Returns the total factor-solves consumed
    (= inner Arnoldi steps + 1 final M^{-1} apply, or 0 if the input
    residual is already zero). Early-exits when the running 2-norm residual
    estimate falls below ``tolerance``; the inf-norm is at most the 2-norm,
    so this is a safe (slightly conservative) trigger for the inf-norm
    convergence test the outer loop applies.
    """
    n = sol.size
    v = np.empty((max_inner + 1, n))
    h = np.zeros((max_inner + 1, max_inner))
    cs = np.zeros(max_inner)
    sn = np.zeros(max_inner)
    g = np.zeros(max_inner + 1)

    beta = float(np.linalg.norm(residual))
    if beta == 0.0:
      return 0

    v[0] = residual / beta
    g[0] = beta

    j_done = 0
    breakdown = False
    for j in range(max_inner):
      # Right preconditioning: build the Krylov subspace of A M^{-1}.
      z_j = self._solver.solve(v[j])
      w = self._solver @ z_j - self._diag_correction * z_j

      # Modified Gram-Schmidt against the existing basis.
      for i in range(j + 1):
        h[i, j] = v[i] @ w
        w -= h[i, j] * v[i]
      h_next = float(np.linalg.norm(w))
      h[j + 1, j] = h_next

      breakdown = h_next == 0.0
      if not breakdown:
        v[j + 1] = w / h_next

      # Apply previously computed Givens rotations to the new column of H.
      for i in range(j):
        t = cs[i] * h[i, j] + sn[i] * h[i + 1, j]
        h[i + 1, j] = -sn[i] * h[i, j] + cs[i] * h[i + 1, j]
        h[i, j] = t

      # New Givens rotation to zero out h[j+1, j].
      rho = float(np.hypot(h[j, j], h[j + 1, j]))
      cs[j] = h[j, j] / rho
      sn[j] = h[j + 1, j] / rho
      h[j, j] = rho
      h[j + 1, j] = 0.0

      # |g[j+1]| is the 2-norm of the least-squares residual after step j.
      g[j + 1] = -sn[j] * g[j]
      g[j] = cs[j] * g[j]

      j_done = j + 1
      if abs(g[j + 1]) < tolerance or breakdown:
        break

    # Solve in Krylov space and convert back to primal space via one M^{-1}
    # apply -- the single extra factor-solve per cycle vs the FGMRES-style
    # variant that stores Z = M^{-1} V column-by-column.
    y = solve_triangular(h[:j_done, :j_done], g[:j_done])
    sol += self._solver.solve(v[:j_done].T @ y)
    return j_done + 1

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
