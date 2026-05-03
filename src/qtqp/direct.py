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
import timeit
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular


RefinementStrategy = Literal["fixed_point", "fgmres"]


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

  @staticmethod
  def _rescale_csx_data(mat: sp.spmatrix, r: np.ndarray) -> None:
    """In-place K[i,j] *= r[i] * r[j] on a CSC/CSR sparse matrix's data array."""
    cols = np.repeat(np.arange(mat.indptr.size - 1), np.diff(mat.indptr))
    mat.data *= r[mat.indices] * r[cols]

  def rescale_off_diagonals(self, r: np.ndarray) -> None:
    """Applies symmetric scaling K[i,j] *= r[i] * r[j] to every entry of the
    stored KKT matrix.

    Used by per-iteration equilibration. Diagonal entries are typically
    overwritten by a subsequent ``update_diag`` call, so backends only need
    to ensure their off-diagonal storage reflects the rescaling. The default
    implementation operates on the upper-triangular CSC scaffold; backends
    that maintain private off-diagonal storage (e.g., the dense Schur-
    complement solver, or sparse backends that hold a separate full-symmetric
    expansion) must override this method to also rescale that storage.
    """
    self._rescale_csx_data(self._kkt, r)
    np.multiply(self._kkt_diag, r * r, out=self._kkt_diag)

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
      deadline: float | None = None,
      refinement_strategy: RefinementStrategy = "fixed_point",
      fgmres_restart: int = 10,
      legacy_stall_check: bool = False,
      equilibrate_per_iteration: int = 0,
  ):
    """Initializes the DirectKktSolver.

    Args:
      a: The constraint matrix from the QP.
      p: The quadratic cost matrix from the QP.
      z: The number of zero elements in the diagonal of the barrier term.
      min_static_regularization: Minimum static regularization to add to the
        diagonal of the KKT matrix.
      max_iterative_refinement_steps: Maximum number of iterative refinement
        steps (= preconditioner solves) to perform.
      atol: Absolute tolerance for iterative refinement.
      rtol: Relative tolerance for iterative refinement.
      solver: An instance of a direct solver class (e.g., MklPardisoSolver).
      deadline: Optional wall-clock deadline (timeit.default_timer() value). When
        set, iterative refinement exits early between steps once the current
        time exceeds it. Cannot interrupt a single in-progress backend call.
      refinement_strategy: Which refinement scheme to use. "fixed_point" is the
        classical residual-correction loop. "fgmres" wraps the same direct
        factorization as a right-preconditioner inside restarted FGMRES, which
        accelerates convergence on ill-conditioned KKT systems where the fixed
        point iteration has a contraction factor close to one.
      fgmres_restart: Inner-iteration count between FGMRES restarts. Each inner
        iteration costs one preconditioner solve plus one matvec, and stores
        two vectors of size n+m. Ignored when refinement_strategy is
        "fixed_point".
      legacy_stall_check: When True, restore the original stall criterion
        ("any non-decrease in residual norm halts refinement"). When False
        (default), use a ratio-and-window criterion that is robust to small
        non-monotonic blips at the roundoff floor and only halts when progress
        truly stops or the residual blows up. Exposed for A/B comparison.
      equilibrate_per_iteration: Number of symmetric Ruiz-style rescalings
        of the full KKT matrix to apply at every ``update`` call (in addition
        to whatever one-shot equilibration the caller already applied to A
        and P). 0 (default) disables per-iteration equilibration entirely;
        N > 0 runs N consecutive Ruiz passes. Each pass reads the current row
        inf-norms of the stored KKT, derives a per-row factor 1/sqrt(row_inf),
        applies it symmetrically as ``S K S``, and accumulates the cumulative
        scale. The IPM continues to operate in its original frame; the per-
        iteration scaling is invisible outside DirectKktSolver. Reported
        residual norms are in the equilibrated frame, so they are not directly
        comparable across iterations or against runs with this set to 0.
        Backends with private off-diagonal storage must implement
        ``rescale_off_diagonals``; the dense and CPU sparse backends do, the
        GPU backends do not.
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
    self._deadline = deadline
    if refinement_strategy not in ("fixed_point", "fgmres"):
      raise ValueError(
          f"Unknown refinement_strategy {refinement_strategy!r}; "
          "expected 'fixed_point' or 'fgmres'."
      )
    if fgmres_restart < 1:
      raise ValueError("fgmres_restart must be >= 1.")
    self.refinement_strategy = refinement_strategy
    self.fgmres_restart = fgmres_restart
    self.legacy_stall_check = legacy_stall_check
    if equilibrate_per_iteration < 0:
      raise ValueError("equilibrate_per_iteration must be >= 0.")
    self.equilibrate_per_iteration = equilibrate_per_iteration
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

    if self.equilibrate_per_iteration:
      # Cumulative symmetric scaling currently baked into self._kkt; identity
      # at construction. Each ``update`` multiplies it by a fresh per-row
      # incremental factor. Solve scales rhs/warm_start by it on entry and
      # unscales the solution on exit.
      self._cumulative_scale = np.ones(self.n + self.m, dtype=np.float64)
      self._true_diags_unscaled = np.empty(self.n + self.m, dtype=np.float64)
      self._reg_diags_unscaled = np.empty(self.n + self.m, dtype=np.float64)

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

    if self.equilibrate_per_iteration:
      # Snapshot the IPM's positive unscaled diagonals once, before applying
      # any Ruiz passes. Each pass needs the original unscaled values to
      # rebuild the scaled diagonals as ``cum_sq * unscaled``; if we instead
      # re-snapshotted from self._true_diags inside the pass, the second pass
      # would treat already-scaled diagonals as unscaled and the cumulative
      # scale would compound twice.
      np.copyto(self._true_diags_unscaled, self._true_diags)
      np.copyto(self._reg_diags_unscaled, self._reg_diags)
      # Apply N Ruiz passes to the full KKT, in place. Each pass rescales the
      # scaffold's off-diagonals via the backend, advances cumulative_scale,
      # and leaves self._true_diags / self._reg_diags scaled into the new
      # cumulative frame (still positive; the quasidefinite negation below
      # applies to the scaled values).
      for _ in range(self.equilibrate_per_iteration):
        self._equilibrate_one_pass()

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

  def _equilibrate_one_pass(self) -> None:
    """One Ruiz-style symmetric rescaling of the stored KKT.

    Reads the IPM's positive unscaled diagonals from
    ``self._true_diags_unscaled`` / ``self._reg_diags_unscaled`` (snapshotted
    by ``update`` before the pass loop) and on exit leaves
    ``self._true_diags`` / ``self._reg_diags`` rescaled into the new
    cumulative frame (still positive; caller applies the quasidefinite
    negation). Rescales the scaffold's off-diagonals via the backend and
    advances ``self._cumulative_scale``.
    """
    n = self.n
    # Write the current iteration's diagonal in the current cumulative frame
    # to the scaffold so that row inf-norms reflect this iteration's KKT.
    # Use self._reg_diags as scratch for the negated, scaled diagonal.
    cum_sq = self._cumulative_scale * self._cumulative_scale
    np.multiply(cum_sq, self._reg_diags_unscaled, out=self._reg_diags)
    self._reg_diags[n:] *= -1.0
    self._solver.update_diag(self._reg_diags)

    # Row inf-norms of the full symmetric KKT. The scaffold stores only one
    # triangle (upper for most backends, lower for Eigen), so combine row-
    # wise and column-wise maxes of |K_triangle| -- this gives the full row
    # inf-norm by symmetry regardless of which triangle is stored. Read from
    # the backend's storage rather than DirectKktSolver's scaffold so backends
    # that hold a private copy (Eigen, etc.) see the just-written diagonal.
    abs_k = abs(self._solver._kkt)  # pylint: disable=protected-access
    row_max = np.asarray(abs_k.max(axis=1).todense()).ravel()
    col_max = np.asarray(abs_k.max(axis=0).todense()).ravel()
    row_inf = np.maximum(row_max, col_max)

    # Incremental scaling. Tiny rows fall back to no rescale; clip the factor
    # to keep one Ruiz step from doing anything wild on a pathological row.
    np.maximum(row_inf, 1e-300, out=row_inf)
    r = 1.0 / np.sqrt(row_inf)
    np.clip(r, 1e-8, 1e8, out=r)

    self._solver.rescale_off_diagonals(r)
    self._cumulative_scale *= r

    # Recompute the scaled diagonals using the NEW cumulative scale; leave
    # them positive so the caller can apply the quasidefinite negation.
    cum_sq = self._cumulative_scale * self._cumulative_scale
    np.multiply(cum_sq, self._true_diags_unscaled, out=self._true_diags)
    np.multiply(cum_sq, self._reg_diags_unscaled, out=self._reg_diags)

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves the linear system with the given factorization.

    Performs iterative refinement to improve the solution accuracy. The
    refinement scheme is selected by the constructor's refinement_strategy
    argument. Both schemes track the best iterate seen and roll back to it
    if a later step degrades the residual.

    Args:
      rhs: The right-hand side of the linear system.
      warm_start: A warm-start for the solution.

    Returns:
      A tuple containing:
        - sol: The solution vector.
        - A dictionary with solve statistics including:
          - "solves": The number of preconditioner solves performed.
          - "final_residual_norm": The final infinity norm of the residual.
          - "rhs_norm": The infinity norm of the KKT right-hand side.
          - "tolerance": The absolute/relative IR stopping threshold.
          - "status": The status of the iterative refinement ("converged",
            "non-converged", "stalled", or "time_limit_exceeded").

    Raises:
      ValueError: If the solution contains NaN values.
    """
    # Adjust RHS to match the quasidefinite KKT form (second block negated).
    # Use pre-allocated buffer to avoid a copy allocation on every call.
    np.copyto(self._kkt_rhs, rhs)
    self._kkt_rhs[self.n :] *= -1.0

    # If equilibrating per iteration, the scaffold and diag_correction live
    # in the cumulative-scaled frame; the IPM still passes rhs/warm_start in
    # the original frame, so scale at the boundary and unscale on exit.
    # Substitution: K_unscaled x = rhs  <=>  K_scaled (S^-1 x) = S rhs.
    if self.equilibrate_per_iteration:
      np.multiply(self._kkt_rhs, self._cumulative_scale, out=self._kkt_rhs)
      sol = warm_start / self._cumulative_scale
    else:
      sol = warm_start.copy()

    rhs_norm = float(np.linalg.norm(self._kkt_rhs, np.inf))
    tolerance = self._atol + self._rtol * rhs_norm

    # Initial sol and residual.
    # The true residual is kkt_rhs - kkt_true @ sol. We split the matvec as:
    #   kkt_true @ sol = kkt_reg @ sol - diag_correction @ sol
    # so residual = kkt_rhs - kkt_reg @ sol + diag_correction * sol.
    # self._solver @ sol computes kkt_reg @ sol (using the factorized matrix).
    residual = self._kkt_rhs - self._solver @ sol + self._diag_correction * sol
    residual_norm = float(np.linalg.norm(residual, np.inf))

    if self.refinement_strategy == "fgmres":
      sol, status, solves, residual_norm = self._refine_fgmres(
          sol, residual, residual_norm, tolerance
      )
    else:
      sol, status, solves, residual_norm = self._refine_fixed_point(
          sol, residual, residual_norm, tolerance
      )

    if self.equilibrate_per_iteration:
      sol *= self._cumulative_scale

    if np.any(np.isnan(sol)):
      raise ValueError("Linear solver returned NaNs.")

    logging.debug(
        "KKT solve: status=%s, solves=%d, res=%e", status, solves, residual_norm
    )

    return sol, {
        "solves": solves,
        "final_residual_norm": residual_norm,
        "rhs_norm": rhs_norm,
        "tolerance": tolerance,
        "converged": status == "converged",
        "status": status,
    }

  def _refine_fixed_point(
      self,
      sol: np.ndarray,
      residual: np.ndarray,
      residual_norm: float,
      tolerance: float,
  ) -> tuple[np.ndarray, str, int, float]:
    """Classical residual-correction iterative refinement.

    Tracks the best iterate seen so far and rolls back to it on exit if a
    later step degraded the residual. Stall detection uses either the
    legacy "any non-decrease" rule or the ratio-and-window rule, depending
    on legacy_stall_check.
    """
    best_sol = sol.copy()
    best_residual_norm = residual_norm
    status = "non-converged"
    solves = 0
    prev_ratio_high = False  # only used by the ratio+window stall check

    # max_iterative_refinement_steps >= 1 so we always do at least one solve.
    for solves in range(1, self.max_iterative_refinement_steps + 1):
      old_residual_norm = residual_norm
      sol += self._solver.solve(residual)
      residual = self._kkt_rhs - self._solver @ sol + self._diag_correction * sol
      residual_norm = float(np.linalg.norm(residual, np.inf))

      if residual_norm < best_residual_norm:
        best_residual_norm = residual_norm
        np.copyto(best_sol, sol)

      if residual_norm < tolerance:
        status = "converged"
        break

      if self._is_stalled(
          residual_norm, old_residual_norm, best_residual_norm, prev_ratio_high
      ):
        logging.debug(
            "Iterative refinement stalled at step %d. Old res: %e, New res: %e",
            solves,
            old_residual_norm,
            residual_norm,
        )
        status = "stalled"
        break

      if not self.legacy_stall_check and old_residual_norm > 0.0:
        prev_ratio_high = (residual_norm / old_residual_norm) > 0.95

      if self._deadline is not None and timeit.default_timer() > self._deadline:
        status = "time_limit_exceeded"
        break
    else:
      logging.debug(
          "Iterative refinement did not converge after %d solves."
          " Final residual: %e > tolerance: %e",
          solves,
          residual_norm,
          tolerance,
      )

    if best_residual_norm < residual_norm:
      np.copyto(sol, best_sol)
      residual_norm = best_residual_norm

    return sol, status, solves, residual_norm

  def _is_stalled(
      self,
      residual_norm: float,
      old_residual_norm: float,
      best_residual_norm: float,
      prev_ratio_high: bool,
  ) -> bool:
    """Returns True when iterative refinement has stopped making progress.

    Legacy mode bails on any non-decrease. The default mode tolerates a
    single sluggish step (ratio in [0.95, 1)) and small non-monotonic blips
    near the roundoff floor, but bails immediately if the residual blows
    up above 2x the best seen, or if two consecutive steps both ran with
    a ratio above 0.95.
    """
    if self.legacy_stall_check:
      return residual_norm >= old_residual_norm
    if residual_norm > 2.0 * best_residual_norm:
      return True
    if old_residual_norm == 0.0:
      return False
    ratio_high = (residual_norm / old_residual_norm) > 0.95
    return ratio_high and prev_ratio_high

  def _refine_fgmres(
      self,
      sol: np.ndarray,
      residual: np.ndarray,
      residual_norm: float,
      tolerance: float,
  ) -> tuple[np.ndarray, str, int, float]:
    """Restarted right-preconditioned FGMRES.

    Operator A x = self._solver @ x - self._diag_correction * x  (= kkt_true x).
    Right preconditioner M^-1 x = self._solver.solve(x)  (the direct factor).

    Each inner iteration costs one preconditioner solve plus one matvec, the
    same per-step cost as the fixed_point loop. FGMRES converges in fewer
    outer iterations when the contraction factor of pure IR is close to one,
    e.g. when min_static_regularization is large relative to the true KKT
    diagonal.
    """
    best_sol = sol.copy()
    best_residual_norm = residual_norm
    status = "non-converged"
    solves = 0

    while solves < self.max_iterative_refinement_steps:
      cycle_budget = min(
          self.fgmres_restart, self.max_iterative_refinement_steps - solves
      )
      old_residual_norm = residual_norm
      used = self._fgmres_cycle(sol, residual, cycle_budget, tolerance)
      solves += used

      # Recompute the exact inf-norm residual; the GMRES estimate is in 2-norm.
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

      # Lucky breakdown inside the cycle (used < cycle_budget) means GMRES
      # exhausted the Krylov subspace; further restarts cannot help.
      if used < cycle_budget:
        status = "stalled"
        break

      if self._is_stalled(
          residual_norm, old_residual_norm, best_residual_norm, False
      ):
        logging.debug(
            "FGMRES stalled after %d solves. Old res: %e, New res: %e",
            solves,
            old_residual_norm,
            residual_norm,
        )
        status = "stalled"
        break

      if self._deadline is not None and timeit.default_timer() > self._deadline:
        status = "time_limit_exceeded"
        break
    else:
      logging.debug(
          "FGMRES did not converge after %d solves."
          " Final residual: %e > tolerance: %e",
          solves,
          residual_norm,
          tolerance,
      )

    if best_residual_norm < residual_norm:
      np.copyto(sol, best_sol)
      residual_norm = best_residual_norm

    return sol, status, solves, residual_norm

  def _fgmres_cycle(
      self,
      sol: np.ndarray,
      residual: np.ndarray,
      max_inner: int,
      tolerance: float,
  ) -> int:
    """One restart cycle of right-preconditioned FGMRES.

    Updates ``sol`` in place. Returns the number of inner iterations taken
    (= preconditioner solves consumed). Early-exits when the running 2-norm
    residual estimate falls below ``tolerance``; since the inf-norm is at
    most the 2-norm, this is a safe (slightly conservative) trigger for the
    inf-norm convergence test the outer loop applies.
    """
    n = sol.size
    v = np.empty((max_inner + 1, n))
    z = np.empty((max_inner, n))
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
    for j in range(max_inner):
      z[j] = self._solver.solve(v[j])
      w = self._solver @ z[j] - self._diag_correction * z[j]

      # Modified Gram-Schmidt orthogonalization against the existing basis.
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

      # Update the projected RHS; |g[j+1]| equals the 2-norm of the residual
      # of the least-squares problem after step j.
      g[j + 1] = -sn[j] * g[j]
      g[j] = cs[j] * g[j]

      j_done = j + 1
      if abs(g[j + 1]) < tolerance or breakdown:
        break

    y = solve_triangular(h[:j_done, :j_done], g[:j_done])
    sol += z[:j_done].T @ y
    return j_done

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
