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

"""Interior point method for solving QPs with Multiple Centrality Corrections."""

import dataclasses
import enum
import logging
import timeit
from typing import Any, Dict, List

import numpy as np
import scipy.sparse as sp

from . import direct

__version__ = "0.0.4"
_HEADER = """| iter |      pcost |      dcost |     pres |     dres |      gap |   infeas |       mu |  q, p, c |     time |"""
_SEPARA = """|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|"""
_norm = np.linalg.norm
_EPS = 1e-15  # Standard epsilon for numerical safety


class LinearSolver(enum.Enum):
  """Available linear solvers."""

  SCIPY = direct.ScipySolver
  PARDISO = direct.MklPardisoSolver
  QDLDL = direct.QdldlSolver
  CHOLMOD = direct.CholModSolver
  CUDSS = direct.CuDssSolver
  EIGEN = direct.EigenSolver
  MUMPS = direct.MumpsSolver


class SolutionStatus(enum.Enum):
  """Possible statuses of the QP solution."""

  SOLVED = "solved"
  INFEASIBLE = "infeasible"
  UNBOUNDED = "unbounded"
  FAILED = "failed"
  UNFINISHED = "unfinished"


@dataclasses.dataclass(frozen=True)
class Solution:
  """Contains the solution to the QP problem.

  Attributes:
    x: The primal solution or certificate of dual infeasibility.
    y: The dual solution or certificate of primal infeasibility.
    s: The slack solution or certificate of dual infeasibility.
    stats: A list of statistics dictionaries from each iteration.
    status: SolutionStatus enum indicating the status.
  """

  x: np.ndarray
  y: np.ndarray
  s: np.ndarray
  stats: List[Dict[str, Any]]
  status: SolutionStatus


class QTQP:
  """Primal-dual interior point method for solving quadratic programs (QPs).

  Solves primal QP problem:
    min. (1/2) x.T @ p @ x + c.T @ x
    s.t. a @ x + s = b
         s[:z] == 0
         s[z:] >= 0

  With dual:
    max. -(1/2) x.T @ p @ x - b.T @ y
    s.t. p @ x + a.T @ y = -c
         y[z:] >= 0
  """

  def __init__(
      self,
      *,
      a: sp.csc_matrix,
      b: np.ndarray,
      c: np.ndarray,
      z: int,
      p: sp.csc_matrix | None = None,
  ):
    """Initialize the QP solver.

    Args:
      a: Constraint matrix in CSC format (m x n).
      b: Right-hand side vector (m,).
      c: Cost vector (n,).
      z: The number of equality constraints (zero-cone size).
      p: QP matrix in CSC format (n x n). Assumed zero if None.
    """
    self.m, self.n = a.shape
    self.z = z

    # Input validation
    if not sp.isspmatrix_csc(a):
      raise TypeError("Constraint matrix 'a' must be in CSC format.")
    self.a = a

    self.b = np.array(b, dtype=np.float64)
    if self.b.shape != (self.m,):
      raise ValueError(f"b must have shape ({self.m},), got {self.b.shape}")

    self.c = np.array(c, dtype=np.float64)
    if self.c.shape != (self.n,):
      raise ValueError(f"c must have shape ({self.n},), got {self.c.shape}")

    if self.z >= self.m:
      raise ValueError(
          f"Number of equality constraints z={self.z} must be strictly less "
          f"than number of rows in A={self.m}"
      )

    if p is None:
      self.p = sp.csc_matrix((self.n, self.n))
    else:
      if not sp.isspmatrix_csc(p):
        raise TypeError("QP matrix 'p' must be in CSC format.")
      if p.shape != (self.n, self.n):
        raise ValueError(
            f"p must have shape ({self.n}, {self.n}, got {p.shape})"
        )
      self.p = p

  def solve(
      self,
      *,
      atol: float = 1e-7,
      rtol: float = 1e-8,
      atol_infeas: float = 1e-8,
      rtol_infeas: float = 1e-9,
      max_iter: int = 100,
      step_size_scale: float = 0.99,
      max_mcc_iterations: int = 3,
      mcc_improvement_threshold: float = 1.05,
      min_static_regularization: float = 1e-8,
      max_iterative_refinement_steps: int = 50,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.SCIPY,
      verbose: bool = True,
      equilibrate: bool = True,
  ) -> Solution:
    """Solves the QP using a primal-dual interior-point method.

    Args:
      atol (float): Absolute tolerance for convergence criteria.
      rtol (float): Relative tolerance for convergence criteria, scaled by
        problem data norms.
      atol_infeas (float): Absolute tolerance for detecting primal or dual
        infeasibility.
      rtol_infeas (float): Relative tolerance for detecting primal or dual
        infeasibility.
      max_iter (int): Maximum number of iterations before stopping.
      step_size_scale (float): A factor in (0, 1) to scale the step size,
        ensuring iterates remain strictly interior.
      max_mcc_iterations (int): Maximum number of extra Multiple Centrality
        Correction (MCC) steps to take per iteration.
      mcc_improvement_threshold (float): Minimum ratio of improvement in step
        size required to accept an additional MCC step (e.g., 1.05 means 5%
        improvement needed).
      min_static_regularization (float): Minimum regularization value used in
        the KKT matrix diagonal for numerical stability.
      max_iterative_refinement_steps (int): Maximum iterative refinement steps
        for the linear solves (includes the initial solve, so must be >= 1).
      linear_solver_atol (float): Absolute tolerance for the iterative
        refinement process within the linear solver.
      linear_solver_rtol (float): Relative tolerance for the iterative
        refinement process within the linear solver.
      linear_solver (LinearSolver): The linear solver to use when solving the
        KKT system.
      verbose (bool): If True, prints a summary of each iteration.
      equilibrate (bool): If True, equilibrate the data for better numerical
        stability.

    Returns:
      A Solution object containing the solution and solve stats.
    """
    assert atol >= 0
    assert rtol >= 0
    assert atol_infeas >= 0
    assert rtol_infeas >= 0
    assert max_iter > 0
    assert 0 < step_size_scale < 1
    assert max_mcc_iterations >= 0
    assert mcc_improvement_threshold >= 1.0
    assert min_static_regularization >= 0
    assert max_iterative_refinement_steps >= 1
    assert linear_solver_atol >= 0
    assert linear_solver_rtol >= 0

    self.linf_neighborhood_scale = 0.1
    self.backtrack_factor = 0.95
    self.step_size_scale = step_size_scale

    self.start_time = timeit.default_timer()
    self.atol, self.rtol = atol, rtol
    self.atol_infeas, self.rtol_infeas = atol_infeas, rtol_infeas
    self.verbose = verbose
    self.equilibrate = equilibrate
    if verbose:
      print(
          f"| QTQP v{__version__}:"
          f" m={self.m}, n={self.n}, z={self.z}, nnz(A)={self.a.nnz},"
          f" nnz(P)={self.p.nnz}, linear_solver={linear_solver.name}"
      )

    # --- Initialization ---
    x = np.zeros(self.n)
    y = np.zeros(self.m)
    s = np.zeros(self.m)

    # Initialize inequality duals and slacksto 1.0 for interiority
    y[self.z :] = 1.0
    s[self.z :] = 1.0

    # tau is homogeneous embedding variable. Kept as 1-element array for
    # consistent vector operations (e.g., @ operator).
    tau = np.array([1.0])

    # Check for valid initial interior point if supplied
    if np.any(y[self.z :] < 0) or np.any(s[self.z :] < 0):
      raise ValueError("Initial y or s has negative values in the pos cone.")
    if np.any(s[: self.z] != 0):
      raise ValueError("Initial s has nonzero values in the zero cone.")

    if self.equilibrate:
      a, p, b, c, self.d, self.e = self._equilibrate()
      x, y, s = self._equilibrate_iterates(x, y, s)
    else:
      a, p, b, c, self.d, self.e = self.a, self.p, self.b, self.c, None, None

    self.q = np.concatenate([c, b])

    self._linear_solver = direct.DirectKktSolver(
        a=a,
        p=p,
        z=self.z,
        min_static_regularization=min_static_regularization,
        max_iterative_refinement_steps=max_iterative_refinement_steps,
        atol=linear_solver_atol,
        rtol=linear_solver_rtol,
        solver=linear_solver.value(),
    )

    stats = []
    self.kinv_q = np.zeros_like(self.q)  # Initialize for warm-start.
    status = SolutionStatus.UNFINISHED
    self._log_header()

    # --- Main Iteration Loop ---
    for self.it in range(max_iter):
      stats_i = {}

      x, y, tau, s = self._normalize(x, y, tau, s)

      # Calculate current complementary slackness error (mu)
      mu = (y @ s) / (self.m - self.z)
      self._linear_solver.update(mu=mu, s=s, y=y)

      # --- Step 1: Solve for KKT @ q ---
      # This is reused for both predictor and corrector parts of the step.
      self.kinv_q, q_lin_sys_stats = self._linear_solver.solve(
          rhs=self.q, warm_start=self.kinv_q
      )
      stats_i.update(q_lin_sys_stats=q_lin_sys_stats)

      # --- Step 2: Predictor (Affine) Step ---
      # Solve KKT with mu_target = 0 to find pure Newton direction.
      x_p, y_p, tau_p, predictor_lin_sys_stats = self._newton_step(
          p=p,
          mu=mu,
          mu_target=0.0,
          r_anchor=np.concatenate([x, y]),
          tau_anchor=tau,
          y=y,
          s=s,
          tau=tau,
          correction=None,
      )
      stats_i.update(predictor_lin_sys_stats=predictor_lin_sys_stats)

      d_x_p, d_y_p, d_tau_p = x_p - x, y_p - y, tau_p - tau
      d_s_p = np.zeros(self.m)
      d_s_p[self.z :] = -y_p[self.z :] * s[self.z :] / y[self.z :]

      # Compute predictor step size and resulting centering parameter (sigma)
      alpha_p, backtracks_p = self._compute_step_size(x, y, tau, s, d_x_p, d_y_p, d_tau_p, d_s_p)
      sigma = self._compute_sigma(
          mu, x, y, tau, s, alpha_p, d_x_p, d_y_p, d_tau_p, d_s_p
      )
      print(f"backtracks_p: {backtracks_p}")

      # --- Step 3: Corrector Step with Multiple Centrality Corrections ---
      # Start with Mehrotra correction term.
      correction = -d_s_p[self.z :] * d_y_p[self.z :] / y[self.z :]

      best_alpha = -1.0
      best_step = None
      corrector_stats_agg = {"solves": 0}

      # Loop for MCC: Standard Mehrotra is iteration 0.
      for mcc_i in range(1 + max_mcc_iterations):
        x_c, y_c, tau_c, corr_stats = self._newton_step(
            p=p,
            mu=mu,
            mu_target=sigma * mu,
            r_anchor=np.concatenate([x_p, y_p]),
            tau_anchor=tau_p,
            y=y,
            s=s,
            tau=tau,
            correction=correction,
        )
        corrector_stats_agg["solves"] += corr_stats.get("solves", 0)

        # Calculate full proposed direction for this correction level.
        d_x_try = x_c - x
        d_y_try = y_c - y
        d_tau_try = tau_c - tau
        d_s_try = np.zeros(self.m)
        # Derive d_s consistent with the linearized complementarity equation
        # used in this step (sigma*mu + y*correction).
        d_s_try[self.z :] = (
            sigma * mu / y[self.z :]
            + correction
            - y_c[self.z :] * s[self.z :] / y[self.z :]
        )

        alpha_try, backtracks_try = self._compute_step_size(x, y, tau, s, d_x_try, d_y_try, d_tau_try, d_s_try)
        print(f"backtracks_try: {backtracks_try}")

        # MCC Decision: Accept if it's the first one (standard Mehrotra) OR
        # if it significantly improves the step size over the best found so far.
        if mcc_i == 0 or alpha_try >= best_alpha * mcc_improvement_threshold:
          best_alpha = alpha_try
          best_step = (d_x_try, d_y_try, d_tau_try, d_s_try)

          # Prepare correction for the NEXT potential iteration by using the
          # latest, better Second-Order approximation.
          if mcc_i < max_mcc_iterations:
            correction = -d_s_try[self.z :] * d_y_try[self.z :] / y[self.z :]
        else:
          # If no significant improvement, stop wasting cheap backsolves.
          break

      stats_i.update(corrector_lin_sys_stats=corrector_stats_agg)
      d_x, d_y, d_tau, d_s = best_step
      alpha = best_alpha

      # --- Step 4: Update Iterates ---
      x += self.step_size_scale * alpha * d_x
      y += self.step_size_scale * alpha * d_y
      tau += self.step_size_scale * alpha * d_tau
      s += self.step_size_scale * alpha * d_s

      # Ensure variables stay strictly in the cone to prevent numerical issues.
      y[self.z :] = np.maximum(y[self.z :], 1e-30)
      s[self.z :] = np.maximum(s[self.z :], 1e-30)
      tau = np.maximum(tau, 1e-30)

      # --- Termination Check---
      status = self._check_termination(x, y, tau, s, alpha, mu, sigma, stats_i)
      self._log_iteration(stats_i)
      stats.append(stats_i)
      if status != SolutionStatus.UNFINISHED:
        break

    # We have terminated for one reason or another.
    self._linear_solver.free()
    if self.equilibrate:
      x, y, s = self._unequilibrate_iterates(x, y, s)
    match status:
      case SolutionStatus.SOLVED:
        self._log_footer("Solved")
        return Solution(x / tau, y / tau, s / tau, stats, status)
      case SolutionStatus.INFEASIBLE:
        self._log_footer("Primal infeasible / dual unbounded")
        x.fill(np.nan)
        s.fill(np.nan)
        return Solution(x, y / abs(self.b @ y), s, stats, status)
      case SolutionStatus.UNBOUNDED:
        self._log_footer("Dual infeasible / primal unbounded")
        y.fill(np.nan)
        abs_ctx = abs(self.c @ x)
        return Solution(x / abs_ctx, y, s / abs_ctx, stats, status)
      case SolutionStatus.UNFINISHED:
        self._log_footer(f"Failed to converge in {max_iter} iterations")
        return Solution(x / tau, y / tau, s / tau, stats, SolutionStatus.FAILED)
      case _:
        raise ValueError(f"Unknown convergence status: {status}")

  def _equilibrate(self, num_iters=10, min_scale=1e-3, max_scale=1e3):
    """Ruiz equilibration to improve numerical conditioning."""
    # Initialize the equilibrated matrices.
    a, p, b, c = (self.a, self.p, self.b, self.c)
    # Initialize the equilibration matrices.
    d, e = (np.ones(self.m), np.ones(self.n))

    for i in range(num_iters):
      # Row norms (infinity norm)
      d_i = sp.linalg.norm(a, np.inf, axis=1)
      d_i = np.where(d_i == 0.0, 1.0, d_i)  # If a row is zero, set d_i 1.0.
      d_i = 1.0 / np.sqrt(d_i)
      d_i = np.clip(d_i, min_scale, max_scale)

      # Column norms (max of A col norms and P col norms)
      e_i_a = sp.linalg.norm(a, np.inf, axis=0)
      e_i_p = sp.linalg.norm(p, np.inf, axis=0)
      e_i = np.maximum(e_i_a, e_i_p)
      e_i = np.where(e_i == 0.0, 1.0, e_i)  # If a col is zero, set e_i 1.0.
      e_i = 1.0 / np.sqrt(e_i)
      e_i = np.clip(e_i, min_scale, max_scale)

      # Apply scaling
      d_mat = sp.diags(d_i)
      e_mat = sp.diags(e_i)
      a = d_mat @ a @ e_mat
      p = e_mat @ p @ e_mat

      # Accumulate scaling factors
      d *= d_i
      e *= e_i
      logging.debug(
          "Equilibration: iter %d: d_i err: %s, e_i err: %s",
          i,
          _norm(d_i - 1, np.inf),
          _norm(e_i - 1, np.inf),
      )

    return a, p, b * d, c * e, d, e

  def _unequilibrate_iterates(self, x, y, s):
    return (self.e * x, self.d * y, s / self.d)

  def _equilibrate_iterates(self, x, y, s):
    return (x / self.e, y / self.d, s * self.d)

  def _max_step_size(self, y: np.ndarray, delta_y: np.ndarray) -> float:
    """Finds maximum step `alpha` in [0, 1] s.t. y + alpha * delta_y >= 0."""
    # Only consider directions that reduce the variable (delta_y < 0)
    # Use a small tolerance to ignore numerical noise
    idx = delta_y < -_EPS
    if not np.any(idx):
      return 1.0
    # The step to hit zero for these variables is -y / delta_y
    min_step = np.min(-y[idx] / delta_y[idx])
    return min(1.0, min_step)

  def _compute_sigma(
      self, mu_curr, x, y, tau, s, alpha, d_x, d_y, d_tau, d_s
  ) -> float:
    """Computes the centering parameter sigma using Mehrotra's heuristic."""
    # Projected complementarity after affine step
    x_aff = x + alpha * d_x
    y_aff = y + alpha * d_y
    tau_aff = tau + alpha * d_tau
    s_aff = s + alpha * d_s
    _, y_aff, _, s_aff = self._normalize(x_aff, y_aff, tau_aff, s_aff)
    mu_aff = (y_aff @ s_aff) / (self.m - self.z)

    # If affine step reduces mu significantly, use small sigma (aggressive)
    sigma = (mu_aff / max(_EPS, mu_curr)) ** 3
    return np.clip(sigma, 0.0, 1.0)

  def _newton_step(
      self, *, p, mu, mu_target, r_anchor, tau_anchor, y, s, tau, correction
  ):
    """Computes a search direction by solving the augmented KKT system."""

    # Prepare RHS for the linear system.
    r = (mu - mu_target) * r_anchor
    r_cone = mu_target / y[self.z :] + s[self.z :]
    if correction is not None:
      r_cone += correction
    r[self.n + self.z :] += r_cone

    kinv_r, lin_sys_stats = self._linear_solver.solve(
        rhs=r,
        warm_start=r_anchor,
    )

    # Solve the 1D quadratic equation for the homogeneous tau component
    try:
      r_tau = (mu - mu_target) * tau_anchor
      tau_plus = self._solve_for_tau(p, kinv_r, mu, mu_target, r_tau)
    except ValueError as e:
      # Fallback if quadratic solve fails numerically (rare but possible)
      logging.warning("Tau solve failed, using previous tau. Error: %s", e)
      tau_plus = tau

    # Reconstruct full (x, y) step from KKT solution components
    xy_plus = kinv_r - self.kinv_q * tau_plus
    x_plus, y_plus = xy_plus[: self.n], xy_plus[self.n :]
    return x_plus, y_plus, tau_plus, lin_sys_stats

  def _solve_for_tau(self, p, kinv_r, mu, mu_target, r_tau) -> np.ndarray:
    """Solves the quadratic equation for the tau step in homogeneous embedding."""
    # Solve a quadratic equation for tau: t_a * tau^2 + t_b * tau + t_c = 0
    n = self.n
    q, kinv_q = self.q, self.kinv_q

    v = p @ np.stack([kinv_r[:n], kinv_q[:n]], axis=1)
    p_kinv_r, p_kinv_q = v[:, 0], v[:, 1]

    t_a = mu + kinv_q @ q - kinv_q[:n] @ p_kinv_q
    t_b = -r_tau[0] - kinv_r @ q + kinv_r[:n] @ p_kinv_q + kinv_q[:n] @ p_kinv_r
    t_c = -kinv_r[:n] @ p_kinv_r - mu_target
    logging.debug("t_a=%s, t_b=%s, t_c=%s", t_a, t_b, t_c)

    # Standard quadratic formula for the positive root
    discriminant = t_b**2 - 4 * t_a * t_c
    if discriminant < -1e-9:
      raise ValueError(f"Negative discriminant: {discriminant}")

    tau_sol = (-t_b + np.sqrt(max(0.0, discriminant))) / (2 * t_a)

    if tau_sol < -1e-10:
      raise ValueError(f"Negative tau solution found: {tau_sol}")

    return np.array([max(0.0, tau_sol)])

  def _normalize(self, x, y, tau, s):
    """Normalizes iterates to have norm as determined by central path."""
    xyt_norm = np.sqrt(x @ x + y @ y + tau @ tau)
    scale = np.sqrt(self.m - self.z + 1) / max(_EPS, xyt_norm)
    return x * scale, y * scale, tau * scale, s * scale

  def _compute_step_size(self, x, y, tau, s, d_x, d_y, d_tau, d_s) -> float:
    """Computes the maximum standard primal-dual step size."""
    alpha_s = self._max_step_size(s[self.z :], d_s[self.z :])
    alpha_y = self._max_step_size(y[self.z :], d_y[self.z :])
    alpha = min(alpha_s, alpha_y)

    # Stay within the neighborhood of the central path.
    def in_neighborhood(alpha):
      x_new = x + self.step_size_scale * alpha * d_x
      y_new = y + self.step_size_scale * alpha * d_y
      tau_new = tau + self.step_size_scale * alpha * d_tau
      s_new = s + self.step_size_scale * alpha * d_s
      x_new, y_new, tau_new, s_new = self._normalize(x_new, y_new, tau_new, s_new)

      vec_norm_sq = x_new @ x_new + y_new @ y_new + tau_new @ tau_new
      mu_new = (y_new @ s_new) / vec_norm_sq

      return np.all(
        y_new[self.z :] * s_new[self.z :]
        > self.linf_neighborhood_scale * mu_new
      )

    backtracks = 0
    # Backtrack until neighborhood condition is satisfied or alpha is small
    while not in_neighborhood(alpha) and alpha > 1e-6:
      alpha *= self.backtrack_factor
      backtracks += 1

    return alpha, backtracks


  def _check_termination(self, x, y, tau_arr, s, alpha, mu, sigma, stats_i):
    """Check termination criteria and compute iteration statistics."""
    sy = s * y
    s_over_y = s / np.maximum(_EPS, y)
    if self.equilibrate:
      x, y, s = self._unequilibrate_iterates(x, y, s)

    inv_tau = 1.0 / max(tau_arr[0], _EPS)

    # Precompute commonly used matrix-vector products
    ax = self.a @ x
    aty = self.a.T @ y
    px = self.p @ x
    xpx = x @ px
    ctx = self.c @ x
    bty = self.b @ y

    # Costs
    pcost = (ctx + 0.5 * xpx * inv_tau) * inv_tau
    dcost = (-bty - 0.5 * xpx * inv_tau) * inv_tau

    # Residuals
    pres = _norm((ax + s) * inv_tau - self.b, np.inf)
    dres = _norm((px + aty) * inv_tau + self.c, np.inf)
    gap = np.abs((ctx + bty + xpx * inv_tau) * inv_tau)
    complementarity = np.abs((y @ s) * inv_tau * inv_tau)

    # Infeasibility certificates
    dinfeas_a = _norm((ax + s), np.inf) / (abs(ctx) + _EPS)
    dinfeas_p = _norm(px, np.inf) / (abs(ctx) + _EPS)
    dinfeas = max(dinfeas_a, dinfeas_p)
    pinfeas = _norm(aty, np.inf) / (abs(bty) + _EPS)

    # Primal residual tolearance relative scale.
    prelrhs = max(
        _norm(ax, np.inf) * inv_tau,
        _norm(s, np.inf) * inv_tau,
        _norm(self.b, np.inf),
    )

    # Dual residual tolearance relative scale.
    drelrhs = max(
        _norm(px, np.inf) * inv_tau,
        _norm(aty, np.inf) * inv_tau,
        _norm(self.c, np.inf),
    )

    norm_x = _norm(x, np.inf)
    norm_y = _norm(y, np.inf)
    norm_s = _norm(s, np.inf)

    if (
        gap < self.atol + self.rtol * min(abs(pcost), abs(dcost))
        and pres < self.atol + self.rtol * prelrhs
        and dres < self.atol + self.rtol * drelrhs
    ):
      status = SolutionStatus.SOLVED
    elif ctx < -_EPS and (
        dinfeas < self.atol_infeas + self.rtol_infeas * norm_x / abs(ctx)
    ):
      status = SolutionStatus.UNBOUNDED
    elif bty < -_EPS and (
        pinfeas < self.atol_infeas + self.rtol_infeas * norm_y / abs(bty)
    ):
      status = SolutionStatus.INFEASIBLE
    else:
      status = SolutionStatus.UNFINISHED

    stats_i.update({
        "iter": self.it,
        "ctx": ctx,
        "bty": bty,
        "pcost": pcost,
        "dcost": dcost,
        "pres": pres,
        "dres": dres,
        "gap": gap,
        "complementarity": complementarity,
        "pinfeas": pinfeas,
        "dinfeas": dinfeas,
        "dinfeas_a": dinfeas_a,
        "dinfeas_p": dinfeas_p,
        "mu": mu,
        "sigma": sigma,
        "alpha": alpha,
        "tau": tau_arr[0],
        "norm_x": norm_x,
        "norm_y": norm_y,
        "norm_s": norm_s,
        "status": status,
        "time": timeit.default_timer() - self.start_time,
        "prelrhs": prelrhs,
        "drelrhs": drelrhs,
        "max_sy": np.max(sy[self.z :]),
        "min_sy": np.min(sy[self.z :]),
        "std_sy": np.std(sy[self.z :]),
        "max_s_over_y": np.max(s_over_y[self.z :]),
        "min_s_over_y": np.min(s_over_y[self.z :]),
        "mean_s_over_y": np.mean(s_over_y[self.z :]),
        "std_s_over_y": np.std(s_over_y[self.z :]),
    })
    return status

  def _log_header(self):
    if self.verbose:
      print(f"{_SEPARA}\n{_HEADER}\n{_SEPARA}")

  def _log_iteration(self, stats_i: Dict[str, Any]):
    """Logs the iteration stats."""
    if not self.verbose:
      return
    infeas = min(stats_i["pinfeas"], stats_i["dinfeas"])

    # Parser for linear solver stats (handles stalled/failed sub-solves)
    def parse_ls(d):
      return " *" if d.get("status") == "stalled" else f"{d.get('solves', 0):2}"

    solves = (
        f"{parse_ls(stats_i['q_lin_sys_stats'])},"
        f"{parse_ls(stats_i['predictor_lin_sys_stats'])},"
        f"{parse_ls(stats_i['corrector_lin_sys_stats'])}"
    )
    print(
        f"| {stats_i['iter']:>4} | {stats_i['pcost']:>10.3e} |"
        f" {stats_i['dcost']:>10.3e} | {stats_i['pres']:>8.2e} |"
        f" {stats_i['dres']:>8.2e} | {stats_i['gap']:>8.2e} |"
        f" {infeas:>8.2e} | {stats_i['mu']:>8.2e} | {solves:>8} |"
        f" {stats_i['time']:>8.2e} |"
    )

  def _log_footer(self, message: str):
    if self.verbose:
      print(f"{_SEPARA}\n| {message}")