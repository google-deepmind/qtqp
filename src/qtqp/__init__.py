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

"""Interior point method for solving QPs."""

import dataclasses
import enum
import logging
import timeit
from typing import Any, Dict, List

import numpy as np
import scipy.sparse as sp

from . import direct

__version__ = "0.0.3"
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
    assert min_static_regularization >= 0
    assert max_iterative_refinement_steps >= 1
    assert linear_solver_atol >= 0
    assert linear_solver_rtol >= 0

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

    if self.equilibrate:
      a, p, b, c, self.d, self.e = self._equilibrate()
      x, y, s = self._equilibrate_iterates(x, y, s)
    else:
      a, p, b, c, self.d, self.e = self.a, self.p, self.b, self.c, None, None

    # q = [c; b]: the tau-scaled part of the KKT right-hand side. The primal and
    # dual feasibility conditions at optimality can be written as:
    #   K @ [x; y] = -q * tau  (where K is the augmented KKT matrix)
    # so the full Newton RHS has the form r - q * tau+. Solving K @ kinv_q = q
    # once per iteration lets us write the parametric solution as:
    #   [x+; y+] = kinv_r - kinv_q * tau+
    # and reuse kinv_q in both the predictor and corrector steps.
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
    self.kinv_q = np.zeros_like(self.q)  # K^{-1}q, warm-started across iterations.
    status = SolutionStatus.UNFINISHED
    self._log_header()

    # Pre-allocate [x; y] and ds to avoid repeated allocation each iteration.
    xy = np.empty(self.n + self.m)   # Combined primal-dual vector [x; y]
    ds = np.zeros(self.m)            # Slack step direction; ds[:z] is always 0

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
      xy[: self.n] = x
      xy[self.n :] = y
      x_p, y_p, tau_p, predictor_lin_sys_stats = self._newton_step(
          p=p,
          mu=mu,
          mu_target=0.0,
          r_anchor=xy,
          tau_anchor=tau,
          y=y,
          s=s,
          tau=tau,
          correction=None,
      )
      stats_i.update(predictor_lin_sys_stats=predictor_lin_sys_stats)

      d_x_p, d_y_p, d_tau_p = x_p - x, y_p - y, tau_p - tau
      # Predictor slack step from the linearized complementarity condition with
      # target=0: (y + d_y)(s + d_s) ≈ 0 => d_s = -(y + d_y)*s/y = -y_p*s/y.
      ds[self.z :] = -y_p[self.z :] * s[self.z :] / y[self.z :]

      # Compute predictor step size and resulting centering parameter (sigma)
      alpha_p = self._compute_step_size(y, s, d_y_p, ds)
      sigma = self._compute_sigma(
          mu, x, y, tau, s, alpha_p, d_x_p, d_y_p, d_tau_p, ds
      )

      # --- Step 3: Corrector Step ---
      # Mehrotra's second-order correction accounts for the nonlinear cross-term
      # that the predictor's linear approximation ignores. Expanding the full
      # complementarity condition to second order:
      #   (y + d_y)(s + d_s) = sigma*mu
      #   => y*d_s + s*d_y + d_y*d_s = sigma*mu - y*s
      # The predictor solved the linearized version (dropping d_y*d_s). Here we
      # feed the predictor's cross-term d_y_p*d_s_p back into the corrector RHS
      # (divided by y because the KKT complementarity block is scaled by 1/y),
      # so the corrector step can cancel it and land closer to the target.
      correction = -ds[self.z :] * d_y_p[self.z :] / y[self.z :]
      xy[: self.n] = x_p
      xy[self.n :] = y_p
      x_c, y_c, tau_c, corrector_lin_sys_stats = self._newton_step(
          p=p,
          mu=mu,
          mu_target=sigma * mu,
          r_anchor=xy,
          tau_anchor=tau_p,
          y=y,
          s=s,
          tau=tau,
          correction=correction,
      )
      stats_i.update(corrector_lin_sys_stats=corrector_lin_sys_stats)

      # --- Step 4: Update Iterates ---
      d_x, d_y, d_tau = x_c - x, y_c - y, tau_c - tau
      # Combined corrector slack step: same form as the predictor but now with
      # centering target sigma*mu and the Mehrotra correction baked in.
      ds[self.z :] = (
          sigma * mu / y[self.z :]
          + correction
          - y_c[self.z :] * s[self.z :] / y[self.z :]
      )

      alpha = self._compute_step_size(y, s, d_y, ds)
      x += step_size_scale * alpha * d_x
      y += step_size_scale * alpha * d_y
      tau += step_size_scale * alpha * d_tau
      s += step_size_scale * alpha * ds

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

      # Apply scaling directly to CSC data arrays, avoiding temporary sparse matrices.
      # D @ A @ E: scale non-zero at row r, col c by d_i[r] * e_i[c].
      col_scale_a = np.repeat(e_i, np.diff(a.indptr))
      a = a.copy()
      a.data *= d_i[a.indices] * col_scale_a
      # E @ P @ E: scale non-zero at row r, col c by e_i[r] * e_i[c].
      if p.nnz > 0:
        col_scale_p = np.repeat(e_i, np.diff(p.indptr))
        p = p.copy()
        p.data *= e_i[p.indices] * col_scale_p

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

    # Compute mu_aff directly without calling _normalize to avoid 4 extra
    # allocations. Equivalent to: normalize then compute (y @ s) / (m - z).
    # scale = sqrt(m-z+1) / max(_EPS, ||(x,y,tau)||), so scale^2 = (m-z+1) /
    # max(_EPS^2, ||(x,y,tau)||^2), giving mu_aff = scale^2 * (y_aff @ s_aff).
    xyt_norm_sq = x_aff @ x_aff + y_aff @ y_aff + tau_aff @ tau_aff
    scale_sq = (self.m - self.z + 1) / max(_EPS**2, xyt_norm_sq)
    mu_aff = scale_sq * (y_aff @ s_aff) / (self.m - self.z)

    # sigma = (mu_aff / mu)^3: Mehrotra's heuristic. If the affine step already
    # drives mu close to zero, sigma is small (aggressive, little centering).
    # If mu_aff ≈ mu (affine step didn't help much), sigma ≈ 1 (full centering).
    # The cubic exponent amplifies the contrast, pushing sigma toward 0 or 1.
    sigma = (mu_aff / max(_EPS, mu_curr)) ** 3
    return np.clip(sigma, 0.0, 1.0)

  def _newton_step(
      self, *, p, mu, mu_target, r_anchor, tau_anchor, y, s, tau, correction
  ):
    """Computes a Newton search direction by solving the augmented KKT system.

    The KKT system K @ [x+; y+] = r - q * tau+ is linear in tau+, giving the
    parametric solution:
        [x+; y+] = K^{-1}(r) - K^{-1}(q) * tau+  =  kinv_r - kinv_q * tau+
    tau+ is then pinned by substituting this back into the tau equation of the
    homogeneous embedding (see _solve_for_tau).

    Args:
      p: The quadratic cost matrix.
      mu: Current barrier parameter.
      mu_target: Complementarity target for this step (0 for predictor,
        sigma*mu for corrector).
      r_anchor: The current [x; y] iterate (as a single vector), used to
        form the Newton RHS via r = (mu - mu_target) * r_anchor + ...
      tau_anchor: The current tau iterate.
      y: Current dual variables.
      s: Current slack variables.
      tau: Current homogeneous variable (used as fallback if tau solve fails).
      correction: Optional Mehrotra second-order correction added to the
        complementarity block of the RHS.
    """
    # Prepare RHS for the linear system (in-place to avoid r_cone allocation).
    r = (mu - mu_target) * r_anchor
    if mu_target != 0.0:
      r[self.n + self.z :] += mu_target / y[self.z :]
    r[self.n + self.z :] += s[self.z :]
    if correction is not None:
      r[self.n + self.z :] += correction

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
    """Solves for tau+ using the homogeneous embedding's tau equation.

    The parametric KKT solution is:
        [x+; y+] = kinv_r - kinv_q * tau+

    Substituting this into the tau equation of the homogeneous embedding — which
    encodes the energy balance of the self-dual formulation including the
    quadratic term x+' P x+ — yields:
        t_a * tau+^2 + t_b * tau+ + t_c = 0

    The coefficients t_a, t_b, t_c are computed from inner products of kinv_r
    and kinv_q with q and P. For LPs (P=0) the P terms drop out. We always take
    the positive root since tau > 0 is required for the embedding to represent
    a feasible point (tau=0 corresponds to a certificate of infeasibility or
    unboundedness, which is handled separately at termination).
    """
    # Coefficients of the quadratic t_a * tau+^2 + t_b * tau+ + t_c = 0.
    n = self.n
    q, kinv_q = self.q, self.kinv_q

    if p.nnz == 0:
      # LP case: all P terms drop out of the quadratic coefficients.
      t_a = mu + kinv_q @ q
      t_b = -r_tau[0] - kinv_r @ q
      t_c = float(-mu_target)
    else:
      v = p @ np.stack([kinv_r[:n], kinv_q[:n]], axis=1)
      p_kinv_r, p_kinv_q = v[:, 0], v[:, 1]
      t_a = mu + kinv_q @ q - kinv_q[:n] @ p_kinv_q
      t_b = -r_tau[0] - kinv_r @ q + kinv_r[:n] @ p_kinv_q + kinv_q[:n] @ p_kinv_r
      t_c = -kinv_r[:n] @ p_kinv_r - mu_target
    logging.debug("t_a=%s, t_b=%s, t_c=%s", t_a, t_b, t_c)

    # Standard quadratic formula for the positive root
    if abs(t_a) < _EPS:
      raise ValueError(f"Near-zero t_a={t_a}, cannot solve for tau")

    discriminant = t_b**2 - 4 * t_a * t_c
    if discriminant < -1e-9:
      raise ValueError(f"Negative discriminant: {discriminant}")

    tau_sol = (-t_b + np.sqrt(max(0.0, discriminant))) / (2 * t_a)

    if not np.isfinite(tau_sol) or tau_sol < -1e-10:
      raise ValueError(f"Invalid tau solution found: {tau_sol}")

    return np.array([max(0.0, tau_sol)])

  def _normalize(self, x, y, tau, s):
    """Normalizes iterates onto the homogeneous self-dual embedding central path.

    The homogeneous self-dual embedding (HSDE) lifts the QP into a projective
    space. Only ratios like x/tau and y/tau matter — tau is the homogeneous
    variable, and the physical solution is recovered as (x/tau, y/tau, s/tau).

    To keep iterates well-scaled and prevent drift, we enforce:
        ||(x, y, tau)||^2 = m - z + 1
    The right-hand side counts complementarity pairs: (m - z) from the
    inequality constraints plus 1 for the tau-kappa pair of the embedding.

    Scaling s by the same factor as (x, y, tau) preserves
        mu = (y @ s) / (m - z)
    since y and s are scaled identically and the factor cancels.
    """
    xyt_norm = np.sqrt(x @ x + y @ y + tau @ tau)
    scale = np.sqrt(self.m - self.z + 1) / max(_EPS, xyt_norm)
    return x * scale, y * scale, tau * scale, s * scale

  def _compute_step_size(self, y, s, d_y, d_s) -> float:
    """Computes the maximum standard primal-dual step size."""
    alpha_s = self._max_step_size(s[self.z :], d_s[self.z :])
    alpha_y = self._max_step_size(y[self.z :], d_y[self.z :])
    return min(alpha_s, alpha_y)

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
    if self.p.nnz == 0:
      px = np.zeros(self.n)
      xpx = 0.0
    else:
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
