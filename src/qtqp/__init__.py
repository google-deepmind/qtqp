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

__version__ = "0.0.1"
_HEADER = """| iter |      pcost |      dcost |     pres |     dres |      gap |   infeas |       mu |    sigma |    alpha |  q, p, c |     time |"""
_SEPARA = """|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|"""
_norm = np.linalg.norm
_EPS = 1e-15  # Standard epsilon for numerical safety


class LinearSolver(enum.Enum):
  """Available linear solvers."""

  SCIPY = direct.ScipySolver
  PARDISO = direct.MklPardisoSolver
  QDLDL = direct.QdldlSolver
  CHOLMOD = direct.CholModSolver
  CUDSS = direct.CuDssSolver


class SolutionStatus(enum.Enum):
  """Possible statuses of the QP solution."""

  SOLVED = "solved"
  INFEASIBLE = "infeasible"
  UNBOUNDED = "unbounded"
  FAILED = "failed"


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

    # Internal buffers
    self._r_buffer = np.zeros(self.n + self.m)

  def solve(
      self,
      *,
      atol: float = 1e-7,
      rtol: float = 1e-8,
      atol_infeas: float = 1e-8,
      rtol_infeas: float = 1e-9,
      max_iter: int = 100,
      step_size_scale: float = 0.99,
      min_static_regularization: float = 1e-7,
      max_iterative_refinement_steps: int = 50,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.SCIPY,
      verbose: bool = True,
      equilibrate: bool = True,
      x: np.ndarray | None = None,
      y: np.ndarray | None = None,
      s: np.ndarray | None = None,
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
        for the linear solves.
      linear_solver_atol (float): Absolute tolerance for the iterative
        refinement process within the linear solver.
      linear_solver_rtol (float): Relative tolerance for the iterative
        refinement process within the linear solver.
      linear_solver (LinearSolver): The linear solver to use.
      verbose (bool): If True, prints a summary of each iteration.
      equilibrate (bool): If True, equilibrate the data for better numerical
        stability.
      x: Initial primal solution vector.
      y: Initial dual solution vector.
      s: Initial slack vector.

    Returns:
      A Solution object containing the solution and solve stats.
    """
    self.start_time = timeit.default_timer()
    self.verbose = verbose
    if self.verbose:
      print(
          f"| QTQP v{__version__}:"
          f" m={self.m}, n={self.n}, z={self.z}, nnz(A)={self.a.nnz},"
          f" nnz(P)={self.p.nnz}, linear_solver={linear_solver.name}"
      )

    # --- Initialization ---
    # Use supplied warm-starts or default cold-starts.
    x = np.zeros(self.n) if x is None else np.array(x, dtype=np.float64)
    if y is None:
      y = np.zeros(self.m)
      # Initialize inequality duals to 1.0 for interiority
      y[self.z :] = 1.0
    else:
      y = np.array(y, dtype=np.float64)
      if y.shape != (self.m,):
        raise ValueError(f"y must have shape ({self.m},), got {y.shape}")

    if s is None:
      s = np.zeros(self.m)
      # Initialize inequality slacks to 1.0 for interiority
      s[self.z :] = 1.0
    else:
      s = np.array(s, dtype=np.float64)
      if s.shape != (self.m,):
        raise ValueError(f"s must have shape ({self.m},), got {s.shape}")

    # tau is homogeneous embedding variable. Kept as 1-element array for
    # consistent vector operations (e.g., @ operator).
    tau = np.array([1.0])

    # Check for valid initial interior point if supplied
    if np.any(y[self.z :] < 0) or np.any(s[self.z :] < 0):
      raise ValueError("Initial y or s has negative values in the pos cone.")
    if np.any(s[: self.z] != 0):
      raise ValueError("Initial s has nonzero values in the zero cone.")

    if equilibrate:
      a, self.equilibrated_p, b, c, self.d, self.e = self._equilibrate()
      self.q = np.concatenate([c, b])
    else:
      a, self.equilibrated_p = self.a, self.p
      self.q = np.concatenate([self.c, self.b])

    self._linear_solver = direct.DirectKktSolver(
        a=a,
        p=self.equilibrated_p,
        z=self.z,
        min_static_regularization=min_static_regularization,
        max_iterative_refinement_steps=max_iterative_refinement_steps,
        atol=linear_solver_atol,
        rtol=linear_solver_rtol,
        solver=linear_solver.value(),
    )

    stats = []
    self.m_q = np.zeros_like(self.q)
    self._log_header()

    # --- Main Iteration Loop ---
    for it in range(max_iter):
      self.it = it

      if equilibrate:
        x, y, s = self._equilibrate_iterates(x, y, s)

      x, y, tau, s = self._normalize(x, y, tau, s)

      # Calculate current complementary slackness error (mu)
      mu = (y @ s) / max(_EPS, x @ x + y @ y + tau @ tau)
      self._linear_solver.update(mu=mu, s=s, y=y)

      # --- Step 1: Solve for M * q ---
      # This is reused for both predictor and corrector parts of the step.
      self.m_q, q_lin_sys_stats = self._linear_solver.solve(
          rhs=self.q, warm_start=self.m_q
      )

      # --- Step 2: Predictor (Affine) Step ---
      # Solve KKT with mu_target = 0 to find pure Newton direction.
      x_p, y_p, tau_p, predictor_lin_sys_stats = self._newton_step(
          mu=mu,
          mu_target=0.0,
          rhs_anchor=np.concatenate([x, y]),
          tau_anchor=tau,
          y=y,
          s=s,
          tau=tau,
          cone_correction=None,
      )

      d_x_p, d_y_p, d_tau_p = x_p - x, y_p - y, tau_p - tau
      d_s_p = np.zeros(self.m)
      d_s_p[self.z :] = -y_p[self.z :] * s[self.z :] / y[self.z :]

      # Compute predictor step size and resulting centering parameter (sigma)
      alpha_p = self._compute_step_size(y, s, d_y_p, d_s_p)
      sigma = self._compute_sigma(
          x, y, tau, s, alpha_p, d_x_p, d_y_p, d_tau_p, d_s_p
      )

      # --- Step 3: Corrector Step ---
      # Mehrotra correction term handles nonlinearity in complementarity.
      cone_correction = -d_s_p[self.z :] * d_y_p[self.z :] / y[self.z :]

      x_c, y_c, tau_c, corrector_lin_sys_stats = self._newton_step(
          mu=mu,
          mu_target=sigma * mu,
          rhs_anchor=np.concatenate([x_p, y_p]),
          tau_anchor=tau_p,
          y=y,
          s=s,
          tau=tau,
          cone_correction=cone_correction,
      )

      # --- Step 4: Update Iterates ---
      d_x, d_y, d_tau = x_c - x, y_c - y, tau_c - tau
      d_s = np.zeros(self.m)
      d_s[self.z :] = (
          sigma * mu / y[self.z :]
          + cone_correction
          - y_c[self.z :] * s[self.z :] / y[self.z :]
      )

      alpha = self._compute_step_size(y, s, d_y, d_s)
      step_size = step_size_scale * alpha
      x, y, tau, s = self._normalize(
          x + step_size * d_x,
          y + step_size * d_y,
          tau + step_size * d_tau,
          s + step_size * d_s,
      )

      # Ensure variables stay strictly in the cone to prevent numerical issues.
      # 1e-30 is a safe small number that is >> 0 but << tolerances.
      y[self.z :] = np.maximum(y[self.z :], 1e-30)
      s[self.z :] = np.maximum(s[self.z :], 1e-30)
      tau = np.maximum(tau, 1e-30)

      if equilibrate:
        x, y, s = self._unequilibrate_iterates(x, y, s)

      # --- Termination Check (non-equilibrated values)---
      (pres, dres, gap, pinfeas, dinfeas, stats_i) = self._compute_residuals(
          x, y, tau, s, alpha, mu, sigma
      )
      stats_i.update(
          q_lin_sys_stats=q_lin_sys_stats,
          predictor_lin_sys_stats=predictor_lin_sys_stats,
          corrector_lin_sys_stats=corrector_lin_sys_stats,
      )
      self._log_iteration(stats_i)
      stats.append(stats_i)

      # Success criteria: primal/dual residuals and gap are below tolerances
      if (
          gap < atol + rtol * min(abs(stats_i["pcost"]), abs(stats_i["dcost"]))
          and pres < atol + rtol * stats_i["prelrhs"]
          and dres < atol + rtol * stats_i["drelrhs"]
      ):
        self._log_footer("Solved")
        return Solution(x / tau, y / tau, s / tau, stats, SolutionStatus.SOLVED)

      ctx = stats_i["ctx"]
      if ctx < -1e-12:
        if dinfeas < atol_infeas + rtol_infeas * _norm(x, np.inf) / abs(ctx):
          self._log_footer("Dual infeasible / primal unbounded")
          y.fill(np.nan)
          return Solution(
              x / abs(ctx), y, s / abs(ctx), stats, SolutionStatus.UNBOUNDED
          )

      bty = stats_i["bty"]
      if bty < -1e-12:
        if pinfeas < atol_infeas + rtol_infeas * _norm(y, np.inf) / abs(bty):
          self._log_footer("Primal infeasible / dual unbounded")
          x.fill(np.nan)
          s.fill(np.nan)
          return Solution(x, y / abs(bty), s, stats, SolutionStatus.INFEASIBLE)

    self._log_footer(f"Failed to converge in {max_iter} iterations")
    return Solution(x / tau, y / tau, s / tau, stats, SolutionStatus.FAILED)

  def _equilibrate(self, num_iters=10, min_scale=1e-3, max_scale=1e3):
    """Ruiz equilibration to improve numerical conditioning."""
    # Initialize the equilibrated matrices.
    a, p, b, c = (self.a, self.p, self.b, self.c)
    # Initialize the equilibration matrices.
    d, e = (np.ones(self.m), np.ones(self.n))

    for i in range(num_iters):
      # Row norms (infinity norm)
      # Add small epsilon to avoid division by zero for zero rows
      d_i = sp.linalg.norm(a, np.inf, axis=1) + _EPS
      d_i = 1.0 / np.sqrt(d_i)
      d_i = np.clip(d_i, min_scale, max_scale)

      # Column norms (max of A col norms and P col norms)
      e_i_a = sp.linalg.norm(a, np.inf, axis=0)
      e_i_p = sp.linalg.norm(p, np.inf, axis=0)
      e_i = np.maximum(e_i_a, e_i_p) + _EPS
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

  def _compute_sigma(self, x, y, tau, s, alpha, d_x, d_y, d_tau, d_s) -> float:
    """Computes the centering parameter sigma using Mehrotra's heuristic."""
    # Current complementarity
    mu_curr = (y @ s) / max(_EPS, x @ x + y @ y + tau @ tau)

    # Projected complementarity after affine step
    x_aff = x + alpha * d_x
    y_aff = y + alpha * d_y
    tau_aff = tau + alpha * d_tau
    s_aff = s + alpha * d_s
    mu_aff = (y_aff @ s_aff) / max(
        _EPS, x_aff @ x_aff + y_aff @ y_aff + tau_aff @ tau_aff
    )

    # If affine step reduces mu significantly, use small sigma (aggressive)
    sigma = (mu_aff / max(_EPS, mu_curr)) ** 3
    return np.clip(sigma, 0.0, 1.0)

  def _newton_step(
      self, *, mu, mu_target, rhs_anchor, tau_anchor, y, s, tau, cone_correction
  ):
    """Computes a search direction by solving the augmented KKT system."""
    # Prepare RHS for the linear system.
    # r = [0, ..., 0, rhs_cone] + (mu - mu_target) * rhs_anchor
    # We use a preallocated buffer to avoid repeated concatenation.
    self._r_buffer.fill(0.0)

    rhs_cone = mu_target / y[self.z :] + s[self.z :]
    if cone_correction is not None:
      rhs_cone += cone_correction

    self._r_buffer[self.n + self.z :] = rhs_cone
    self._r_buffer += (mu - mu_target) * rhs_anchor

    m_r, lin_sys_stats = self._linear_solver.solve(
        rhs=self._r_buffer,
        warm_start=rhs_anchor,
    )

    # Solve the 1D quadratic equation for the homogeneous tau component
    try:
      tau_plus = self._solve_for_tau(
          m_r, mu, mu_target, (mu - mu_target) * tau_anchor
      )
    except ValueError as e:
      # Fallback if quadratic solve fails numerically (rare but possible)
      logging.warning("Tau solve failed, using previous tau. Error: %s", e)
      tau_plus = tau

    # Reconstruct full (x, y) step from KKT solution components
    vec_plus = m_r - self.m_q * tau_plus
    x_plus, y_plus = vec_plus[: self.n], vec_plus[self.n :]

    return x_plus, y_plus, tau_plus, lin_sys_stats

  def _solve_for_tau(self, m_r, mu, mu_target, r_tau) -> np.ndarray:
    """Solves the quadratic equation for the tau step in homogeneous embedding."""
    # Solve a quadratic equation for tau: t_a * tau^2 + t_b * tau + t_c = 0
    n = self.n
    q, m_q = self.q, self.m_q

    # v = P @ [m_r_x, m_q_x]^T
    v = self.equilibrated_p @ np.stack([m_r[:n], m_q[:n]], axis=1)
    p_m_r, p_m_q = v[:, 0], v[:, 1]

    t_a = mu + m_q @ q - m_q[:n] @ p_m_q
    t_b = -r_tau[0] - m_r @ q + m_r[:n] @ p_m_q + m_q[:n] @ p_m_r
    t_c = -m_r[:n] @ p_m_r - mu_target
    logging.debug("t_a=%s, t_b=%s, t_c=%s", t_a, t_b, t_c)

    # Theoretical guarantees state t_a > 0 and t_c <= 0.
    # We allow small numerical violations but error on large ones.
    if t_a < -1e-9:
      raise ValueError(f"t_a should be positive, got {t_a}")
    t_a = max(t_a, _EPS)  # Ensure strictly positive for division

    if t_c > 1e-9:
      raise ValueError(f"t_c should be non-positive, got {t_c}")
    t_c = min(t_c, 0.0)

    # Standard quadratic formula for the positive root
    discriminant = t_b**2 - 4 * t_a * t_c
    if discriminant < -1e-9:
      raise ValueError(f"Negative discriminant: {discriminant}")

    tau_sol = (-t_b + np.sqrt(max(0.0, discriminant))) / (2 * t_a)

    if tau_sol < -1e-10:
      raise ValueError(f"Negative tau solution found: {tau_sol}")

    return np.array([max(0.0, tau_sol)])

  def _normalize(self, x, y, tau, s):
    """Normalizes the homogeneous iterates."""
    vec_norm = np.sqrt(x @ x + y @ y + tau @ tau)
    scale = np.sqrt(self.m - self.z + 1) / max(1e-15, vec_norm)
    return x * scale, y * scale, tau * scale, s * scale

  def _compute_step_size(self, y, s, d_y, d_s) -> float:
    """Computes the maximum standard primal-dual step size."""
    alpha_s = self._max_step_size(s[self.z :], d_s[self.z :])
    alpha_y = self._max_step_size(y[self.z :], d_y[self.z :])
    return min(alpha_s, alpha_y)

  def _compute_residuals(self, x, y, tau_arr, s, alpha, mu, sigma):
    """Compute convergence residuals and statistics."""
    tau = tau_arr[0]  # Unbox
    inv_tau = 1.0 / max(tau, _EPS)

    # Precompute commonly used matrix-vector products
    ax = self.a @ x
    aty = self.a.T @ y
    px = self.p @ x
    xpx = x @ px
    ctx = self.c @ x
    bty = self.b @ y

    # Unscaled costs
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
    # Primal residual check relative scale.
    prelrhs = max(
        _norm(ax, np.inf) * inv_tau,
        _norm(s, np.inf) * inv_tau,
        _norm(self.b, np.inf),
    )
    # Dual residual check relative scale.
    drelrhs = max(
        _norm(px, np.inf) * inv_tau,
        _norm(aty, np.inf) * inv_tau,
        _norm(self.c, np.inf),
    )
    stats = {
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
        "tau": tau,
        "time": timeit.default_timer() - self.start_time,
        "prelrhs": prelrhs,
        "drelrhs": drelrhs,
    }

    return pres, dres, gap, pinfeas, dinfeas, stats

  def _log_header(self):
    if self.verbose:
      print(f"{_SEPARA}\n{_HEADER}\n{_SEPARA}")

  def _log_iteration(self, result: Dict[str, Any]):
    """Logs the iteration stats."""
    if not self.verbose:
      return
    infeas = min(result["pinfeas"], result["dinfeas"])

    # Parser for linear solver stats (handles stalled/failed sub-solves)
    def parse_ls(d):
      return " *" if d.get("status") == "stalled" else f"{d.get('solves', 0):2}"

    solves = (
        f"{parse_ls(result['q_lin_sys_stats'])},"
        f"{parse_ls(result['predictor_lin_sys_stats'])},"
        f"{parse_ls(result['corrector_lin_sys_stats'])}"
    )
    print(
        f"| {result['iter']:>4} | {result['pcost']:>10.3e} |"
        f" {result['dcost']:>10.3e} | {result['pres']:>8.2e} |"
        f" {result['dres']:>8.2e} | {result['gap']:>8.2e} |"
        f" {infeas:>8.2e} | {result['mu']:>8.2e} | {result['sigma']:>8.2e} |"
        f" {result['alpha']:>8.2e} | {solves:>8} | {result['time']:>8.2e} |"
    )

  def _log_footer(self, message: str):
    if self.verbose:
      print(f"{_SEPARA}\n| {message}")
