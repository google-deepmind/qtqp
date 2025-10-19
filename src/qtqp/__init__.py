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

_QTQP_EMOJI = "\N{smiling face with heart-shaped eyes}"
_UNBOUNDED_EMOJI = "\N{shocked face with exploding head}" * 3
_INFEASIBLE_EMOJI = "\N{face screaming in fear}" * 3
_FAILED_EMOJI = "\N{pouting face}" * 3
_SOLVED_EMOJI = "\N{rocket}" * 3


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
      p: sp.csc_matrix = None,
  ):
    """Initialize the QP solver.

    Args:
      a: Constraint matrix in CSC format.
      b: Right-hand side vector for the constraints.
      c: Cost vector for the objective function.
      z: The number of equality constraints (zero-cone size).
      p: QP matrix in CSC format. If None, assumed to be zero.
    """
    self.m, self.n = a.shape
    self.c = np.array(c)
    self.b = np.array(b)
    self.z = z

    if self.z >= self.m:
      raise ValueError(
          f"Number of equality constraints z={self.z} must be strictly less"
          f" than the number of rows in A ={self.m}"
      )

    if not sp.isspmatrix_csc(a):
      raise TypeError("Constraint matrix 'a' must be in CSC format.")
    self.a = a

    if p is None:
      self.p = sp.csc_matrix((self.n, self.n))
    else:
      if not sp.isspmatrix_csc(p):
        raise TypeError("QP matrix 'p' must be in CSC format.")
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
      min_static_regularization: float = 1e-7,
      max_iterative_refinement_steps: int = 50,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.SCIPY,
      verbose: bool = True,
      equilibrate: bool = True,
      x: np.ndarray = None,
      y: np.ndarray = None,
      s: np.ndarray = None,
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
          f"| {_QTQP_EMOJI} QTQP v{__version__} {_QTQP_EMOJI}:"
          f" m={self.m}, n={self.n}, z={self.z}, nnz(A)={self.a.nnz},"
          f" nnz(P)={self.p.nnz}, linear_solver={linear_solver.name}"
      )

    # --- Initialization ---
    x = np.zeros(self.n) if x is None else x
    if y is None:
      y = np.ones(self.m)
      y[: self.z] = 0.0
    s = np.ones(self.m) if s is None else s
    s[: self.z] = 0.0
    tau = np.array([1.0])

    if equilibrate:
      a, self.equilibrated_p, b, c, self.d, self.e = self._equilibrate()
      self.q = np.concatenate([c, b])
    else:
      a, self.equilibrated_p = self.a, self.p
      self.q = np.concatenate([self.c, self.b])

    if np.any(y[self.z :] < 0) or np.any(s[self.z :] < 0):
      raise ValueError("Initial y or s has negative values in the cone.")

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
    self.m_q = np.zeros_like(self.q)  # Initialize.
    self._log_header()
    # --- Main Iteration Loop ---
    for it in range(max_iter):
      self.it = it

      if equilibrate:
        x, y, s = self._equilibrate_iterates(x, y, s)

      x, y, tau, s = self._normalize(x, y, tau, s)
      mu = (y @ s) / max(1e-15, x @ x + y @ y + tau @ tau)
      self._linear_solver.update(mu=mu, s=s, y=y)

      # --- q = [c, b] Linear System Solve ---
      logging.debug("q linear system solve: %d", it)
      self.m_q, q_lin_sys_stats = self._linear_solver.solve(
          rhs=self.q,
          warm_start=self.m_q,
      )
      logging.debug("q_lin_sys_stats: %s", q_lin_sys_stats)

      # --- Predictor Step ---
      logging.debug("Predictor step: %d", it)
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
      logging.debug("predictor_lin_sys_stats: %s", predictor_lin_sys_stats)

      d_x_p, d_y_p, d_tau_p = x_p - x, y_p - y, tau_p - tau
      d_s_p = np.zeros(self.m)
      d_s_p[self.z :] = -y_p[self.z :] * s[self.z :] / y[self.z :]
      alpha_p = self._compute_step_size(y, s, d_y_p, d_s_p)
      sigma = self._compute_sigma(
          x, y, tau, s, alpha_p, d_x_p, d_y_p, d_tau_p, d_s_p
      )

      # --- Corrector Step ---
      logging.debug("Corrector step: %d", it)
      # Mehrotra's second-order correction term
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
      logging.debug("corrector_lin_sys_stats: %s", corrector_lin_sys_stats)

      # --- Update Iterates ---
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

      # Project onto non-negative cone to handle numerical errors
      y[self.z :] = np.maximum(y[self.z :], 1e-30)
      s[self.z :] = np.maximum(s[self.z :], 1e-30)
      tau = np.maximum(tau, 1e-30)

      if equilibrate:
        x, y, s = self._unequilibrate_iterates(x, y, s)

      # Check convergence, note these quantities are all **non-equilibrated**.
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

      if (
          gap < atol + rtol * min(abs(stats_i["pcost"]), abs(stats_i["dcost"]))
          and pres < atol + rtol * stats_i["prelrhs"]
          and dres < atol + rtol * stats_i["drelrhs"]
      ):
        self._log_footer(f"Solved {_SOLVED_EMOJI}")
        return Solution(x / tau, y / tau, s / tau, stats, SolutionStatus.SOLVED)

      ctx = stats_i["ctx"]
      if ctx < -1e-12:
        if dinfeas < atol_infeas + rtol_infeas * _norm(x, np.inf) / abs(ctx):
          self._log_footer(
              f"Dual infeasible / primal unbounded {_UNBOUNDED_EMOJI}"
          )
          y.fill(np.nan)
          return Solution(
              x / abs(ctx), y, s / abs(ctx), stats, SolutionStatus.UNBOUNDED
          )

      bty = stats_i["bty"]
      if bty < -1e-12:
        if pinfeas < atol_infeas + rtol_infeas * _norm(y, np.inf) / abs(bty):
          self._log_footer(
              f"Primal infeasible / dual unbounded {_INFEASIBLE_EMOJI}"
          )
          x.fill(np.nan)
          s.fill(np.nan)
          return Solution(x, y / abs(bty), s, stats, SolutionStatus.INFEASIBLE)

    self._log_footer(
        f"Failed to converge in {max_iter} iterations {_FAILED_EMOJI}"
    )
    return Solution(x / tau, y / tau, s / tau, stats, SolutionStatus.FAILED)

  def _equilibrate(self, num_iters=10, min_scale=1e-3, max_scale=1e3):
    """Equilibrate the data for better numerical stability."""
    # Initialize the equilibrated matrices.
    a, p, b, c = (self.a, self.p, self.b, self.c)
    # Initialize the equilibration matrices.
    d, e = (np.ones(self.m), np.ones(self.n))
    for i in range(num_iters):
      # Row norms
      d_i = sp.linalg.norm(a, np.inf, axis=1)
      d_i = 1 / (np.sqrt(d_i) + 1e-8)
      d_i = np.clip(d_i, min_scale, max_scale)
      # Column norms
      e_i = sp.linalg.norm(a, np.inf, axis=0)
      e_i = np.maximum(e_i, sp.linalg.norm(p, np.inf, axis=0))
      e_i = 1 / (np.sqrt(e_i) + 1e-8)
      e_i = np.clip(e_i, min_scale, max_scale)
      # Equilibrate rows and cols of A.
      a = sp.diags(d_i) @ a @ sp.diags(e_i)
      # Equilibrate P, it's symmetric and only affected by e.
      p = sp.diags(e_i) @ p @ sp.diags(e_i)
      # Update scaling matrices.
      d *= d_i
      e *= e_i
      logging.info(
          "Equilibration: iter %d: d_i err: %s, e_i err: %s",
          i,
          _norm(d_i - 1, np.inf),
          _norm(e_i - 1, np.inf),
      )

    # Return equiilibrated data and the equilibration matrices and scale.
    return a, p, b * d, c * e, d, e

  def _unequilibrate_iterates(self, x, y, s):
    """Unequilibrate the iterates."""
    return (self.e * x, self.d * y, s / self.d)

  def _equilibrate_iterates(self, x, y, s):
    """Unequilibrate the iterates."""
    return (x / self.e, y / self.d, s * self.d)

  def _max_step_size(self, y, delta_y):
    """Finds maximum step `alpha` in [0,  1] s.t. y + alpha * delta_y >= 0."""
    if y.size == 0:
      return 1.0
    # Identify indices where a step could violate non-negativity
    indices = delta_y < -1e-15
    if not np.any(indices):
      return 1.0
    # For these indices, calculate the largest step possible
    steps = -y[indices] / delta_y[indices]
    return min(1.0, np.min(steps))

  def _compute_sigma(self, x, y, tau, s, a, d_x, d_y, d_tau, d_s):
    """Computes the centering parameter sigma using Mehrotra's heuristic."""
    ip_old = (y @ s) / max(1e-15, x @ x + y @ y + tau @ tau)

    x_new = x + a * d_x
    y_new = y + a * d_y
    tau_new = tau + a * d_tau
    s_new = s + a * d_s

    ip_new = (y_new @ s_new) / max(
        1e-15, x_new @ x_new + y_new @ y_new + tau_new @ tau_new
    )

    # Heuristic for centering parameter
    sigma = (ip_new / max(1e-15, ip_old)) ** 3
    return max(0.0, min(sigma, 1.0))

  def _newton_step(
      self,
      *,
      mu,
      mu_target,
      rhs_anchor,
      tau_anchor,
      y,
      s,
      tau,
      cone_correction,
  ):
    """Computes a predictor or corrector step by solving the KKT system."""
    rhs_cone = mu_target / y[self.z :] + s[self.z :]
    if cone_correction is not None:
      rhs_cone += cone_correction

    r = np.concatenate([np.zeros(self.n + self.z), rhs_cone])
    r += (mu - mu_target) * rhs_anchor
    m_r, lin_sys_stats = self._linear_solver.solve(
        rhs=r,
        warm_start=rhs_anchor,
    )
    try:
      tau_plus = self._solve_for_tau(
          m_r, mu, mu_target, (mu - mu_target) * tau_anchor
      )
    except ValueError as e:
      logging.debug("Failed to solve for tau, falling back: %s", e)
      tau_plus = tau  # Fallback to previous tau on numerical failure

    vec_plus = m_r - self.m_q * tau_plus
    x_plus, y_plus = vec_plus[: self.n], vec_plus[self.n :]

    return x_plus, y_plus, tau_plus, lin_sys_stats

  def _solve_for_tau(self, m_r, mu, mu_target, r_tau):
    """Solves the quadratic equation for the tau step."""

    q, m_q = self.q, self.m_q
    v = self.equilibrated_p @ (np.vstack([m_r[: self.n], m_q[: self.n]]).T)
    p_m_r, p_m_q = v[:, 0], v[:, 1]

    # Coefficients of the quadratic equation t_a*tau^2 + t_b*tau + t_c = 0.
    t_a = mu + m_q @ q - m_q[: self.n] @ p_m_q
    t_b = -r_tau[0] - m_r @ q + m_r[: self.n] @ p_m_q + m_q[: self.n] @ p_m_r
    t_c = -m_r[: self.n] @ p_m_r - mu_target

    logging.debug("t_a=%s, t_b=%s, t_c=%s", t_a, t_b, t_c)

    if t_c > 1e-9:
      raise ValueError(f"Positive t_c encountered: {t_c=}")
    if t_a < 0.0:
      raise ValueError(f"Negative t_a encountered: {t_a=}")
    t_c = min(t_c, 0)  # Enforce non-positivity due to potential float errors

    discriminant = (t_b / t_a) ** 2 - 4 * t_c / t_a
    if discriminant < -1e-10:
      raise ValueError(f"Negative discriminant encountered: {discriminant=}")

    tau = (-t_b / t_a + np.sqrt(max(discriminant, 0))) / 2

    if tau < -1e-10:
      raise ValueError(f"Negative tau solution: {tau=}")
    return np.array([tau])

  def _normalize(self, x, y, tau, s):
    """Normalizes the homogeneous iterates."""
    vec_norm = np.sqrt(x @ x + y @ y + tau @ tau)
    scale = np.sqrt(self.m - self.z + 1) / max(1e-15, vec_norm)
    return x * scale, y * scale, tau * scale, s * scale

  def _compute_step_size(self, y, s, d_y, d_s):
    """Computes the primal and dual step size."""
    alpha_s = self._max_step_size(s[self.z :], d_s[self.z :])
    alpha_y = self._max_step_size(y[self.z :], d_y[self.z :])
    return min(alpha_s, alpha_y)

  def _compute_residuals(self, x, y, tau, s, alpha, mu, sigma):
    """Compute the primal and dual residuals and populate the stats dict."""

    tau = tau[0]  # Unbox.
    ax = self.a @ x
    aty = self.a.T @ y
    px = self.p @ x
    xpx = x @ px
    ctx = self.c @ x
    bty = self.b @ y

    stats = dict(
        iter=self.it,
        ctx=ctx,
        bty=bty,
        pcost=ctx / tau + 0.5 * xpx / tau / tau,
        dcost=-bty / tau - 0.5 * xpx / tau / tau,
        pres=_norm(ax / tau + s / tau - self.b, np.inf),
        dres=_norm(px / tau + aty / tau + self.c, np.inf),
        complementarity=np.abs((y @ s) / tau / tau),
        gap=np.abs(ctx / tau + bty / tau + xpx / tau / tau),
        dinfeas_a=_norm((ax + s) / (abs(ctx) + 1e-15), np.inf),
        dinfeas_p=_norm(px / (abs(ctx) + 1e-15), np.inf),
        pinfeas=_norm(aty / (abs(bty) + 1e-15), np.inf),
        mu=mu,
        sigma=sigma,
        tau=tau,
        alpha=alpha,
        time=timeit.default_timer() - self.start_time,
        vec_norm=np.sqrt(x @ x + y @ y + tau * tau),
        prelrhs=max(
            _norm(ax, np.inf) / max(tau, 1e-12),
            _norm(s, np.inf) / max(tau, 1e-12),
            _norm(self.b, np.inf),
        ),
        drelrhs=max(
            _norm(px, np.inf) / max(tau, 1e-12),
            _norm(aty, np.inf) / max(tau, 1e-12),
            _norm(self.c, np.inf),
        ),
    )
    return (
        stats["pres"],
        stats["dres"],
        stats["gap"],
        stats["pinfeas"],
        max(stats["dinfeas_a"], stats["dinfeas_p"]),
        stats,
    )

  def _log_header(self):
    if self.verbose:
      print(_SEPARA + "\n" + _HEADER + "\n" + _SEPARA)

  def _log_iteration(self, result: Dict[str, Any]):
    """Logs the iteration stats."""
    if not self.verbose:
      return
    infeas = min(
        result["pinfeas"], max(result["dinfeas_a"], result["dinfeas_p"])
    )
    parser = lambda d: " *" if d["status"] == "stalled" else f"{d['solves']:2}"
    solves = (
        f"{parser(result['q_lin_sys_stats'])},"
        f"{parser(result['predictor_lin_sys_stats'])},"
        f"{parser(result['corrector_lin_sys_stats'])}"
    )
    print(
        f"| {result['iter']:>4} | {result['pcost']:>10.3e} |"
        f" {result['dcost']:>10.3e} | {result['pres']:>8.2e} |"
        f" {result['dres']:>8.2e} | {result['gap']:>8.2e} |"
        f" {infeas:>8.2e} | {result['mu']:>8.2e} | {result['sigma']:>8.2e} |"
        f" {result['alpha']:>8.2e} | {solves} | {result['time']:>8.2e} |"
    )

  def _log_footer(self, message: str):
    if self.verbose:
      print(_SEPARA + "\n" + f"| {message}")
