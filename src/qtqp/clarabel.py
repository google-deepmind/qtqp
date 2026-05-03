import logging
import math
import timeit
from typing import Any, Dict, List

import numpy as np
import scipy.sparse as sp

from . import (
    QTQP,
    EquilibrationMethod,
    Initialization,
    LinearSolver,
    Solution,
    SolutionStatus,
    __version__,
    _EPS,
    _HEADER,
    _SEPARA,
    _norm,
    _resolve_linear_solver,
    direct,
)


class Clarabel(QTQP):
  """Clarabel interior point method for quadratic programs.

  Implements the Clarabel algorithm (Goulart & Chen, 2024) which differs from
  the base QTQP Mehrotra predictor-corrector in several ways:
    - No normalization of iterates
    - Explicit tracking of tau and kappa
    - Centering parameter sigma = (1 - alpha_aff)^3
    - Two-part Newton step decomposition via the G operator
    - KKT regularization via min_static_regularization (mu=0 in update)

  The `y` variable in this code corresponds to `z` in the Clarabel paper.
  """

  def solve(
      self,
      *,
      atol: float = 1e-7,
      rtol: float = 1e-8,
      atol_infeas: float = 1e-8,
      rtol_infeas: float = 1e-9,
      max_iter: int = 100,
      time_limit_secs: float | None = None,
      step_size_scale: float = 0.99,
      min_static_regularization: float = 1e-8,
      max_iterative_refinement_steps: int = 20,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.AUTO,
      refinement_strategy: direct.RefinementStrategy = "fixed_point",
      fgmres_restart: int = 10,
      legacy_stall_check: bool = False,
      equilibrate_per_iteration: int = 0,
      equilibrate: bool = True,
      equilibration_method: EquilibrationMethod = "kkt",
      verbose: bool = True,
      initialization: Initialization = Initialization.TRIVIAL,
      collect_stats: bool = False,
  ) -> Solution:
    """Solves the QP using the Clarabel interior point method."""
    self._linear_solver = None
    try:
      return self._solve_impl(
          atol=atol,
          rtol=rtol,
          atol_infeas=atol_infeas,
          rtol_infeas=rtol_infeas,
          max_iter=max_iter,
          time_limit_secs=time_limit_secs,
          step_size_scale=step_size_scale,
          min_static_regularization=min_static_regularization,
          max_iterative_refinement_steps=max_iterative_refinement_steps,
          linear_solver_atol=linear_solver_atol,
          linear_solver_rtol=linear_solver_rtol,
          linear_solver=linear_solver,
          refinement_strategy=refinement_strategy,
          fgmres_restart=fgmres_restart,
          legacy_stall_check=legacy_stall_check,
          equilibrate_per_iteration=equilibrate_per_iteration,
          equilibrate=equilibrate,
          equilibration_method=equilibration_method,
          verbose=verbose,
          initialization=initialization,
          collect_stats=collect_stats,
      )
    finally:
      if self._linear_solver is not None:
        self._linear_solver.free()
        self._linear_solver = None

  def _solve_impl(
      self,
      *,
      atol: float,
      rtol: float,
      atol_infeas: float,
      rtol_infeas: float,
      max_iter: int,
      time_limit_secs: float | None,
      step_size_scale: float,
      min_static_regularization: float,
      max_iterative_refinement_steps: int,
      linear_solver_atol: float,
      linear_solver_rtol: float,
      linear_solver: LinearSolver,
      refinement_strategy: direct.RefinementStrategy,
      fgmres_restart: int,
      legacy_stall_check: bool,
      equilibrate_per_iteration: int,
      equilibrate: bool,
      equilibration_method: EquilibrationMethod,
      verbose: bool,
      initialization: Initialization,
      collect_stats: bool,
  ) -> Solution:
    assert atol >= 0
    assert rtol >= 0
    assert atol_infeas >= 0
    assert rtol_infeas >= 0
    assert max_iter > 0
    assert time_limit_secs is None or time_limit_secs > 0
    assert 0 < step_size_scale < 1
    assert min_static_regularization >= 0
    assert max_iterative_refinement_steps >= 1
    assert linear_solver_atol >= 0
    assert linear_solver_rtol >= 0
    assert refinement_strategy in ("fixed_point", "fgmres")
    assert fgmres_restart >= 1
    assert equilibrate_per_iteration >= 0

    resolved_linear_solver, linear_solver_backend = _resolve_linear_solver(
        linear_solver
    )
    self.start_time = timeit.default_timer()
    self._deadline = (
        self.start_time + time_limit_secs if time_limit_secs is not None
        else None
    )
    self.atol, self.rtol = atol, rtol
    self.atol_infeas, self.rtol_infeas = atol_infeas, rtol_infeas
    self.verbose = verbose
    self.equilibrate = equilibrate
    self.initialization = initialization

    if verbose:
      print(
          f"| Clarabel (QTQP v{__version__}):"
          f" m={self.m}, n={self.n}, z={self.z}, nnz(A)={self.a.nnz},"
          f" nnz(P)={self.p.nnz},"
          f" linear_solver={resolved_linear_solver.name}"
      )

    if self.equilibrate:
      (a, p, b, c, self.d, self.e,
       self.primal_scale, self.dual_scale) = self._equilibrate(
          method=equilibration_method,
      )
    else:
      a, p, b, c = self.a, self.p, self.b, self.c
      self.d = self.e = self.primal_scale = self.dual_scale = None

    self._norm_b = _norm(self.b, np.inf)
    self._norm_c = _norm(self.c, np.inf)

    self._linear_solver = direct.DirectKktSolver(
        a=a,
        p=p,
        z=self.z,
        min_static_regularization=min_static_regularization,
        max_iterative_refinement_steps=max_iterative_refinement_steps,
        atol=linear_solver_atol,
        rtol=linear_solver_rtol,
        solver=linear_solver_backend,
        deadline=self._deadline,
        refinement_strategy=refinement_strategy,
        fgmres_restart=fgmres_restart,
        legacy_stall_check=legacy_stall_check,
        equilibrate_per_iteration=equilibrate_per_iteration,
    )

    # --- Initialization ---
    x, y, s, tau, _ = self._init_variables(a, p, b, c)
    kappa = 1.0

    constant_rhs = np.concatenate([-c, -b])
    # Reused as the warm-start across all per-iteration linear solves: never
    # mutated because DirectKktSolver.solve copies its warm_start internally.
    zero_warm_start = np.zeros(self.n + self.m)
    # Preallocated input buffer for the corrector slack RHS; its [:z] slice
    # stays zero for the lifetime of the solve and only [z:] is rewritten.
    ds_cor_in = np.zeros(self.m)

    stats: List[Dict[str, Any]] = []
    status = SolutionStatus.UNFINISHED
    self._log_header()

    for self.it in range(max_iter):
      stats_i: Dict[str, Any] = {}

      mu = (s @ y) / (self.m - self.z)

      # mu=0: regularization handled by min_static_regularization (Sec. 3.3)
      self._linear_solver.update(mu=0, s=s, y=y)

      # --- Solve constant system K @ Delta2 = [-c; -b] ---
      delta2, q_stats = self._linear_solver.solve(
          rhs=constant_rhs, warm_start=zero_warm_start,
      )
      dx2, dy2 = delta2[:self.n], delta2[self.n:]

      if self._time_limit_reached():
        status = SolutionStatus.HIT_TIME_LIMIT
        break

      rx, ry, r_tau = self._compute_residuals(
          x=x, y=y, s=s, tau=tau, kappa=kappa, a=a, b=b, p=p, c=c,
      )

      # --- Predictor (affine) step ---
      (dx_aff, dy_aff, ds_aff, dtau_aff, dkappa_aff,
       pred_stats) = self._clarabel_newton_step(
          x=x, y=y, s=s, tau=tau, kappa=kappa,
          p=p, b=b, c=c, dx2=dx2, dy2=dy2,
          dx=rx, dy=ry, ds=s, d_tau=r_tau, d_kappa=kappa * tau,
          warm_start=zero_warm_start,
      )

      if self._time_limit_reached():
        status = SolutionStatus.HIT_TIME_LIMIT
        break

      alpha_aff = self._compute_step_size(y, s, dy_aff, ds_aff)
      if dtau_aff < 0:
        alpha_aff = min(alpha_aff, tau / -dtau_aff)
      if dkappa_aff < 0:
        alpha_aff = min(alpha_aff, kappa / -dkappa_aff)
      sigma = (1.0 - alpha_aff) ** 3

      # --- Corrector step ---
      eta = ds_aff * dy_aff
      ds_cor_in[self.z:] = (
          s[self.z:] + (eta[self.z:] - sigma * mu) / y[self.z:]
      )

      (dx_cor, dy_cor, ds_cor, dtau_cor, dkappa_cor,
       cor_stats) = self._clarabel_newton_step(
          x=x, y=y, s=s, tau=tau, kappa=kappa,
          p=p, b=b, c=c, dx2=dx2, dy2=dy2,
          dx=(1.0 - sigma) * rx,
          dy=(1.0 - sigma) * ry,
          ds=ds_cor_in,
          d_tau=(1.0 - sigma) * r_tau,
          d_kappa=kappa * tau + dkappa_aff * dtau_aff - sigma * mu,
          warm_start=zero_warm_start,
      )

      alpha_cor = self._compute_step_size(y, s, dy_cor, ds_cor)
      if dtau_cor < 0:
        alpha_cor = min(alpha_cor, tau / -dtau_cor)
      if dkappa_cor < 0:
        alpha_cor = min(alpha_cor, kappa / -dkappa_cor)

      # --- Update iterates ---
      step = step_size_scale * alpha_cor
      x += step * dx_cor
      y += step * dy_cor
      s += step * ds_cor
      tau += step * dtau_cor
      kappa += step * dkappa_cor

      y[self.z:] = np.maximum(y[self.z:], 1e-30)
      s[self.z:] = np.maximum(s[self.z:], 1e-30)
      tau = max(tau, 1e-30)

      # --- Termination ---
      stats_i["q_lin_sys_stats"] = q_stats
      stats_i["predictor_lin_sys_stats"] = pred_stats
      stats_i["corrector_lin_sys_stats"] = cor_stats

      status = self._check_termination(
          x, y, tau, s, alpha_cor, mu, sigma, stats_i, collect_stats,
      )
      if status == SolutionStatus.UNFINISHED and self._time_limit_reached():
        status = SolutionStatus.HIT_TIME_LIMIT
        stats_i["status"] = status
      self._log_iteration(stats_i)
      if collect_stats:
        stats.append(stats_i)
      if status != SolutionStatus.UNFINISHED:
        break
    else:
      status = SolutionStatus.HIT_MAX_ITER
      if collect_stats and stats:
        stats[-1]["status"] = status

    # --- Result construction ---
    if self.equilibrate:
      x, y, s = self._unequilibrate_iterates(x, y, s)
    match status:
      case SolutionStatus.SOLVED:
        self._log_footer("Solved")
        x, y, s = x / tau, y / tau, s / tau
        y, s = self._postsolve(y, s, s_dropped=self._dropped_slack(x))
        return Solution(x, y, s, stats, status)
      case SolutionStatus.INFEASIBLE:
        self._log_footer("Primal infeasible / dual unbounded")
        x.fill(np.nan)
        s.fill(np.nan)
        y_scaled = y / abs(self.b @ y)
        y_scaled, s = self._postsolve(y_scaled, s)
        return Solution(x, y_scaled, s, stats, status)
      case SolutionStatus.UNBOUNDED:
        self._log_footer("Dual infeasible / primal unbounded")
        y.fill(np.nan)
        abs_ctx = abs(self.c @ x)
        x, s = x / abs_ctx, s / abs_ctx
        y, s = self._postsolve(y, s, y_dropped=np.nan)
        return Solution(x, y, s, stats, status)
      case SolutionStatus.HIT_MAX_ITER:
        self._log_footer("Hit maximum iterations")
        x, y, s = x / tau, y / tau, s / tau
        y, s = self._postsolve(y, s, s_dropped=self._dropped_slack(x))
        return Solution(x, y, s, stats, status)
      case SolutionStatus.HIT_TIME_LIMIT:
        self._log_footer("Hit time limit")
        x, y, s = x / tau, y / tau, s / tau
        y, s = self._postsolve(y, s, s_dropped=self._dropped_slack(x))
        return Solution(x, y, s, stats, status)
      case _:
        self._log_footer("Failed to converge")
        x, y, s = x / tau, y / tau, s / tau
        y, s = self._postsolve(y, s, s_dropped=self._dropped_slack(x))
        return Solution(x, y, s, stats, SolutionStatus.FAILED)

  @staticmethod
  def _compute_residuals(
      *, x, y, s, tau, kappa, a, b, p, c,
  ):
    """The non-linear operator G from Eq. (4)."""
    rx = -p @ x - a.T @ y - tau * c
    ry = s + a @ x - tau * b
    r_tau = kappa + c @ x + b @ y + (x @ (p @ x)) / tau
    return rx, ry, r_tau

  def _clarabel_newton_step(
      self,
      *, x, y, s, tau, kappa, p, b, c,
      dx2, dy2, dx, dy, ds, d_tau, d_kappa,
      warm_start,
  ):
    """Clarabel Newton step from Eq. (12).

    Decomposes the Newton system into a constant part (dx2, dy2 precomputed
    from [-c; -b]) and a variable part, then combines via the tau equation.
    """
    delta1, lin_sys_stats = self._linear_solver.solve(
        rhs=np.concatenate([dx, dy - ds]),
        warm_start=warm_start,
    )
    dx1, dy1 = delta1[:self.n], delta1[self.n:]

    xi = x / tau
    if p.nnz > 0:
      p_xi = p @ xi
      theta = 2.0 * p_xi + c
      xi_p_xi = xi @ p_xi
    else:
      theta = c
      xi_p_xi = 0.0

    dtau_num = d_tau - d_kappa / tau + theta @ dx1 + b @ dy1
    dtau_den = kappa / tau + xi_p_xi - theta @ dx2 - b @ dy2
    if abs(dtau_den) < 1e-16:
      logging.warning("Degenerate tau denominator; skipping tau/kappa update.")
      d_tau_step = 0.0
      d_kappa_step = 0.0
    else:
      d_tau_step = dtau_num / dtau_den
      d_kappa_step = -(d_kappa + kappa * d_tau_step) / tau

    d_x = dx1 + d_tau_step * dx2
    d_y = dy1 + d_tau_step * dy2
    d_s = np.zeros(self.m)
    d_s[self.z:] = -ds[self.z:] - s[self.z:] / y[self.z:] * d_y[self.z:]

    return d_x, d_y, d_s, d_tau_step, d_kappa_step, lin_sys_stats
