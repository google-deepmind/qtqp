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

"""Interior point method for solving QPs.

  Algorithm: Mehrotra predictor-corrector interior point method with a
  homogeneous embedding. Each iteration does:

    1. Normalize (x, y, tau, s) to have the norm of the central path.
    2. Pre-solve K^{-1} @ [c; b] (shared between predictor and corrector steps).
    3. Predictor step: Newton direction with mu_target=0 (no centering).
    4. Compute sigma (centering parameter) from predictor step quality.
    5. Corrector step: Newton direction with mu_target=sigma*mu and
       Mehrotra's second-order correction to improve complementarity.
    6. Update iterates and check termination.

  The embedding augments the problem with a homogeneous variable tau so the
  algorithm can detect primal/dual infeasibility without a separate Phase I.

"""

import dataclasses
import enum
import logging
import math
import sys
import timeit
from typing import Any, Dict, List

import numpy as np
import scipy.sparse as sp

from . import direct
from .direct import RefinementStrategy

__version__ = "0.0.5"
_HEADER = """| iter |      pcost |      dcost |     pres |     dres |      gap |   infeas |       mu |  q, p, c |     time |"""
_SEPARA = """|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|"""
_norm = np.linalg.norm
_EPS = 1e-15  # Standard epsilon for numerical safety
# ALMOST_SOLVED acceptance: on HIT_MAX_ITER or numerical breakdown, the
# best iterate seen is returned as ALMOST_SOLVED when it meets the same
# criteria form at tolerances this factor looser than the user-requested
# atol/rtol (at the 1e-9 defaults: 1e-6-grade, still a usable solution).
_ALMOST_FACTOR = 1000.0
# Floor on the complementarity parameter mu wherever it enters the
# algorithm (KKT shift, corrector targets, barrier terms). Below this
# scale mu carries no information in double precision relative to O(1)
# equilibrated data, and letting it underflow leaves the Newton system
# effectively unregularized: at the 1e-9 default tolerances the endgame
# reaches depths where an underflowed mu makes the KKT factorization
# effectively singular (observed: CHOLMOD NotPositiveDefinite on
# Windows). Healthy solves terminate at mu ~ 1e-9..1e-11 and never
# touch the floor.
_MU_FLOOR = 1e-14


class LinearSolver(enum.Enum):
  """Available linear solvers."""

  AUTO = "auto"
  ACCELERATE = direct.AccelerateSolver
  SCIPY = direct.ScipySolver
  SCIPY_DENSE = direct.ScipyDenseSolver
  CUPY_DENSE = direct.CupyDenseSolver
  UMFPACK = direct.UmfpackSolver
  PARDISO = direct.MklPardisoSolver
  QDLDL = direct.QdldlSolver
  CHOLMOD = direct.CholModSolver
  CUDSS = direct.CuDssSolver
  EIGEN = direct.EigenSolver
  MUMPS = direct.MumpsSolver


_AUTO_SOLVER_CACHE: dict[str, LinearSolver] = {}
_AUTO_UNAVAILABLE_ERRORS = (ImportError, OSError)


def _instantiate_linear_solver(linear_solver: LinearSolver) -> direct.LinearSolver:
  """Instantiate a concrete linear solver backend."""
  if linear_solver is LinearSolver.AUTO:
    raise ValueError("AUTO must be resolved before instantiating a backend.")
  return linear_solver.value()


def _auto_linear_solver_order() -> list[LinearSolver]:
  """Return AUTO backend candidates in priority order.

  The first platform-specific choice is intentional:
    * Linux / Windows -> PARDISO
    * macOS -> ACCELERATE

  The remaining sparse CPU fallbacks are shared across platforms and ordered
  from the current feasible-instance benchmark in the python312 environment.
  """
  fallbacks = [
      LinearSolver.CHOLMOD,
      LinearSolver.QDLDL,
      LinearSolver.EIGEN,
      LinearSolver.MUMPS,
      LinearSolver.UMFPACK,
      LinearSolver.SCIPY,
  ]

  if sys.platform == "darwin":
    return [LinearSolver.ACCELERATE] + fallbacks

  return [LinearSolver.PARDISO] + fallbacks


def _resolve_linear_solver(
    linear_solver: LinearSolver,
) -> tuple[LinearSolver, direct.LinearSolver]:
  """Resolve a requested solver enum to a concrete backend instance."""
  if linear_solver is not LinearSolver.AUTO:
    return linear_solver, _instantiate_linear_solver(linear_solver)

  cached = _AUTO_SOLVER_CACHE.get(sys.platform)
  if cached is not None:
    return cached, _instantiate_linear_solver(cached)

  for candidate in _auto_linear_solver_order():
    try:
      backend = _instantiate_linear_solver(candidate)
      _AUTO_SOLVER_CACHE[sys.platform] = candidate
      return candidate, backend
    except _AUTO_UNAVAILABLE_ERRORS as e:
      logging.debug("AUTO skipped %s: %s", candidate.name, e)

  raise RuntimeError("AUTO could not initialize any linear solver backend.")


class SolutionStatus(enum.Enum):
  """Possible statuses of the QP solution."""

  SOLVED = "solved"
  INFEASIBLE = "infeasible"
  UNBOUNDED = "unbounded"
  HIT_MAX_ITER = "hit_max_iter"
  ALMOST_SOLVED = "almost_solved"
  FAILED = "failed"
  UNFINISHED = "unfinished"


class EquilibrationStrategy(enum.Enum):
  """Available equilibration strategies applied to the problem data.

  NONE:
    Do not equilibrate. Pass (A, P, b, c) through unchanged.

  RUIZ:
    Ruiz equilibration on the constraint matrix A and Hessian P. Symmetric
    diagonal scalings D (rows) and E (columns) are chosen so that, at each
    iteration, the inf-norm of every row of D A E and every column of
    [D A E ; E P E] is driven toward 1. The vectors b and c are passively
    rescaled as b <- D b and c <- E c.

  AUGMENTED:
    Ruiz equilibration on the symmetric augmented matrix

        M = [ P    A^T   c ]
            [ A     0   -b ]
            [ c^T -b^T   0 ]

    so b and c participate in determining the row/column norms rather than
    being scaled passively. The scaling has three blocks (E for x columns,
    D for y rows, and a scalar sigma for the augmented row/column), giving

        A_eq = D A E,    P_eq = E P E,
        b_eq = sigma * (D b),   c_eq = sigma * (E c).

    The sigma factor maps to the homogenization variable (tau_orig =
    sigma * tau_eq), so iterate (un)equilibration must apply 1/sigma to
    keep the recovered x/tau, y/tau, s/tau in the original scale.
  """

  NONE = "none"
  RUIZ = "ruiz"
  AUGMENTED = "augmented"


class InitStrategy(enum.Enum):
  """Available initialization strategies for the IPM iterates.

  TRIVIAL:
    Unit vectors: y[z:] = s[z:] = 1, x = 0, tau = 1. Cheap, dimensionless.
    Default for backward compatibility.

  ORTHANT:
    Closed-form non-negative orthant centering. Picks mu_0 = mu_scale *
    ||b[z:]|| (see the init_mu_scale solve() kwarg) and solves the
    per-component centering condition
        mu_0 * y_i + b_i - mu_0 / y_i = 0
    for inequality rows, giving y_i = (-beta_i + sqrt(beta_i^2 + 4)) / 2 with
    beta = b/mu_0. Strictly interior by construction; cost is O(m).

  CVXOPT:
    Least-squares initialization (CVXOPT / Clarabel style). For QPs,
    solves the saddle-point system
        [P + eps*I,  A^T ] [x]   [-c]
        [    A,      -I  ] [y] = [ b]
    treating all rows as equalities; the (2, 2) block is -I, giving the
    bounded least-squares point y ~ Ax - b. For LPs (P with no
    nonzeros) the coupled solve is ill-posed, so it splits into two
    solves with the same factorization - [0; b] for the primal
    (feasibility only) and [-c; 0] for the dual (optimality only),
    matching Clarabel's LP initialization. The solves run through the
    session's DirectKktSolver (same backend, ordering, static
    regularization, and iterative refinement as the main loop). Then
    s = b - A @ x, and the inequality blocks of (y, s) are shifted by a
    uniform constant so min(y[z:]) and min(s[z:]) are at least 1.0
    (CVXOPT's standard interior-point shift). Default.
  """

  TRIVIAL = "trivial"
  ORTHANT = "orthant"
  CVXOPT = "cvxopt"


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


@dataclasses.dataclass(frozen=True)
class _PresolveState:
  """Data needed to restore rows dropped during presolve."""

  keep: np.ndarray
  a_dropped: sp.csc_matrix
  b_dropped: np.ndarray


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
    if not np.all(np.isfinite(a.data)):
      raise ValueError("Constraint matrix 'a' must contain only finite values.")
    if a.has_canonical_format:
      self.a = a
    else:
      # Non-canonical CSC (duplicate entries) is summed correctly by the
      # KKT assembly (sp.bmat goes through COO, which sums duplicates)
      # but not by the equilibration norms, which would see max(|d1|,
      # |d2|) where the true entry is |d1 + d2|. Canonicalize a copy.
      self.a = a.copy()
      self.a.sum_duplicates()

    self.b = np.array(b, dtype=np.float64)
    if self.b.shape != (self.m,):
      raise ValueError(f"b must have shape ({self.m},), got {self.b.shape}")

    self.c = np.array(c, dtype=np.float64)
    if self.c.shape != (self.n,):
      raise ValueError(f"c must have shape ({self.n},), got {self.c.shape}")
    if not np.all(np.isfinite(self.c)):
      raise ValueError("Cost vector 'c' must contain only finite values.")

    if self.z < 0 or self.z > self.m:
      raise ValueError(
          f"Number of equality constraints z={self.z} must satisfy "
          f"0 <= z <= m={self.m}"
      )

    self._presolve()
    if self.z == self.m:
      raise ValueError(
          "effective z == m after presolve; some rows may have been dropped"
      )

    if p is None:
      self.p = sp.csc_matrix((self.n, self.n))
    else:
      if not sp.isspmatrix_csc(p):
        raise TypeError("QP matrix 'p' must be in CSC format.")
      if not p.has_canonical_format:
        p = p.copy()
        p.sum_duplicates()
      if p.shape != (self.n, self.n):
        raise ValueError(
            f"p must have shape ({self.n}, {self.n}), got {p.shape}"
        )
      if not np.all(np.isfinite(p.data)):
        raise ValueError("QP matrix 'p' must contain only finite values.")
      asymmetry = p - p.T
      p_scale = max(1.0, np.max(np.abs(p.data), initial=0.0))
      if (
          asymmetry.nnz
          and np.max(np.abs(asymmetry.data), initial=0.0) > 1e-12 * p_scale
      ):
        raise ValueError("QP matrix 'p' must be symmetric.")
      self.p = p

    # Defaults so _check_termination works in tests that call it directly
    # before solve() has initialized the tracking state.
    self._best_almost_score = math.inf
    self._best_almost_iterate = None

  def _presolve(self, inf_bound: float = 1e20):
    """Drop inequality rows with trivially-satisfied RHS (b[i] >= inf_bound
    or +inf). Equality RHS must be finite; inequality RHS may not be NaN or -inf.
    """
    self._presolve_state = None
    if not np.all(np.isfinite(self.b[: self.z])):
      raise ValueError("Equality RHS entries in 'b' must be finite.")
    ineq_b = self.b[self.z :]
    if np.any(np.isnan(ineq_b) | np.isneginf(ineq_b)):
      raise ValueError(
          "Inequality RHS entries in 'b' must be finite, +inf, or >= inf_bound."
      )
    drop = np.zeros(self.m, dtype=bool)
    # Tolerate representation noise in the sentinel: benchmark files store
    # +-1e20 infinity markers with ULP- or float32-level error (e.g.
    # 9.999999999999998e19), which a strict >= comparison classifies as a
    # genuine finite bound -- materializing a 1e20-magnitude row that
    # silently poisons equilibration and residual scales.
    drop[self.z :] = ineq_b >= inf_bound * (1.0 - 1e-6)
    if not np.any(drop):
      return
    keep = ~drop
    self._presolve_state = _PresolveState(
        keep=keep, a_dropped=self.a[drop], b_dropped=self.b[drop],
    )
    self.a = self.a[keep]
    self.b = self.b[keep]
    self.m = int(keep.sum())

  def _postsolve(self, y, s, y_dropped=0.0, s_dropped=np.nan):
    """Restore full-sized (y, s) after presolve dropped rows.

    Kept entries are copied from (y, s); dropped entries take the values
    passed in y_dropped and s_dropped, each of which may be a scalar or
    an array of length (m_full - m_kept).
    """
    if self._presolve_state is None:
      return y, s
    ps = self._presolve_state
    m_full = ps.keep.shape[0]
    drop = ~ps.keep
    y_full = np.empty(m_full, dtype=y.dtype)
    s_full = np.empty(m_full, dtype=s.dtype)
    y_full[ps.keep] = y
    y_full[drop] = y_dropped
    s_full[ps.keep] = s
    s_full[drop] = s_dropped
    return y_full, s_full

  def _dropped_slack(self, x):
    """Slack b - A @ x for dropped rows; 0.0 (harmless scalar) if none."""
    ps = self._presolve_state
    if ps is None:
      return 0.0
    return ps.b_dropped - ps.a_dropped @ x

  def _lambda_local(self, x, y, s, tau, a, p, b, c):
    """Local-norm distance-to-path measure at a working-scale point.

    lambda = ||T_mu(u)||_{H^-1} with H = mu*I + mu*hess(Phi) (diagonal),
    evaluated at the point's own complementarity mu = y's/(m-z). Three
    matvecs. Certified-distance semantics when small; a scaled residual
    score when large. Returns inf when the point has no positive
    complementarity to define a barrier parameter at (e.g. junk warm
    points with sign-mixed y), which reads as maximally far from the path.
    """
    mu0 = float(y @ s) / (self.m - self.z)
    if not np.isfinite(mu0) or mu0 <= 0.0:
      return math.inf
    t_x0 = p @ x + a.T @ y + c * tau + mu0 * x
    t_y0 = -(a @ x) + b * tau + mu0 * y
    t_y0[self.z :] -= mu0 / y[self.z :]
    px0_ = float(x @ (p @ x)) if p.nnz else 0.0
    t_t0 = (-(float(c @ x) + float(b @ y)) - px0_ / max(_EPS, tau)
            + mu0 * (tau - 1.0 / max(_EPS, tau)))
    h_y0 = np.full(self.m, mu0)
    h_y0[self.z :] += mu0 / (y[self.z :] * y[self.z :])
    h_t0 = mu0 + mu0 / max(_EPS, tau * tau)
    return math.sqrt(
        float(t_x0 @ t_x0) / max(_EPS, mu0)
        + float((t_y0 * t_y0) @ (1.0 / h_y0))
        + t_t0 * t_t0 / max(_EPS, h_t0)
    )

  def _init_variables(self, strategy, mu_scale, a, p, b, c):
    """Dispatch to the requested initialization strategy.

    The (a, p, b, c) passed in are the operating-scale problem data: equilibrated
    if equilibration is on, original otherwise. The selected strategy produces
    iterates already in that scale, with the exception of TRIVIAL which keeps
    its historical behavior (build unit iterates and then equilibrate via
    _equilibrate_iterates) for backward compatibility.
    """
    if strategy is InitStrategy.TRIVIAL:
      return self._init_trivial()
    if strategy is InitStrategy.ORTHANT:
      return self._init_orthant(b, mu_scale)
    if strategy is InitStrategy.CVXOPT:
      return self._init_cvxopt(a, p, b, c)
    raise ValueError(f"Unknown init strategy: {strategy}")

  def _init_trivial(self):
    """Unit-vector init: y[z:] = s[z:] = 1, x = 0, tau = 1."""
    x = np.zeros(self.n)
    y = np.zeros(self.m)
    s = np.zeros(self.m)
    y[self.z :] = 1.0
    s[self.z :] = 1.0
    if self.equilibration_strategy is not EquilibrationStrategy.NONE:
      x, y, s = self._equilibrate_iterates(x, y, s)
    return x, y, s, 1.0, {}

  def _init_orthant(self, b, mu_scale):
    """Closed-form non-negative orthant init (see InitStrategy.ORTHANT)."""
    m, n, z = self.m, self.n, self.z
    x = np.zeros(n)
    y = np.zeros(m)
    s = np.zeros(m)

    b_ineq = b[z:]
    norm_b = _norm(b_ineq, 2)
    if norm_b == 0.0:
      mu_0 = max(mu_scale, np.finfo(np.float64).tiny)
      y[z:] = 1.0
      s[z:] = mu_0  # complementarity y_i * s_i = mu_0 with y_i = 1.
    else:
      mu_0 = mu_scale * norm_b
      beta = b_ineq / mu_0
      sq = np.sqrt(beta * beta + 4.0)
      # Branch-selected stable form: cancellation-free for both signs of beta.
      y_ineq = np.where(beta <= 0, 0.5 * (-beta + sq), 2.0 / (beta + sq))
      assert np.all(y_ineq > 0), "orthant init produced non-positive y component"
      y[z:] = y_ineq
      s[z:] = mu_0 / y_ineq

    return x, y, s, 1.0, {"mu_0": mu_0}

  def _init_cvxopt(self, a, p, b, c, reg=1e-8, interior_margin=1.0):
    """CVXOPT-style init: solve regularized saddle-point KKT, then shift."""
    m, n, z = self.m, self.n, self.z
    a_csc = a.tocsc() if not sp.isspmatrix_csc(a) else a
    # The (2,2) block is -I, not -reg*I: this is the standard
    # least-squares initialization (CVXOPT/Clarabel), giving y ~ Ax - b.
    # With -reg*I the block is near-singular and y is amplified by 1/reg
    # (measured: ||y|| ~ 3e10, mu_0 ~ 2e12 on netlib/25fv47).
    #
    # The solve runs through the session's DirectKktSolver - same backend,
    # symbolic ordering, static regularization, and iterative refinement
    # as every main-loop solve (a scipy.spsolve here cost +61% total
    # wall-clock on the kennington instances). The main loop's first
    # update() refactorizes afterwards as usual.
    if getattr(self, "_linear_solver", None) is not None:
      self._linear_solver.update_unit_dual(reg)

      def _saddle_solve(rx, ry):
        # DirectKktSolver.solve applies [P+reg*I, A'; A, -I] with the
        # second RHS block negated internally; pass (rx, -ry) so the
        # solved system is [P+reg*I, A'; A, -I] @ sol = (rx, ry).
        rhs = np.concatenate([rx, -ry])
        sol, _ = self._linear_solver.solve(rhs=rhs, warm_start=np.zeros_like(rhs))
        return sol

    else:
      # Standalone use (tests): assemble and solve directly.
      p_reg = (p + reg * sp.eye(n, format="csc")).tocsc()
      kkt = sp.bmat(
          [[p_reg, a_csc.T], [a_csc, -sp.eye(m, format="csc")]],
          format="csc",
      )

      def _saddle_solve(rx, ry):
        return sp.linalg.spsolve(kkt, np.concatenate([rx, ry]))

    if p.nnz == 0:
      # LP initialization (Clarabel's split): solve [0; b] for the primal
      # (feasibility only) and [-c; 0] for the dual (optimality only).
      # Coupling both right-hand sides into one solve, as the QP branch
      # does, is ill-posed when the (1,1) block is empty and produces
      # wildly scaled initial points (measured mu_0 up to 4e15 on netlib).
      xy_p = _saddle_solve(np.zeros(n), b)
      xy_d = _saddle_solve(-c, np.zeros(m))
      xy = np.concatenate([xy_p[:n], xy_d[n:]])
    else:
      xy = _saddle_solve(-c, b)
    if not np.all(np.isfinite(xy)):
      # Fall back to trivial init if the KKT solve produced non-finite values.
      logging.warning(
          "CVXOPT init KKT solve produced non-finite values; falling back to"
          " trivial init."
      )
      return self._init_trivial()
    x = xy[:n]
    y_full = xy[n:]
    s_full = b - a_csc @ x

    y = np.zeros(m)
    s = np.zeros(m)
    y[:z] = y_full[:z]  # Equality multipliers: any sign, s stays 0.
    if z < m:
      y_ineq = y_full[z:]
      s_ineq = s_full[z:]
      shift_y = interior_margin - np.min(y_ineq)
      if shift_y > 0:
        y_ineq = y_ineq + shift_y
      shift_s = interior_margin - np.min(s_ineq)
      if shift_s > 0:
        s_ineq = s_ineq + shift_s
      y[z:] = y_ineq
      s[z:] = s_ineq

    return x, y, s, 1.0, {}

  def solve(
      self,
      *,
      atol: float = 1e-9,
      rtol: float = 1e-9,
      atol_infeas: float = 1e-8,
      rtol_infeas: float = 1e-9,
      max_iter: int = 100,
      step_size_scale: float = 0.99,
      min_static_regularization: float = 1e-8,
      max_iterative_refinement_steps: int = 20,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.AUTO,
      verbose: bool = True,
      equilibration_strategy: EquilibrationStrategy = EquilibrationStrategy.RUIZ,
      collect_stats: bool = False,
      init_strategy: InitStrategy = InitStrategy.CVXOPT,
      init_mu_scale: float = 1.0,
      refinement_strategy: RefinementStrategy = RefinementStrategy.RICHARDSON,
      gmres_restart: int = 10,
      fused_corrector_division: bool = False,
      warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
      warm_start_threshold: float = 100.0,
  ) -> Solution:
    """Solves the QP using a primal-dual interior-point method."""
    self._linear_solver = None
    try:
      return self._solve_impl(
          atol=atol,
          rtol=rtol,
          atol_infeas=atol_infeas,
          rtol_infeas=rtol_infeas,
          max_iter=max_iter,
          step_size_scale=step_size_scale,
          min_static_regularization=min_static_regularization,
          max_iterative_refinement_steps=max_iterative_refinement_steps,
          linear_solver_atol=linear_solver_atol,
          linear_solver_rtol=linear_solver_rtol,
          linear_solver=linear_solver,
          verbose=verbose,
          equilibration_strategy=equilibration_strategy,
          collect_stats=collect_stats,
          init_strategy=init_strategy,
          init_mu_scale=init_mu_scale,
          refinement_strategy=refinement_strategy,
          gmres_restart=gmres_restart,
          fused_corrector_division=fused_corrector_division,
          warm_start=warm_start,
          warm_start_threshold=warm_start_threshold,
      )
    finally:
      if self._linear_solver is not None:
        self._linear_solver.free()
        self._linear_solver = None

  def _solve_impl(
      self,
      *,
      atol: float = 1e-9,
      rtol: float = 1e-9,
      atol_infeas: float = 1e-8,
      rtol_infeas: float = 1e-9,
      max_iter: int = 100,
      step_size_scale: float = 0.99,
      min_static_regularization: float = 1e-8,
      max_iterative_refinement_steps: int = 20,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.AUTO,
      verbose: bool = True,
      equilibration_strategy: EquilibrationStrategy = EquilibrationStrategy.RUIZ,
      collect_stats: bool = False,
      init_strategy: InitStrategy = InitStrategy.CVXOPT,
      init_mu_scale: float = 1.0,
      refinement_strategy: RefinementStrategy = RefinementStrategy.RICHARDSON,
      gmres_restart: int = 10,
      fused_corrector_division: bool = False,
      warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
      warm_start_threshold: float = 100.0,
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
      equilibration_strategy (EquilibrationStrategy): Which scaling to apply
        to (A, P, b, c) before iterating. See EquilibrationStrategy for
        descriptions. Defaults to RUIZ.
      collect_stats (bool): If True, collect per-iteration stats (sy, s_over_y
        statistics, complementarity, etc.) and return them in Solution.stats.
        Defaults to False for faster throughput; set True when per-iteration
        diagnostics are needed.
      init_strategy (InitStrategy): Which initialization to use for (x, y, s,
        tau). See InitStrategy for descriptions. Defaults to TRIVIAL.
      init_mu_scale (float): Multiplier on ||b[z:]|| that sets the initial
        barrier parameter mu_0 = init_mu_scale * ||b[z:]|| for
        InitStrategy.ORTHANT. Larger values produce iterates closer to the
        canonical center; smaller values produce more aggressive starts. Must
        be positive and finite. Ignored for other strategies.
      refinement_strategy (RefinementStrategy): Which iterative-refinement
        scheme drives each KKT solve. See RefinementStrategy for descriptions.
        Defaults to RICHARDSON.
      gmres_restart (int): Krylov dimension per GMRES restart cycle. Each
        inner Arnoldi step consumes one factor-solve. Smaller values reduce
        per-cycle cost at the price of more restarts. Ignored when
        refinement_strategy is RICHARDSON.
      fused_corrector_division (bool): If True, compute the corrector
        slack update via a single division by y[z:] with the three
        numerator terms (sigma*mu, the Mehrotra cross product, and
        y_c*s) assembled first. Algebraically equivalent to the default
        three-division formula `sigma*mu/y + correction - y_c*s/y`, but
        avoids catastrophic cancellation when y_i is small and the
        terms have similar magnitudes with opposing signs. Default False
        preserves the legacy bit-for-bit behavior.
      warm_start (tuple | None): Optional (x, y, s) from a nearby problem
        (original scale, e.g. a previous Solution's arrays). The point is
        equilibrated into the operating scale, embedded interior at a few
        centering shifts, and the best embedding is accepted only when the
        distance-to-path certificate measures lambda <=
        warm_start_threshold; otherwise the solve falls back to the
        configured init_strategy. After solve, `warm_lambda` (the measured
        lambda, or None when no warm start was given) and `warm_accepted`
        are available as attributes on the solver.
      warm_start_threshold (float): Acceptance threshold for the certified
        warm start. Default 100.0: poisoned or mis-scaled points measure
        orders of magnitude above it, same-problem re-entries orders of
        magnitude below.

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
    if not (np.isfinite(init_mu_scale) and init_mu_scale > 0):
      raise ValueError(
          f"init_mu_scale must be a positive finite float,"
          f" got {init_mu_scale}"
      )
    self._fused_corrector_division = bool(fused_corrector_division)

    resolved_linear_solver, linear_solver_backend = _resolve_linear_solver(
        linear_solver
    )
    self.start_time = timeit.default_timer()
    self.atol, self.rtol = atol, rtol
    self.atol_infeas, self.rtol_infeas = atol_infeas, rtol_infeas
    self.verbose = verbose
    self.equilibration_strategy = equilibration_strategy
    if verbose:
      print(
          f"| QTQP v{__version__}:"
          f" m={self.m}, n={self.n}, z={self.z}, nnz(A)={self.a.nnz},"
          f" nnz(P)={self.p.nnz}, linear_solver={resolved_linear_solver.name},"
          f" equilibration={equilibration_strategy.name}"
      )

    if equilibration_strategy is EquilibrationStrategy.NONE:
      a, p, b, c = self.a, self.p, self.b, self.c
      self.d, self.e, self.sigma_eq = None, None, 1.0
    else:
      a, p, b, c, self.d, self.e, self.sigma_eq = self._equilibrate()
    # Operating-scale b, c, kept for the distance-to-path certificate.
    self._b_op, self._c_op = b, c

    # q = [c; b]: the KKT right-hand side. The primal and dual feasibility
    # conditions at optimality can be written as: K @ [x; y] = -q * tau, where
    # K is the augmented KKT matrix, so the full Newton RHS has the form r - q *
    # tau+. Solving kinv_q = K^{-1} q once per iteration lets us write the
    # parametric solution as: [x+; y+] = kinv_r - kinv_q * tau+ and reuse kinv_q
    # in both the predictor and corrector steps.
    self.q = np.concatenate([c, b])

    # Precompute constant norms used in termination checks. _check_termination
    # unequilibrates iterates and compares against the original self.b / self.c,
    # so we use self.b and self.c here (not the equilibrated local b, c).
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
        refinement_strategy=refinement_strategy,
        gmres_restart=gmres_restart,
    )

    stats = []
    self.kinv_q = np.zeros_like(self.q)  # K^{-1}q, warm-started across iterations.
    x, y, s, tau, _ = self._init_variables(
        init_strategy, init_mu_scale, a, p, b, c
    )
    self._best_almost_score = math.inf
    self._best_almost_iterate = None
    status = SolutionStatus.UNFINISHED

    # Certified warm start: ingest a caller-supplied (x, y, s) from a
    # nearby problem, embed it interior at a few centering shifts, and
    # accept the best embedding only when the certificate says it is
    # actually near the path (lambda <= warm_start_threshold). A vetoed
    # warm start falls back to the configured init: the certificate makes
    # warm starting safe, which heuristic IPM warm starts are not.
    self.warm_lambda = None
    self.warm_accepted = False
    if warm_start is not None:
      if not (np.isfinite(warm_start_threshold) and warm_start_threshold > 0):
        raise ValueError(
            "warm_start_threshold must be a positive finite float, got"
            f" {warm_start_threshold}"
        )
      wx, wy, ws = (np.asarray(v, dtype=float).copy() for v in warm_start)
      if wx.shape != (self.n,) or wy.shape != (self.m,) or ws.shape != (self.m,):
        raise ValueError(
            "warm_start must be (x, y, s) with shapes"
            f" ({self.n},), ({self.m},), ({self.m},); got"
            f" {wx.shape}, {wy.shape}, {ws.shape}"
        )
      if self.equilibration_strategy is not EquilibrationStrategy.NONE:
        wx, wy, ws = self._equilibrate_iterates(wx, wy, ws)
      best = (math.inf, None)
      for eta in (1e-6, 1e-4, 1e-2, 1.0):
        cx = wx.copy()
        cy = wy.copy()
        cs = ws.copy()
        cy[self.z :] = np.maximum(cy[self.z :], eta)
        cs[self.z :] = np.maximum(cs[self.z :], eta)
        cs[: self.z] = 0.0
        ctau = 1.0
        cx, cy, ctau, cs = self._normalize(cx, cy, ctau, cs)
        lam = self._lambda_local(cx, cy, cs, ctau, a, p, b, c)
        if lam < best[0]:
          best = (lam, (cx, cy, cs, ctau))
      self.warm_lambda = best[0]
      if best[0] <= warm_start_threshold:
        cx, cy, cs, ctau = best[1]
        x, y, s, tau = cx, cy, cs, ctau
        self.warm_accepted = True
      logging.debug(
          "warm start: lambda = %e, accepted = %s",
          self.warm_lambda, self.warm_accepted,
      )

    # Initial-point diagnostic: the local-norm distance-to-path measure at
    # the starting iterate (warm or configured init), before the first
    # step. Healthy problems measure small (<= ~1e7 across NETLIB +
    # Maros-Meszaros); pathologically scaled data announces itself here
    # by tens of orders of magnitude — this diagnostic is how corrupted
    # infinity-sentinel bounds in the benchmark datasets were found.
    self.lambda_init = self._lambda_local(x, y, s, tau, a, p, b, c)

    self._log_header()

    # Pre-allocate [x; y] and d_s to avoid repeated allocation each iteration.
    xy = np.empty(self.n + self.m)   # Combined primal-dual vector [x; y]
    d_s = np.zeros(self.m)           # Slack step direction; d_s[:z] is always 0

    alpha = sigma = 0.0

    # --- Main Iteration Loop ---
    # Numeric failures inside the iteration (singular KKT systems,
    # NaN propagation, degenerate tau equations) are converted to a
    # FAILED status carrying the best iterate seen, per the documented
    # contract, instead of escaping as exceptions. Programming errors
    # (TypeError, NameError, ...) still propagate.
    try:
      # self.it counts IPM steps already taken.
      for self.it in range(max_iter):
        stats_i = {}
        if self.it == 0:
          stats_i["lambda_init"] = self.lambda_init
        x, y, tau, s = self._normalize(x, y, tau, s)

        mu = max((y @ s) / (self.m - self.z), _MU_FLOOR)

        # --- Take an IPM step ---
        self._linear_solver.update(mu=mu, s=s, y=y)

        # --- Step 1: Precompute kinv_q = K^{-1} @ q ---
        # This is reused for both predictor and corrector parts of the step.
        self.kinv_q, q_lin_sys_stats = self._linear_solver.solve(
            rhs=self.q, warm_start=self.kinv_q
        )
        stats_i["q_lin_sys_stats"] = q_lin_sys_stats

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
            x=x,
            y=y,
            s=s,
            tau=tau,
            correction=None,
        )
        stats_i["predictor_lin_sys_stats"] = predictor_lin_sys_stats

        d_x_p, d_y_p, d_tau_p = x_p - x, y_p - y, tau_p - tau
        # Predictor slack step from the linearized complementarity condition with
        # target=0: (y + d_y)(s + d_s) ≈ 0 => d_s = -(y + d_y)*s/y = -y_p*s/y.
        d_s[self.z :] = -y_p[self.z :] * s[self.z :] / y[self.z :]

        # Pre-compute the Mehrotra cross term in un-divided form only when the
        # fused-corrector path will consume it (otherwise stay on the legacy
        # `-d_s * d_y_p / y` formulation verbatim).
        if self._fused_corrector_division:
          cross_p = d_s[self.z :] * d_y_p[self.z :]

        # Compute predictor step size and resulting centering parameter (sigma)
        alpha_p = self._compute_step_size(y, s, d_y_p, d_s)
        sigma = self._compute_sigma(
            mu, x, y, tau, s, alpha_p, d_x_p, d_y_p, d_tau_p, d_s
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
        # so the corrector step can incorporate it to land closer to the target.
        if self._fused_corrector_division:
          correction = -cross_p / y[self.z :]
        else:
          correction = -d_s[self.z :] * d_y_p[self.z :] / y[self.z :]
        xy[: self.n] = x_p
        xy[self.n :] = y_p
        x_c, y_c, tau_c, corrector_lin_sys_stats = self._newton_step(
            p=p,
            mu=mu,
            mu_target=sigma * mu,
            r_anchor=xy,
            tau_anchor=tau_p,
            x=x,
            y=y,
            s=s,
            tau=tau,
            correction=correction,
        )
        stats_i["corrector_lin_sys_stats"] = corrector_lin_sys_stats

        # --- Step 4: Update Iterates ---
        d_x, d_y, d_tau = x_c - x, y_c - y, tau_c - tau
        if self._fused_corrector_division:
          # Combined-numerator corrector slack step. Algebraically equivalent
          # to `sigma*mu/y + correction - y_c*s/y`, but assembles the
          # numerator before the single division by y[z:], avoiding
          # catastrophic cancellation when y_i is small and the three terms
          # have similar magnitudes with opposing signs.
          d_s[self.z :] = (
              sigma * mu - cross_p - y_c[self.z :] * s[self.z :]
          ) / y[self.z :]
        else:
          # Legacy three-division formula.
          d_s[self.z :] = (
              sigma * mu / y[self.z :]
              + correction
              - y_c[self.z :] * s[self.z :] / y[self.z :]
          )

        alpha = self._compute_step_size(y, s, d_y, d_s)
        step = step_size_scale * alpha
        x += step * d_x
        y += step * d_y
        tau += step * d_tau
        s += step * d_s

        # Ensure variables stay strictly in the cone to prevent numerical issues.
        y[self.z :] = np.maximum(y[self.z :], 1e-30)
        s[self.z :] = np.maximum(s[self.z :], 1e-30)
        tau = max(tau, 1e-30)

        status = self._check_termination(
            x, y, tau, s, alpha, mu, sigma, stats_i, collect_stats
        )
        self._log_iteration(stats_i)
        if collect_stats:
          stats.append(stats_i)
        if status != SolutionStatus.UNFINISHED:
          break
      else:
        status = SolutionStatus.HIT_MAX_ITER
        if collect_stats:
          stats[-1]["status"] = status

    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
      logging.warning("Numeric failure at iteration %d: %s", self.it, exc)
      status = SolutionStatus.UNFINISHED
      if collect_stats and stats:
        stats[-1]["status"] = SolutionStatus.FAILED

    # We have terminated for one reason or another.
    if self.equilibration_strategy is not EquilibrationStrategy.NONE:
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
      case SolutionStatus.HIT_MAX_ITER | SolutionStatus.UNFINISHED:
        # Salvage: the best iterate seen, if it meets the criteria at
        # _ALMOST_FACTOR looser tolerances, is an honestly-labeled
        # near-solution - more useful than the raw final iterate after a
        # cap-out or a numerical breakdown. The stored iterate is already
        # unequilibrated (captured inside _check_termination).
        if (
            self._best_almost_iterate is not None
            and self._best_almost_score < 1.0
        ):
          bx, by, bs, btau = self._best_almost_iterate
          self._log_footer("Almost solved (best iterate salvage)")
          bx, by, bs = bx / btau, by / btau, bs / btau
          by, bs = self._postsolve(by, bs, s_dropped=self._dropped_slack(bx))
          return Solution(bx, by, bs, stats, SolutionStatus.ALMOST_SOLVED)
        if status is SolutionStatus.HIT_MAX_ITER:
          self._log_footer("Hit maximum iterations")
          final = status
        else:
          self._log_footer("Failed to converge")
          final = SolutionStatus.FAILED
        x, y, s = x / tau, y / tau, s / tau
        y, s = self._postsolve(y, s, s_dropped=self._dropped_slack(x))
        return Solution(x, y, s, stats, final)
      case _:
        raise ValueError(f"Unknown convergence status: {status}")

  def _equilibrate(self, num_iters=10, min_scale=1e-3, max_scale=1e3):
    """Dispatch to the selected equilibration strategy.

    Returns a 7-tuple (a, p, b, c, d, e, sigma) of equilibrated problem data
    and accumulated scalings. For RUIZ, sigma == 1.0; for AUGMENTED, sigma is
    the scalar scaling on the augmented row/column (== tau_orig / tau_eq).
    """
    if self.equilibration_strategy is EquilibrationStrategy.RUIZ:
      return self._equilibrate_ruiz(num_iters, min_scale, max_scale)
    if self.equilibration_strategy is EquilibrationStrategy.AUGMENTED:
      return self._equilibrate_augmented(num_iters, min_scale, max_scale)
    raise ValueError(
        f"Unknown equilibration strategy: {self.equilibration_strategy}"
    )

  def _equilibrate_ruiz(self, num_iters, min_scale, max_scale):
    """Ruiz equilibration on A and P. b, c rescaled passively by d, e."""
    # Work on copies so self.a / self.p are not modified in-place; they are
    # used unequilibrated later (e.g. in _check_termination). The constructor
    # already enforces CSC, so .copy() preserves format without a re-convert.
    a, p = self.a.copy(), self.p.copy()
    b, c = self.b, self.c
    # Initialize the equilibration matrices.
    d, e = (np.ones(self.m), np.ones(self.n))

    # Sparsity patterns are static across iterations; the per-column nnz
    # counts only need to be computed once for the scaling-broadcast step.
    a_col_counts = np.diff(a.indptr)
    p_col_counts = np.diff(p.indptr) if p.nnz > 0 else None

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
      # Equivalent to (for CSC matrices):
      #     d_mat, e_mat = sp.diags(d_i), sp.diags(e_i)
      #     a = d_mat @ a @ e_mat
      #     p = e_mat @ p @ e_mat
      col_scale_a = np.repeat(e_i, a_col_counts)
      a.data *= d_i[a.indices] * col_scale_a
      if p_col_counts is not None:
        # E @ P @ E: scale non-zero at row r, col c by e_i[r] * e_i[c].
        col_scale_p = np.repeat(e_i, p_col_counts)
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

    return a, p, b * d, c * e, d, e, 1.0

  def _equilibrate_augmented(self, num_iters, min_scale, max_scale):
    """Ruiz equilibration on the symmetric augmented matrix.

        M = [ P    A^T   c ]
            [ A     0   -b ]
            [ c^T -b^T   0 ]

    The augmented row/column for the homogenization variable introduces a
    scalar scaling sigma in addition to the row scaling d and column scaling
    e, so b and c are rescaled by sigma * (d ⊙ b) and sigma * (e ⊙ c).
    """
    # Constructor enforces CSC; .copy() preserves format without a re-convert.
    a, p = self.a.copy(), self.p.copy()
    b, c = self.b.copy(), self.c.copy()
    d, e = np.ones(self.m), np.ones(self.n)
    sigma = 1.0

    # Sparsity patterns are static; pre-compute the per-column nnz counts.
    a_col_counts = np.diff(a.indptr)
    p_col_counts = np.diff(p.indptr) if p.nnz > 0 else None

    for i in range(num_iters):
      # Column inf-norms of the symmetric augmented matrix.
      # x-columns (0..n): max(||P[:,j]||_inf, ||A[:,j]||_inf, |c[j]|)
      norms_e = np.maximum(
          sp.linalg.norm(p, np.inf, axis=0),
          sp.linalg.norm(a, np.inf, axis=0),
      )
      norms_e = np.maximum(norms_e, np.abs(c))
      # y-columns (n..n+m): max(||A[i,:]||_inf, |b[i]|)
      norms_d = np.maximum(sp.linalg.norm(a, np.inf, axis=1), np.abs(b))
      # tau-column (n+m): max(||c||_inf, ||b||_inf)
      norm_sigma = max(_norm(c, np.inf), _norm(b, np.inf))

      e_i = 1.0 / np.sqrt(np.where(norms_e == 0.0, 1.0, norms_e))
      d_i = 1.0 / np.sqrt(np.where(norms_d == 0.0, 1.0, norms_d))
      sigma_i = 1.0 / math.sqrt(norm_sigma) if norm_sigma > 0.0 else 1.0

      e_i = np.clip(e_i, min_scale, max_scale)
      d_i = np.clip(d_i, min_scale, max_scale)
      sigma_i = float(np.clip(sigma_i, min_scale, max_scale))

      # A: D_i A E_i (same in-place CSC scaling as RUIZ).
      col_scale_a = np.repeat(e_i, a_col_counts)
      a.data *= d_i[a.indices] * col_scale_a
      if p_col_counts is not None:
        col_scale_p = np.repeat(e_i, p_col_counts)
        p.data *= e_i[p.indices] * col_scale_p
      # b, c absorb the tau-column scaling sigma_i in addition to d_i / e_i.
      b = sigma_i * d_i * b
      c = sigma_i * e_i * c

      d *= d_i
      e *= e_i
      sigma *= sigma_i
      logging.debug(
          "Augmented equilibration iter %d: d_i err: %s, e_i err: %s,"
          " sigma_i err: %s",
          i,
          _norm(d_i - 1, np.inf),
          _norm(e_i - 1, np.inf),
          abs(sigma_i - 1.0),
      )

    return a, p, b, c, d, e, sigma

  def _unequilibrate_iterates(self, x, y, s):
    """Map equilibrated iterates back to original-problem scale.

    Bakes the 1/sigma factor into (x, y, s) so the subsequent division by
    the (equilibrated-space) tau produces the original-problem solution.
    """
    inv_sigma = 1.0 / self.sigma_eq
    return (
        inv_sigma * self.e * x,
        inv_sigma * self.d * y,
        inv_sigma * s / self.d,
    )

  def _equilibrate_iterates(self, x, y, s):
    """Inverse of _unequilibrate_iterates: original scale -> equilibrated."""
    return (
        self.sigma_eq * x / self.e,
        self.sigma_eq * y / self.d,
        self.sigma_eq * s * self.d,
    )

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
    xyt_norm_sq = x_aff @ x_aff + y_aff @ y_aff + tau_aff * tau_aff
    scale_sq = (self.m - self.z + 1) / max(_EPS * _EPS, xyt_norm_sq)
    mu_aff = scale_sq * (y_aff @ s_aff) / (self.m - self.z)

    # sigma = (mu_aff / mu)^3: Mehrotra's heuristic. If the affine step already
    # drives mu close to zero, sigma is small (aggressive, little centering).
    # If mu_aff ≈ mu (affine step didn't help much), sigma ≈ 1 (full centering).
    # The cubic exponent amplifies the contrast, pushing sigma toward 0 or 1.
    sigma_base = mu_aff / max(_EPS, mu_curr)
    sigma = sigma_base * sigma_base * sigma_base  # More stable than **3.
    return np.clip(sigma, 0.0, 1.0)

  def _newton_step(
      self, *, p, mu, mu_target, r_anchor, tau_anchor, x, y, s, tau, correction,
  ):
    """Computes a Newton search direction by solving the augmented KKT system.

    The KKT system K @ [x+; y+] = r - q * tau+ is linear in tau+, giving the
    parametric solution:
        [x+; y+] = K^{-1}(r) - K^{-1}(q) * tau+  =  kinv_r - kinv_q * tau+
    tau+ is then pinned by substituting this back into the tau equation of the
    homogeneous embedding (see _solve_for_tau).

    The central-path equation r + mu^p * u = 0 contributes the
    (mu - mu_target) coefficient on the linear-residual side; cone-product
    corrections (s_i * y_i = mu_target, tau * kappa = mu_target) keep the
    unmodified mu_target.

    Uses the exact quadratic tau solve when the KKT solve is accurate, and a
    linearized fallback (avoids squaring solver noise) when it's noisy or the
    quadratic residual check fails.
    """
    # Prepare RHS for the linear system.
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

    # Tau solve: always attempt the exact quadratic; fall back to the
    # linearized form only when it fails. The former residual pre-check
    # (skip the quadratic unless converged or residual < 1e-7) was
    # measured to be harmful on the one instance where it fired
    # materially: under a deterministic backend, d6cube flips from
    # HIT_MAX_ITER to SOLVED when the quadratic is always attempted,
    # and no instance in NETLIB+Maros-Meszaros degrades. The exception
    # path fires twice across both suites, both benign.
    tau_plus = None
    try:
      r_tau = (mu - mu_target) * tau_anchor
      tau_plus = self._solve_for_tau(p, kinv_r, mu, mu_target, r_tau)
      lin_sys_stats["tau_method"] = "quadratic"
    except ValueError:
      logging.debug("Primary tau solve failed; falling back to linearized.")

    if tau_plus is None:
      lin_sys_stats["tau_method"] = "linearized"
      logging.debug("Using linearized tau fallback.")
      tau_plus = self._solve_for_tau_linearized_fallback(
          p, kinv_r, mu, mu_target, x, y, tau, tau_anchor
      )

    # Reconstruct [x+; y+] = kinv_r - kinv_q * tau+ (in-place on kinv_r).
    kinv_r -= self.kinv_q * tau_plus
    x_plus, y_plus = kinv_r[: self.n], kinv_r[self.n :]
    return x_plus, y_plus, tau_plus, lin_sys_stats

  def _solve_for_tau(self, p, kinv_r, mu, mu_target, r_tau) -> float:
    """Solves for tau+ using the homogeneous embedding's tau equation.

    The parametric KKT solution is:
        [x+; y+] = kinv_r - kinv_q * tau+

    Substituting this into the tau equation of the homogeneous embedding yields:
        t_a * tau+^2 + t_b * tau+ + t_c = 0

    The coefficients t_a, t_b, t_c are computed from inner products of kinv_r
    and kinv_q with q and P. For LPs (P=0) the P terms drop out. We always take
    the positive root since tau >= 0 is required for the embedding to represent
    a feasible point (tau=0 corresponds to a certificate of infeasibility or
    unboundedness, which is handled separately at termination).

    t_c = -mu_target keeps the unmodified mu_target since it comes from the
    cone-product equation tau * kappa = mu_target.
    """
    # Coefficients of the quadratic t_a * tau+^2 + t_b * tau+ + t_c = 0.
    n = self.n
    q, kinv_q = self.q, self.kinv_q

    t_a = mu + kinv_q @ q
    t_b = -r_tau - kinv_r @ q
    t_c = -mu_target
    if p.nnz > 0:
      # Memory access for the sparse matrix P is the bottleneck here.
      # np.stack enables a single pass over P's data and indices, which
      # is ~25% faster than two separate SpMVs (p @ kinv_r and p @ kinv_q).
      p_kinv_r, p_kinv_q = (p @ np.stack([kinv_r[:n], kinv_q[:n]], axis=1)).T
      t_a -= kinv_q[:n] @ p_kinv_q
      t_b += kinv_r[:n] @ p_kinv_q + kinv_q[:n] @ p_kinv_r
      t_c -= kinv_r[:n] @ p_kinv_r
    logging.debug("t_a=%s, t_b=%s, t_c=%s", t_a, t_b, t_c)

    if abs(t_a) < _EPS:
      if abs(t_b) < _EPS:
        raise ValueError(
            f"Degenerate tau equation: t_a={t_a}, t_b={t_b}, t_c={t_c}"
        )
      tau_sol = -t_c / t_b
      if not np.isfinite(tau_sol) or tau_sol < -1e-10:
        raise ValueError(f"Invalid linear tau solution found: {tau_sol}")
      return max(0.0, tau_sol)

    discriminant = t_b * t_b - 4 * t_a * t_c
    if discriminant < -1e-9:
      raise ValueError(f"Negative discriminant: {discriminant}")
    discriminant = max(0.0, discriminant)

    # Stable Quadratic Formula (Muller)
    if t_b > 0:
      q_muller = -0.5 * (t_b + math.sqrt(discriminant))
      tau_sol = t_c / q_muller
    else:
      q_muller = -0.5 * (t_b - math.sqrt(discriminant))
      tau_sol = q_muller / t_a

    if not np.isfinite(tau_sol) or tau_sol < -1e-10:
      raise ValueError(f"Invalid tau solution found: {tau_sol}")

    return max(0.0, tau_sol)

  def _solve_for_tau_linearized_fallback(
      self, p, kinv_r, mu, mu_target, x, y, tau_curr, tau_anchor,
  ) -> float:
    """Linearized fallback for tau via first-order Taylor expansion of G(z,tau).

    Replaces the exact quadratic with a linearization around z_curr = [x; y]
    and tau_curr. P only multiplies the safe current iterate x, so KKT noise
    enters linearly rather than quadratically. A [0.1x, 10x] trust region
    prevents manifold drift from the first-order approximation.

    Linear-residual coefficients on tau use mu and mu_target; the
    cone-product constant -mu_target keeps the unmodified mu_target.
    """
    n = self.n
    q, kinv_q = self.q, self.kinv_q

    px = p @ x if p.nnz > 0 else np.zeros(n)

    # Scalar inner products; avoids allocating z_curr = [x; y] or r_z.
    q_z = q[:n] @ x + q[n:] @ y
    x_px = x @ px
    q_kinv_q = q @ kinv_q
    px_kinv_q = px @ kinv_q[:n]
    # r_z = kinv_r - tau_curr * kinv_q - z_curr, collapsed into scalar dots.
    q_rz = q @ kinv_r - tau_curr * q_kinv_q - q_z
    px_rz = px @ kinv_r[:n] - tau_curr * px_kinv_q - x_px

    # Base residual G(z_curr, tau_curr).
    g = (mu * tau_curr * tau_curr
         + (mu_target - mu) * tau_anchor * tau_curr
         - tau_curr * q_z - mu_target - x_px)

    # Numerator: G + (dG/dz) @ r_z.  Denominator: dG/dtau - (dG/dz) @ kinv_q.
    num = g - tau_curr * q_rz - 2.0 * px_rz
    den = (2.0 * mu * tau_curr + (mu_target - mu) * tau_anchor - q_z +
           tau_curr * q_kinv_q + 2.0 * px_kinv_q)

    tau_sol = tau_curr + (0.0 if abs(den) < 1e-16 else -num / den)

    if not np.isfinite(tau_sol):
      logging.warning("Linearized tau fallback non-finite; using current tau.")
      return tau_curr
    return min(max(tau_sol, 0.1 * tau_curr), 10.0 * tau_curr)

  def _normalize(self, x, y, tau, s):
    """Normalizes iterates to match the homogeneous embedding central path norm.

    The homogeneous embedding lifts the QP into a projective space. Only ratios
    like x/tau and y/tau matter — tau is the homogeneous variable, and the final
    solution is recovered as (x/tau, y/tau, s/tau).

    We enforce the norm of the central path, which ensures convergence to
    non-trivial solution, ie:
        ||(x, y, tau)||^2 = m - z + 1
    The right-hand side counts complementarity pairs: (m - z) from the
    inequality constraints plus 1 for the tau-kappa pair of the embedding.

    Operates in-place on the iterate arrays and returns them for convenience.
    """
    xyt_norm = math.sqrt(x @ x + y @ y + tau * tau)
    scale = math.sqrt(self.m - self.z + 1) / max(_EPS, xyt_norm)
    x *= scale
    y *= scale
    tau *= scale
    s *= scale
    return x, y, tau, s

  def _compute_step_size(self, y, s, d_y, d_s) -> float:
    """Computes the maximum standard primal-dual step size."""
    alpha_s = self._max_step_size(s[self.z :], d_s[self.z :])
    alpha_y = self._max_step_size(y[self.z :], d_y[self.z :])
    return min(alpha_s, alpha_y)

  def _check_termination(self, x, y, tau, s, alpha, mu, sigma, stats_i, collect_stats):
    """Check termination criteria and compute iteration statistics."""
    # Working-scale references (equilibrated when equilibration is on),
    # kept for the distance-to-path certificate below; mu_hat is the
    # complementarity of THIS iterate in the operating scale.
    x_w, y_w, s_w = x, y, s
    mu_hat = float(y_w @ s_w) / (self.m - self.z)

    if self.equilibration_strategy is not EquilibrationStrategy.NONE:
      x, y, s = self._unequilibrate_iterates(x, y, s)

    inv_tau = 1.0 / max(tau, _EPS)

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

    # Residuals. ax_plus_s and px_plus_aty are reused for the infeasibility
    # certificates below, so compute them once.
    ax_plus_s = ax + s
    px_plus_aty = px + aty
    pres = _norm(ax_plus_s * inv_tau - self.b, np.inf)
    dres = _norm(px_plus_aty * inv_tau + self.c, np.inf)
    gap = abs((ctx + bty + xpx * inv_tau) * inv_tau)

    # A posteriori distance-to-path certificate. The regularized path map
    # T_mu is mu-strongly monotone, so
    #     ||u - u*(mu)|| <= ||T_mu(u)|| / mu  =: delta_path,
    # computable at every iterate. Basically free: the three SpMVs above
    # are reused via diagonal rescaling into the operating scale. The
    # bound saturates at the floating-point floor once mu approaches
    # roundoff of the summands (late endgame); treat large-mu iterates as
    # the informative regime.
    if self.equilibration_strategy is not EquilibrationStrategy.NONE:
      se = self.sigma_eq * self.e
      sd = self.sigma_eq * self.d
      sig2 = self.sigma_eq * self.sigma_eq
      ax_w = sd * ax
      aty_w = se * aty
      px_w = se * px
      ctx_w, bty_w, xpx_w = sig2 * ctx, sig2 * bty, sig2 * xpx
    else:
      ax_w, aty_w, px_w = ax, aty, px
      ctx_w, bty_w, xpx_w = ctx, bty, xpx
    b_op = getattr(self, "_b_op", self.b)
    c_op = getattr(self, "_c_op", self.c)
    t_x = px_w + aty_w + c_op * tau
    t_x += mu_hat * x_w
    t_y = -ax_w + b_op * tau
    t_y += mu_hat * y_w
    t_y[self.z :] -= mu_hat / y_w[self.z :]
    inv_tau_w = 1.0 / max(tau, _EPS)
    t_tau = (-(ctx_w + bty_w) - xpx_w * inv_tau_w
             + mu_hat * (tau - inv_tau_w))
    t_norm = math.sqrt(
        float(t_x @ t_x) + float(t_y @ t_y) + t_tau * t_tau
    )
    stats_i["delta_path"] = t_norm / max(_EPS, mu_hat)
    # Local-norm proximity measure (diagnostic): the same T vector weighted
    # by the inverse of H = mu*I + mu*diag(hess barrier), the barrier's
    # local metric. Deflates the barrier rows that make the Euclidean
    # certificate conservative on aggressively centered iterates
    # (Newton-decrement flavor), and remains informative for aggressive
    # iterates where the Euclidean bound saturates.
    h_x = mu_hat
    h_y = np.full(self.m, mu_hat)
    h_y[self.z :] += mu_hat / (y_w[self.z :] * y_w[self.z :])
    h_tau = mu_hat + mu_hat / max(_EPS, tau * tau)
    lam_sq = (
        float(t_x @ t_x) / max(_EPS, h_x)
        + float((t_y * t_y) @ (1.0 / h_y))
        + t_tau * t_tau / max(_EPS, h_tau)
    )
    stats_i["delta_path_local"] = math.sqrt(max(0.0, lam_sq))

    # Infeasibility certificates (Farkas-type, from the embedding structure).
    # If the primal is unbounded (dual infeasible) this produces a ray x with
    # c'x < 0 that satisfies the homogeneous primal conditions Ax + s ≈ 0 and Px
    # ≈ 0.  dinfeas measures how well x/|c'x| certifies dual infeasibility.
    norm_aty = _norm(aty, np.inf)
    norm_px = _norm(px, np.inf)
    dinfeas_a = _norm(ax_plus_s, np.inf) / (abs(ctx) + _EPS)
    dinfeas_p = norm_px / (abs(ctx) + _EPS)
    dinfeas = max(dinfeas_a, dinfeas_p)
    # If the primal is infeasible (dual unbounded) this produces a ray y
    # with b'y < 0 that satisfies the homogeneous dual condition A'y ≈ 0.
    # pinfeas measures how well y/|b'y| certifies primal infeasibility.
    pinfeas = norm_aty / (abs(bty) + _EPS)

    norm_x = _norm(x, np.inf)
    norm_y = _norm(y, np.inf)

    # Residual tolerance relative scales. Each is the max of two families:
    # the summand norms (the floor below which the residual cannot even be
    # evaluated in floating point) and the iterate norm (the backward-error
    # allowance: dres <= rtol * ||x||/tau accepts a point that is exactly
    # dual-feasible for P + dP with ||dP|| <= rtol, which is precisely the
    # mu*I perturbation the regularized path itself commits; likewise
    # pres <= rtol * ||y||/tau on the primal side).
    prelrhs = max(
        _norm(ax, np.inf) * inv_tau,
        _norm(s, np.inf) * inv_tau,
        self._norm_b,
        norm_y * inv_tau,
    )

    drelrhs = max(
        norm_px * inv_tau,
        norm_aty * inv_tau,
        self._norm_c,
        norm_x * inv_tau,
    )

    # Gap tolerance relative scale: the cost magnitudes (measurement floor)
    # or the first-power iterate norm (the empirically calibrated allowance
    # for the regularized path's O(mu*||u||^2) objective bias; see the
    # criteria validation on NETLIB + Maros-Meszaros).
    gaprelrhs = max(
        min(abs(pcost), abs(dcost)),
        (norm_x + norm_y) * inv_tau,
    )

    # Solved: duality gap and both residuals are within tolerance.
    if (
        gap < self.atol + self.rtol * gaprelrhs
        and pres < self.atol + self.rtol * prelrhs
        and dres < self.atol + self.rtol * drelrhs
    ):
      status = SolutionStatus.SOLVED
    # Unbounded: x is a dual infeasibility certificate (primal unbounded ray).
    elif ctx < -_EPS and (
        dinfeas < self.atol_infeas + self.rtol_infeas * norm_x / abs(ctx)
    ):
      status = SolutionStatus.UNBOUNDED
    # Infeasible: y is a primal infeasibility certificate (dual unbounded ray).
    elif bty < -_EPS and (
        pinfeas < self.atol_infeas + self.rtol_infeas * norm_y / abs(bty)
    ):
      status = SolutionStatus.INFEASIBLE
    else:
      status = SolutionStatus.UNFINISHED

    # Track the best iterate seen, scored by the max normalized residual
    # at the ALMOST_SOLVED thresholds (same criteria form, looser
    # constants). Used to return an honestly-labeled near-solution when
    # the iteration cap is reached without meeting the SOLVED contract.
    almost_atol = _ALMOST_FACTOR * self.atol
    almost_rtol = _ALMOST_FACTOR * self.rtol
    almost_score = max(
        gap / (almost_atol + almost_rtol * gaprelrhs),
        pres / (almost_atol + almost_rtol * prelrhs),
        dres / (almost_atol + almost_rtol * drelrhs),
    )
    if almost_score < self._best_almost_score:
      self._best_almost_score = almost_score
      self._best_almost_iterate = (x.copy(), y.copy(), s.copy(), tau)

    stats_i.update({
        "iter": self.it,
        "ctx": ctx,
        "bty": bty,
        "pcost": pcost,
        "dcost": dcost,
        "pres": pres,
        "dres": dres,
        "gap": gap,
        "pinfeas": pinfeas,
        "dinfeas": dinfeas,
        "dinfeas_a": dinfeas_a,
        "dinfeas_p": dinfeas_p,
        "mu": mu,
        "sigma": sigma,
        "alpha": alpha,
        "tau": tau,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "status": status,
        "time": timeit.default_timer() - self.start_time,
        "prelrhs": prelrhs,
        "drelrhs": drelrhs,
    })

    if collect_stats:
      stats_i["complementarity"] = abs((y @ s) * inv_tau * inv_tau)
      stats_i["norm_s"] = _norm(s, np.inf)
      # Per-inequality stats only meaningful when inequalities exist.
      if self.z < self.m:
        sy = s[self.z :] * y[self.z :]
        s_over_y = s[self.z :] / np.maximum(_EPS, y[self.z :])
        stats_i.update({
            "max_sy": np.max(sy),
            "min_sy": np.min(sy),
            "std_sy": np.std(sy),
            "max_s_over_y": np.max(s_over_y),
            "min_s_over_y": np.min(s_over_y),
            "mean_s_over_y": np.mean(s_over_y),
            "std_s_over_y": np.std(s_over_y),
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
