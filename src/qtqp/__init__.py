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
from typing import Any, Dict, List, Literal

import numpy as np
import scipy.sparse as sp

from . import direct


EquilibrationMethod = Literal["ruiz", "curtis_reid"]

__version__ = "0.0.5"
_HEADER = """| iter |      pcost |      dcost |     pres |     dres |      gap |   infeas |       mu |  q, p, c |     time |"""
_SEPARA = """|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|"""
_norm = np.linalg.norm
_EPS = 1e-15  # Standard epsilon for numerical safety


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
  HIT_TIME_LIMIT = "hit_time_limit"
  FAILED = "failed"
  UNFINISHED = "unfinished"


class Initialization(enum.Enum):
  """How to construct the IPM's starting iterate (x, y, s, tau)."""

  # Trivial: x = 0, y[z:] = s[z:] = 1, tau = 1.
  TRIVIAL = "trivial"

  # Mehrotra/CVXOPT-style: solve (P + A^T A) x = A^T b - c (the KKT system
  # [P, A^T; A, -I][x; y] = [-c; b] after eliminating y), set s = b - Ax and
  # y = Ax - b, then shift the inequality slices into the cone interior.
  CVXOPT = "cvxopt"

  # Fix s and y in the cone interior (s[z:] = y[z:] = 1) and solve for x by a
  # direct method that minimizes ||A x + s - b||^2 + ||P x + A^T y + c||^2,
  # i.e. (A^T A + P^T P) x = A^T (b - s) - P (A^T y + c).
  LEAST_SQUARES = "least_squares"


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


def _curtis_reid_balance(
    a: sp.spmatrix,
    p: sp.spmatrix,
    *,
    cg_rtol: float,
    cg_maxiter: int,
) -> np.ndarray:
  """Curtis-Reid (1972) symmetric scaling of |[P, A.T; A, 0]|.

  Finds positive scales x = exp(s), with s in R^(n+m), that minimize the
  sum of squared log-magnitudes of the scaled nonzero entries:

      L(s) = sum_{(i,j) in nz, i <= j}  ( log|K_ij| + s_i + s_j )^2

  The first-order optimality conditions are linear in s and form the symmetric
  positive-(semi)definite system

      Q s = b,
      Q[k,k] = N_k_off + 4 * 1[diag k nonzero]
      Q[k,j] = 1 for j != k with K_{kj} nonzero
      b[k]   = - sum_{j != k: nz} log|K_{kj}|  -  2 * 1[diag k nz] * log|K_{kk}|

  where N_k_off counts off-diagonal nonzeros in row k. Solved via CG.

  Memory-efficient implementation: the augmented |K| is never materialized.
  We work directly with the input A and P matrices' data and index arrays,
  scattering scalar quantities (row counts, log-sums, diagonal indicators)
  into N-sized vectors, and apply Q implicitly via a LinearOperator that
  composes matvecs against pattern-only views of A and P (sparse matrices
  built with shared indices/indptr arrays and a fresh data array of ones).

  Rows with no nonzero entries in K -- truly empty rows of A or P -- have
  N_k_off = 0 and diag_indicator = 0, so Q's k-th row collapses to a single
  diagonal entry of zero. We replace that entry with 1 so the CG system
  stays nonsingular; the corresponding b entry is also 0, so CG produces
  s_k = 0 (i.e., scale 1) automatically.

  Returns x of length n + m. e = x[:n] scales the columns of A and the rows
  /columns of P; d = x[n:] scales the rows of A.
  """
  n, m = p.shape[0], a.shape[0]
  N = n + m

  # Work in CSC throughout. Both .tocsc() calls are no-ops if already CSC.
  a_csc = a.tocsc() if a.format != "csc" else a
  p_csc = p.tocsc() if p.nnz > 0 and p.format != "csc" else p
  has_p = p_csc.nnz > 0

  if a_csc.nnz == 0 and not has_p:
    return np.ones(N, dtype=np.float64)

  # Per-row nonzero counts of K = [P, A.T; A, 0]. Top-n rows accumulate
  # contributions from |P| rows and from |A.T| rows (= |A| columns); the
  # bottom-m rows accumulate from |A| rows. CSC indices are row indices, so
  # scatter-add on indices counts per-row; np.diff(indptr) counts per-column.
  n_per_row = np.zeros(N, dtype=np.float64)
  if has_p:
    np.add.at(n_per_row[:n], p_csc.indices, 1.0)
  n_per_row[:n] += np.diff(a_csc.indptr).astype(np.float64)  # |A| col counts
  np.add.at(n_per_row[n:], a_csc.indices, 1.0)               # |A| row counts

  # Diagonal indicator and log magnitudes of diagonal: only the top-n block
  # can have a nonzero diagonal entry (bottom-right block is zero by
  # construction).
  diag_p = p_csc.diagonal() if has_p else np.zeros(n, dtype=np.float64)
  diag_indicator = np.zeros(N, dtype=np.float64)
  log_diag = np.zeros(N, dtype=np.float64)
  diag_pos = np.abs(diag_p) > 0
  diag_indicator[:n][diag_pos] = 1.0
  log_diag[:n][diag_pos] = np.log(np.abs(diag_p[diag_pos]))

  # Per-row sum of log magnitudes, including the diagonal contribution.
  # Same scatter-add pattern as the row counts but weighted by log|.|.
  total_log = np.zeros(N, dtype=np.float64)
  if has_p:
    log_p_data = np.log(np.abs(p_csc.data))
    np.add.at(total_log[:n], p_csc.indices, log_p_data)
    del log_p_data
  log_a_data = np.log(np.abs(a_csc.data))
  # |A| column-wise log sums into top-n: scatter by column index.
  a_col_idx = np.repeat(np.arange(n, dtype=np.intp), np.diff(a_csc.indptr))
  np.add.at(total_log[:n], a_col_idx, log_a_data)
  del a_col_idx
  np.add.at(total_log[n:], a_csc.indices, log_a_data)
  del log_a_data

  b_rhs = -(total_log + diag_indicator * log_diag)
  del total_log, log_diag

  # Q diagonal seen by the matvec: Q v = adj(K) v + adj_diag * v, where
  # adj_diag = N_k_off + 3 * diag_indicator = n_per_row + 2 * diag_indicator.
  # Empty rows (no entries in K) get adj_diag = 0; bump them to 1 so the
  # CG system stays nonsingular and resolves s_k = b_k / 1 = 0 there.
  adj_diag = n_per_row + 2.0 * diag_indicator
  adj_diag[adj_diag == 0.0] = 1.0
  del n_per_row, diag_indicator

  # Pattern-only views of A and P: same indices/indptr arrays as the inputs
  # (no copy), with the data array replaced by ones. These give us
  # `pattern @ v` = "sum v[j] over non-zero columns j of each row" matvecs.
  a_ones = sp.csc_matrix(
      (np.ones(a_csc.nnz, dtype=np.float64), a_csc.indices, a_csc.indptr),
      shape=a_csc.shape,
  )
  p_ones = sp.csc_matrix(
      (np.ones(p_csc.nnz, dtype=np.float64), p_csc.indices, p_csc.indptr),
      shape=p_csc.shape,
  ) if has_p else None
  a_ones_T = a_ones.T  # CSR view, no data copy

  def Q_matvec(v: np.ndarray) -> np.ndarray:
    out = np.empty(N, dtype=np.float64)
    v_top, v_bot = v[:n], v[n:]
    if p_ones is not None:
      out[:n] = p_ones @ v_top
      out[:n] += a_ones_T @ v_bot
    else:
      out[:n] = a_ones_T @ v_bot
    out[n:] = a_ones @ v_top
    out += adj_diag * v
    return out

  op = sp.linalg.LinearOperator(
      (N, N), matvec=Q_matvec, dtype=np.float64,
  )
  s, info = sp.linalg.cg(op, b_rhs, rtol=cg_rtol, maxiter=cg_maxiter)
  if info != 0:
    logging.debug(
        "Curtis-Reid CG returned info=%d (>0: maxiter, <0: breakdown)", info
    )

  return np.exp(s)


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
    self.a = a

    self.b = np.array(b, dtype=np.float64)
    if self.b.shape != (self.m,):
      raise ValueError(f"b must have shape ({self.m},), got {self.b.shape}")

    self.c = np.array(c, dtype=np.float64)
    if self.c.shape != (self.n,):
      raise ValueError(f"c must have shape ({self.n},), got {self.c.shape}")

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
      if p.shape != (self.n, self.n):
        raise ValueError(
            f"p must have shape ({self.n}, {self.n}), got {p.shape}"
        )
      if not np.all(np.isfinite(p.data)):
        raise ValueError("QP matrix 'p' must contain only finite values.")
      asymmetry = p - p.T
      if (
          asymmetry.nnz
          and np.max(np.abs(asymmetry.data), initial=0.0) > 1e-12
      ):
        raise ValueError("QP matrix 'p' must be symmetric.")
      self.p = p

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
    drop[self.z :] = ineq_b >= inf_bound
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

  def _init_variables(self):
    """Build the IPM starting iterate in original (un-equilibrated) space and
    map to equilibrated space if applicable. tau is always 1.
    """
    match self.initialization:
      case Initialization.TRIVIAL:
        x, y, s = self._init_trivial()
      case Initialization.CVXOPT:
        x, y, s = self._init_cvxopt()
      case Initialization.LEAST_SQUARES:
        x, y, s = self._init_least_squares()
      case _:
        raise ValueError(f"Unknown initialization: {self.initialization}")
    if self.equilibrate:
      x, y, s = self._equilibrate_iterates(x, y, s)
    return x, y, s, 1.0, {}

  def _init_trivial(self):
    x = np.zeros(self.n)
    y = np.zeros(self.m)
    s = np.zeros(self.m)
    y[self.z :] = 1.0
    s[self.z :] = 1.0
    return x, y, s

  def _init_cvxopt(self, reg: float = 1e-10, shift_buffer: float = 1.0):
    """Mehrotra/CVXOPT-style starting point.

    Solve the auxiliary saddle-point KKT system directly:
        [P + reg*I,  A^T] [x]   [-c]
        [A,          -I ] [y] = [ b]
    From the second block y = A x - b and Ax + s = b gives s = b - A x = -y on
    the inequality slice; both y[z:] and s[z:] are then shifted into the cone
    interior with a +1 buffer (Mehrotra heuristic).

    Eliminating y collapses to (P + A^T A + reg*I) x = A^T b - c, but we never
    form A^T A: it is typically much denser than A and P themselves and can be
    full in the worst case. Solving the augmented form via a single sparse
    factorization avoids that cost. reg regularizes only the (1,1) block so the
    elimination still matches the (P + A^T A + reg*I) formulation exactly.
    """
    z, m, n = self.z, self.m, self.n
    a, p = self.a, self.p
    p_reg = p + reg * sp.eye(n, format="csc") if reg > 0 else p
    kkt = sp.bmat(
        [[p_reg, a.T], [a, -sp.eye(m, format="csc")]],
        format="csc",
    )
    sol = sp.linalg.spsolve(kkt, np.concatenate([-self.c, self.b]))
    x = sol[:n]
    y = sol[n:]
    s = -y.copy()
    if m > z:
      ts = -np.min(s[z:])
      if ts >= 0:
        s[z:] += shift_buffer + ts
      ty = -np.min(y[z:])
      if ty >= 0:
        y[z:] += shift_buffer + ty
    s[:z] = 0.0
    return x, y, s

  def _init_least_squares(
      self, reg: float = 1e-10, atol: float = 1e-6, btol: float = 1e-6,
  ):
    """Fix s, y in the cone interior and solve for x via iterative least squares.

    Sets s[z:] = y[z:] = 1 and solves
        x = argmin (1/2) ||A x + s - b||^2 + (1/2) ||P x + A^T y + c||^2
    by treating it as the sparse least-squares problem
        min ||M x - d||^2 + reg*||x||^2,   M = [A; P], d = [b-s; -(A^T y + c)]
    and applying LSQR (Paige/Saunders). LSQR only requires matvecs with M and
    M^T, so we expose M as a LinearOperator: A^T A, P^2, the stacked matrix,
    and any direct factorization are all avoided. This keeps peak memory at
    O(nnz(A) + nnz(P)) — suitable for very large problems where even an
    augmented KKT factorization would exhaust memory. Initialization tolerance
    is loose (atol/btol = 1e-6) since the IPM only needs a feasible interior
    starting point, not a high-accuracy least-squares solution.
    """
    z, m, n = self.z, self.m, self.n
    a, p = self.a, self.p
    s = np.zeros(m)
    y = np.zeros(m)
    s[z:] = 1.0
    y[z:] = 1.0

    def matvec(v):
      return np.concatenate([a @ v, p @ v])

    def rmatvec(w):
      return a.T @ w[:m] + p @ w[m:]

    m_op = sp.linalg.LinearOperator(
        shape=(m + n, n), matvec=matvec, rmatvec=rmatvec, dtype=a.dtype,
    )
    d = np.concatenate([self.b - s, -(a.T @ y + self.c)])
    damp = np.sqrt(reg) if reg > 0 else 0.0
    x = sp.linalg.lsqr(m_op, d, damp=damp, atol=atol, btol=btol)[0]
    return x, y, s

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
      verbose: bool = True,
      equilibrate: bool = True,
      collect_stats: bool = False,
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
          equilibrate=equilibrate,
          collect_stats=collect_stats,
      )
    finally:
      if self._linear_solver is not None:
        self._linear_solver.free()
        self._linear_solver = None

  def _solve_impl(
      self,
      *,
      atol: float = 1e-7,
      rtol: float = 1e-8,
      atol_infeas: float = 1e-8,
      rtol_infeas: float = 1e-9,
      max_iter: int = 100,
      step_size_scale: float = 0.99,
      min_static_regularization: float = 1e-8,
      max_iterative_refinement_steps: int = 20,
      linear_solver_atol: float = 1e-12,
      linear_solver_rtol: float = 1e-12,
      linear_solver: LinearSolver = LinearSolver.AUTO,
      refinement_strategy: direct.RefinementStrategy = "fixed_point",
      fgmres_restart: int = 10,
      legacy_stall_check: bool = False,
      equilibrate_per_iteration: bool = False,
      verbose: bool = True,
      equilibrate: bool = True,
      equilibrate_constant_d_in_cone: bool = False,
      equilibrate_separate_scales: bool = False,
      equilibration_method: EquilibrationMethod = "ruiz",
      initialization: Initialization = Initialization.TRIVIAL,
      collect_stats: bool = False,
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
      time_limit_secs (float | None): If set, stop after the elapsed wall-clock
        time exceeds this many seconds and return SolutionStatus.HIT_TIME_LIMIT
        with the latest iterate. The check fires between linear solves and
        between iterative-refinement steps; it cannot interrupt a single
        in-progress backend call (notably KKT factorization), so the actual
        runtime can overshoot by the cost of one such call.
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
      refinement_strategy ("fixed_point" | "fgmres"): Iterative refinement
        scheme. "fixed_point" (default) is the classical residual-correction
        loop. "fgmres" wraps the same direct factorization as a right-
        preconditioner inside restarted FGMRES, which converges in fewer
        outer iterations on ill-conditioned KKT systems where pure IR has a
        contraction factor close to one.
      fgmres_restart (int): Inner-iteration count between FGMRES restarts.
        Each inner iteration costs one preconditioner solve plus one matvec
        and stores two vectors of size n+m. Ignored when refinement_strategy
        is "fixed_point".
      legacy_stall_check (bool): When True, restore the original stall
        criterion ("any non-decrease in residual norm halts refinement").
        When False (default), use a ratio-and-window criterion that tolerates
        small non-monotonic blips at the roundoff floor. Exposed for
        A/B comparison; the rest of the iterative-refinement loop (best-
        iterate rollback) is unaffected by this flag.
      equilibrate_per_iteration (bool): When True, apply one Ruiz-style
        symmetric rescaling of the full KKT matrix at every IPM iteration,
        on top of whatever one-shot equilibration was applied to A and P at
        the start. The per-iteration scaling targets the cone-block diagonal
        which spans many orders of magnitude as the active set separates;
        it is invisible to the IPM. Reported linear-solve residual norms
        are in the equilibrated frame and not directly comparable across
        iterations or against runs with the flag off. Not supported with
        GPU backends.
      verbose (bool): If True, prints a summary of each iteration.
      equilibrate (bool): If True, equilibrate the data for better numerical
        stability.
      equilibrate_constant_d_in_cone (bool): If True (and equilibrate is True),
        constrain the row scaling D to be constant within the positive-orthant
        cone (SCS convention for preserving cone membership under arbitrary
        cones). For QTQP's non-negative orthant, any positive diagonal D
        already preserves cone membership, so the default is False.
      equilibrate_separate_scales (bool): If True (and equilibrate is True),
        compute primal_scale (for c) and dual_scale (for b) independently from
        their inf-norms (the historical SCS behavior). If False, both equal a
        single sigma derived from max(||c||_inf, ||b||_inf) (current SCS).
      equilibration_method ("ruiz" | "curtis_reid"): Algorithm used by the
        one-shot equilibration of [P, A.T; A, 0] before the IPM starts.
        "ruiz" (default) is the classical inf-norm fixed-point. "curtis_reid"
        is the Curtis-Reid (1972) least-squares balancing in log-space, a
        single CG solve on a symmetric positive-definite normal-equations
        matrix. The downstream L2 polish and primal/dual scaling are
        identical for both methods.
      initialization (Initialization): How to construct the starting iterate.
        TRIVIAL uses x = 0 and unit y, s in the cone interior. CVXOPT solves
        a Mehrotra-style auxiliary KKT system and shifts s, y into the cone
        interior. LEAST_SQUARES fixes s and y in the cone interior and solves
        for x by a direct method that minimises the primal and dual residuals.
      collect_stats (bool): If True, collect per-iteration stats (sy, s_over_y
        statistics, complementarity, etc.) and return them in Solution.stats.
        Defaults to False for faster throughput; set True when per-iteration
        diagnostics are needed.

    Returns:
      A Solution object containing the solution and solve stats.
    """
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

    resolved_linear_solver, linear_solver_backend = _resolve_linear_solver(
        linear_solver
    )
    self.start_time = timeit.default_timer()
    self._deadline = (
        self.start_time + time_limit_secs if time_limit_secs is not None else None
    )
    self.atol, self.rtol = atol, rtol
    self.atol_infeas, self.rtol_infeas = atol_infeas, rtol_infeas
    self.verbose = verbose
    self.equilibrate = equilibrate
    self.initialization = initialization
    if verbose:
      print(
          f"| QTQP v{__version__}:"
          f" m={self.m}, n={self.n}, z={self.z}, nnz(A)={self.a.nnz},"
          f" nnz(P)={self.p.nnz}, linear_solver={resolved_linear_solver.name}"
      )

    if self.equilibrate:
      (a, p, b, c, self.d, self.e,
       self.primal_scale, self.dual_scale) = self._equilibrate(
          constant_d_in_cone=equilibrate_constant_d_in_cone,
          separate_scales=equilibrate_separate_scales,
          method=equilibration_method,
      )
    else:
      a, p, b, c = self.a, self.p, self.b, self.c
      self.d = self.e = self.primal_scale = self.dual_scale = None

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
        deadline=self._deadline,
        refinement_strategy=refinement_strategy,
        fgmres_restart=fgmres_restart,
        legacy_stall_check=legacy_stall_check,
        equilibrate_per_iteration=equilibrate_per_iteration,
    )

    stats = []
    self.kinv_q = np.zeros_like(self.q)  # K^{-1}q, warm-started across iterations.
    x, y, s, tau, _ = self._init_variables()
    status = SolutionStatus.UNFINISHED
    self._log_header()

    # Pre-allocate [x; y] and d_s to avoid repeated allocation each iteration.
    xy = np.empty(self.n + self.m)   # Combined primal-dual vector [x; y]
    d_s = np.zeros(self.m)           # Slack step direction; d_s[:z] is always 0

    alpha = sigma = 0.0

    # --- Main Iteration Loop ---
    # self.it counts IPM steps already taken.
    for self.it in range(max_iter):
      stats_i = {}
      x, y, tau, s = self._normalize(x, y, tau, s)

      mu = (y @ s) / (self.m - self.z)

      # --- Take an IPM step ---
      self._linear_solver.update(mu=mu, s=s, y=y)

      # --- Step 1: Precompute kinv_q = K^{-1} @ q ---
      # This is reused for both predictor and corrector parts of the step.
      self.kinv_q, q_lin_sys_stats = self._linear_solver.solve(
          rhs=self.q, warm_start=self.kinv_q
      )
      stats_i["q_lin_sys_stats"] = q_lin_sys_stats

      if self._time_limit_reached():
        status = SolutionStatus.HIT_TIME_LIMIT
        break

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

      if self._time_limit_reached():
        status = SolutionStatus.HIT_TIME_LIMIT
        break

      d_x_p, d_y_p, d_tau_p = x_p - x, y_p - y, tau_p - tau
      # Predictor slack step from the linearized complementarity condition with
      # target=0: (y + d_y)(s + d_s) ≈ 0 => d_s = -(y + d_y)*s/y = -y_p*s/y.
      d_s[self.z :] = -y_p[self.z :] * s[self.z :] / y[self.z :]

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
      # Combined corrector slack step: same form as the predictor but now with
      # centering target sigma*mu and the Mehrotra correction baked in.
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
      if collect_stats:
        stats[-1]["status"] = status

    # We have terminated for one reason or another.
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
      case SolutionStatus.UNFINISHED:
        self._log_footer(f"Failed to converge")
        x, y, s = x / tau, y / tau, s / tau
        y, s = self._postsolve(y, s, s_dropped=self._dropped_slack(x))
        return Solution(x, y, s, stats, SolutionStatus.FAILED)
      case _:
        raise ValueError(f"Unknown convergence status: {status}")

  def _equilibrate(
      self,
      num_ruiz_passes: int = 25,
      num_l2_passes: int = 1,
      min_norm: float = 1e-4,
      max_norm: float = 1e4,
      constant_d_in_cone: bool = False,
      separate_scales: bool = False,
      method: EquilibrationMethod = "ruiz",
      cr_cg_rtol: float = 1e-8,
      cr_cg_maxiter: int = 200,
  ):
    """SCS-style Ruiz + ell_2 equilibration on [[P, A^T], [A, 0]] plus
    primal/dual scaling.

    Produces diagonal scales E (size n), D (size m) and positive scalars
    primal_scale, dual_scale (equal in single-scale mode) such that:
        P_hat = (rho_p / rho_d) E P E,  A_hat = D A E,
        c_hat = rho_p E c,             b_hat = rho_d D b.
    The (rho_p / rho_d) factor on P_hat keeps the equilibrated dual KKT in
    standard form when the two scales differ; when they are equal it is just
    E P E. The original solution is recovered as
        x = E x_hat / rho_d,  y = D y_hat / rho_p,  s = D^{-1} s_hat / rho_d.

    constant_d_in_cone:
      If True, D is held constant across the positive-orthant cone (Ruiz uses
      the in-cone ell_infinity, L2 uses the in-cone mean) so cone membership
      is preserved under arbitrary cones (SCS convention). For QTQP the only
      cone is the non-negative orthant, which is preserved by any positive
      diagonal D, so the default is False (per-row scaling).

    separate_scales:
      If False (default, current SCS behavior), one shared sigma rescales both
      b and c, computed from max(||c||_inf, ||b||_inf). If True, primal_scale
      is computed from ||c||_inf and dual_scale from ||b||_inf independently
      (the historical SCS form before they were unified).

    method:
      "ruiz" (default) runs the classical Ruiz inf-norm fixed-point pass for
      num_ruiz_passes iterations. "curtis_reid" runs the Curtis-Reid (1972)
      least-squares balancing on |[P, A.T; A, 0]|, which finds positive scales
      x such that the scaled nonzero entries cluster around magnitude one in
      the log-mean-squared sense. CR is a single sparse linear solve (CG on a
      symmetric positive-definite normal-equations matrix), not an iterative
      pass. The downstream L2 polish and primal/dual scaling are identical
      for both methods.
    """
    # Work on copies so self.a / self.p stay in original space for use in
    # _check_termination (which un-equilibrates iterates and compares against
    # the original problem data).
    a, p = self.a.copy(), self.p.copy()
    b, c = self.b.copy(), self.c.copy()
    d, e = np.ones(self.m), np.ones(self.n)
    z, m = self.z, self.m

    def _to_scale(v):
      # Treat tiny norms as 1 (no scaling) and clip large norms; both Ruiz and
      # L2 then take 1/sqrt as the scaling factor.
      v = np.where(v < min_norm, 1.0, np.minimum(v, max_norm))
      return 1.0 / np.sqrt(v)

    def _apply(d_step, e_step):
      # In-place D A E and E P E on CSC data arrays.
      col_scale_a = np.repeat(e_step, np.diff(a.indptr))
      a.data *= d_step[a.indices] * col_scale_a
      if p.nnz > 0:
        col_scale_p = np.repeat(e_step, np.diff(p.indptr))
        p.data *= e_step[p.indices] * col_scale_p

    if method == "ruiz":
      for i in range(num_ruiz_passes):
        d_step = sp.linalg.norm(a, np.inf, axis=1)
        e_step = np.maximum(
            sp.linalg.norm(a, np.inf, axis=0),
            sp.linalg.norm(p, np.inf, axis=0),
        )
        if constant_d_in_cone and m > z:
          d_step[z:] = np.max(d_step[z:])
        d_step, e_step = _to_scale(d_step), _to_scale(e_step)
        _apply(d_step, e_step)
        d *= d_step
        e *= e_step
        logging.debug(
            "Ruiz pass %d: d err %s, e err %s",
            i, _norm(d_step - 1, np.inf), _norm(e_step - 1, np.inf),
        )
    elif method == "curtis_reid":
      x = _curtis_reid_balance(
          a, p, cg_rtol=cr_cg_rtol, cg_maxiter=cr_cg_maxiter,
      )
      e_step, d_step = x[: self.n], x[self.n :]
      if constant_d_in_cone and m > z:
        d_step[z:] = np.max(d_step[z:])
      lo, hi = 1.0 / math.sqrt(max_norm), 1.0 / math.sqrt(min_norm)
      np.clip(d_step, lo, hi, out=d_step)
      np.clip(e_step, lo, hi, out=e_step)
      _apply(d_step, e_step)
      d *= d_step
      e *= e_step
      logging.debug(
          "Curtis-Reid: d err %s, e err %s",
          _norm(d_step - 1, np.inf), _norm(e_step - 1, np.inf),
      )
    else:
      raise ValueError(
          f"Unknown equilibration method {method!r}; "
          "expected 'ruiz' or 'curtis_reid'."
      )

    for i in range(num_l2_passes):
      d_step = sp.linalg.norm(a, ord=2, axis=1)
      e_step = np.sqrt(
          sp.linalg.norm(a, ord=2, axis=0) ** 2
          + sp.linalg.norm(p, ord=2, axis=0) ** 2
      )
      if constant_d_in_cone and m > z:
        d_step[z:] = np.mean(d_step[z:])
      d_step, e_step = _to_scale(d_step), _to_scale(e_step)
      _apply(d_step, e_step)
      d *= d_step
      e *= e_step
      logging.debug(
          "L2 pass %d: d err %s, e err %s",
          i, _norm(d_step - 1, np.inf), _norm(e_step - 1, np.inf),
      )

    # primal_scale rescales c, dual_scale rescales b. Norms are taken on
    # E*c and D*b, clipped to [min_norm, max_norm]; tiny values fall back to
    # 1.0 to avoid amplifying noise.
    c *= e
    b *= d

    def _scale_from_norm(nrm):
      if nrm < min_norm:
        nrm = 1.0
      return 1.0 / min(nrm, max_norm)

    if separate_scales:
      primal_scale = _scale_from_norm(_norm(c, np.inf))
      dual_scale = _scale_from_norm(_norm(b, np.inf))
      # Absorb the (rho_p/rho_d) factor into P_hat so the IPM still sees a
      # standard KKT system when the two scales differ.
      if primal_scale != dual_scale and p.nnz > 0:
        p.data *= primal_scale / dual_scale
    else:
      sigma = _scale_from_norm(max(_norm(c, np.inf), _norm(b, np.inf)))
      primal_scale = dual_scale = sigma

    c *= primal_scale
    b *= dual_scale

    return a, p, b, c, d, e, primal_scale, dual_scale

  def _unequilibrate_iterates(self, x, y, s):
    return (
        self.e * x / self.dual_scale,
        self.d * y / self.primal_scale,
        s / (self.d * self.dual_scale),
    )

  def _equilibrate_iterates(self, x, y, s):
    return (
        self.dual_scale * x / self.e,
        self.primal_scale * y / self.d,
        self.dual_scale * self.d * s,
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

    # Tau solve: exact quadratic when KKT solve converged, linearized
    # fallback when noisy (avoids squaring O(eps) into O(eps^2)).
    tau_plus = None
    if lin_sys_stats["converged"] or lin_sys_stats["final_residual_norm"] < 1e-7:
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
    g = (mu * tau_curr * tau_curr + (mu_target - mu) * tau_anchor * tau_curr -
         tau_curr * q_z - mu_target - x_px)

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
    if self.equilibrate:
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

    # Residuals
    pres = _norm((ax + s) * inv_tau - self.b, np.inf)
    dres = _norm((px + aty) * inv_tau + self.c, np.inf)
    gap = abs((ctx + bty + xpx * inv_tau) * inv_tau)

    # Infeasibility certificates (Farkas-type, from the embedding structure).
    # If the primal is unbounded (dual infeasible) this produces a ray x with
    # c'x < 0 that satisfies the homogeneous primal conditions Ax + s ≈ 0 and Px
    # ≈ 0.  dinfeas measures how well x/|c'x| certifies dual infeasibility.
    norm_aty = _norm(aty, np.inf)
    norm_px = _norm(px, np.inf)
    dinfeas_a = _norm((ax + s), np.inf) / (abs(ctx) + _EPS)
    dinfeas_p = norm_px / (abs(ctx) + _EPS)
    dinfeas = max(dinfeas_a, dinfeas_p)
    # If the primal is infeasible (dual unbounded) this produces a ray y
    # with b'y < 0 that satisfies the homogeneous dual condition A'y ≈ 0.
    # pinfeas measures how well y/|b'y| certifies primal infeasibility.
    pinfeas = norm_aty / (abs(bty) + _EPS)

    # Primal residual tolerance relative scale.
    prelrhs = max(
        _norm(ax, np.inf) * inv_tau,
        _norm(s, np.inf) * inv_tau,
        self._norm_b,
    )

    # Dual residual tolerance relative scale.
    drelrhs = max(
        norm_px * inv_tau,
        norm_aty * inv_tau,
        self._norm_c,
    )

    norm_x = _norm(x, np.inf)
    norm_y = _norm(y, np.inf)

    # Solved: duality gap and both residuals are within tolerance.
    if (
        gap < self.atol + self.rtol * min(abs(pcost), abs(dcost))
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

  def _time_limit_reached(self) -> bool:
    """True iff a deadline was set and the current time has exceeded it."""
    return (
        self._deadline is not None
        and timeit.default_timer() > self._deadline
    )

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
