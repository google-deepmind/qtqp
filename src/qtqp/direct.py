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
"""Direct KKT linear system solvers."""

import logging
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp


class LinearSolver:
  """Base class for KKT linear system solvers.

  To add a new solver, subclass this and implement factorize, solve, and format.
  set_kkt, __matmul__, and free are provided; override __matmul__ for a more
  efficient matvec (e.g. a pre-allocated dense array).
  """

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    """Stores the KKT matrix; called by DirectKktSolver before factorize."""
    self._kkt = kkt

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
    return self._kkt @ x

  def format(self) -> str:
    """Preferred sparse format for the KKT scaffold ('csc' or 'csr')."""
    raise NotImplementedError

  def free(self) -> None:
    pass


class MklPardisoSolver(LinearSolver):
  """Wrapper around pydiso.mkl_solver.MKLPardisoSolver."""

  def __init__(self):
    import pydiso.mkl_solver  # pylint: disable=g-import-not-at-top

    self.mkl_solver = pydiso.mkl_solver
    self.factorization: pydiso.mkl_solver.MKLPardisoSolver | None = None

  def factorize(self):
    if self.factorization is None:
      self.factorization = self.mkl_solver.MKLPardisoSolver(
          self._kkt, matrix_type="real_symmetric_indefinite"
      )
      # iparm(12)=1: improved accuracy via symmetric weighted matching.
      # iparm(24)=1: two-level parallel factorization algorithm.
      # iparm(25)=2: parallel forward/backward substitution.
      #   Not yet in pydiso's settable whitelist, see:
      #   https://github.com/simpeg/pydiso/issues/XX
      # Note: these are set after __init__ (which calls analyze+factor),
      # so they only take effect from the second factorization onward.
      self.factorization.set_iparm(12, 1)
      self.factorization.set_iparm(24, 1)
      # self.factorization.set_iparm(25, 2)
    else:
      self.factorization.refactor(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    try:
      return self.factorization.solve(rhs)
    except self.mkl_solver.PardisoError as e:
      logging.warning("PardisoError: %s", e)
      logging.warning("Performing analysis and factorization steps again.")
      self.factorization._analyze()  # pylint: disable=protected-access
      self.factorization._factor()  # pylint: disable=protected-access
      return self.factorization.solve(rhs)

  def format(self) -> Literal["csr"]:
    return "csr"


class QdldlSolver(LinearSolver):
  """Wrapper around qdldl.Solver for quasi-definite LDL factorization."""

  def __init__(self):
    import qdldl  # pylint: disable=g-import-not-at-top

    self.qdldl = qdldl
    self.factorization: qdldl.Solver | None = None

  def factorize(self):
    if self.factorization is None:
      self.factorization = self.qdldl.Solver(self._kkt)
    else:
      self.factorization.update(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class ScipySolver(LinearSolver):
  """Wrapper around scipy.linalg.factorized."""

  def __init__(self):
    self.factorization = None

  def factorize(self):
    self.factorization = sp.linalg.factorized(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class CholModSolver(LinearSolver):
  """Wrapper around sksparse.cholmod for Cholesky LDLt factorization."""

  def __init__(self):
    import sksparse.cholmod  # pylint: disable=g-import-not-at-top

    self.cholmod = sksparse.cholmod
    self.factorization: sksparse.cholmod.CholeskyFactor | None = None

  def factorize(self):
    if self.factorization is None:
      self.factorization = self.cholmod.cholesky(self._kkt, mode="simplicial")
    else:
      self.factorization.cholesky_inplace(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class EigenSolver(LinearSolver):
  """Wrapper around Eigen Simplicial LDL^T."""

  def __init__(self):
    import nanoeigenpy  # pylint: disable=g-import-not-at-top

    self.nanoeigenpy = nanoeigenpy
    self._solver: nanoeigenpy.SimplicialLDLT | None = None

  def factorize(self):
    if self._solver is None:
      self._solver = self.nanoeigenpy.SimplicialLDLT()
      self._solver.analyzePattern(self._kkt)

    self._solver.factorize(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self._solver.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class MumpsSolver(LinearSolver):
  """Wrapper for MUMPS solver (via petsc4py)."""

  def __init__(self):
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top

    self.PETSc = petsc4py.PETSc  # pylint: disable=invalid-name
    self.ksp = self.PETSc.KSP().create()

    # Configure as a direct solver (apply preconditioner only)
    self.ksp.setType(self.PETSc.KSP.Type.PREONLY)
    self.ksp.getPC().setType(self.PETSc.PC.Type.LU)
    self.ksp.getPC().setFactorSolverType("mumps")

    # Allow command-line customization (eg, -mat_mumps_icntl_14 20).
    self.ksp.setFromOptions()

  def factorize(self):
    kkt_wrapper = self.PETSc.Mat().createAIJ(
        size=self._kkt.shape, csr=(self._kkt.indptr, self._kkt.indices, self._kkt.data)
    )
    kkt_wrapper.setOption(self.PETSc.Mat.Option.SYMMETRIC, True)
    kkt_wrapper.setOption(self.PETSc.Mat.Option.SPD, False)
    kkt_wrapper.assemble()

    # Check if KSP already has a matrix defined to determine the flag
    already_factorized = self.ksp.getOperators()[0] is not None
    if already_factorized:
      flag = self.PETSc.Mat.Structure.SAME_NONZERO_PATTERN
    else:
      flag = self.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN

    try:
      self.ksp.setOperators(kkt_wrapper, kkt_wrapper, flag)
    except TypeError:
      # Fallback for older petsc4py API; usually auto-detects reuse
      self.ksp.setOperators(kkt_wrapper, kkt_wrapper)

    # Force factorization (symbolic first time, numeric every time)
    self.ksp.setUp()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    b = self.PETSc.Vec().createWithArray(rhs.ravel())
    x = self.PETSc.Vec().createSeq(rhs.size)
    self.ksp.solve(b, x)
    return x.getArray().real.reshape(rhs.shape)

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self.ksp is not None:
      self.ksp.destroy()
      self.ksp = None


class CuDssSolver(LinearSolver):
  """Wrapper around Nvidia's CuDSS for GPU-accelerated solving."""

  def __init__(self):
    import nvmath  # pylint: disable=g-import-not-at-top

    self.nvmath = nvmath
    self._solver: nvmath.sparse.advanced.DirectSolver | None = None

  def factorize(self):
    if self._solver is None:
      sparse_system_type = (
          self.nvmath.sparse.advanced.DirectSolverMatrixType.SYMMETRIC
      )
      # Turn off annoying logs by default.
      logger = logging.getLogger("null")
      logger.disabled = True
      options = self.nvmath.sparse.advanced.DirectSolverOptions(
          sparse_system_type=sparse_system_type, logger=logger
      )
      # RHS must be in column major order (Fortran) for cuDSS.
      dummy_rhs = np.empty(self._kkt.shape[1], order="F", dtype=np.float64)
      self._solver = self.nvmath.sparse.advanced.DirectSolver(
          self._kkt, dummy_rhs, options=options
      )
      self._solver.plan()
    else:
      self._solver.reset_operands(a=self._kkt)

    self._solver.factorize()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    # Ensure RHS is Fortran contiguous for cuDSS expected input format
    rhs_fortran = np.asfortranarray(rhs, dtype=np.float64)
    self._solver.reset_operands(b=rhs_fortran)
    return self._solver.solve()

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self._solver is not None:
      self._solver.free()
      self._solver = None
      # Force clean up any 'zombie' references, in order to avoid cuda errors.
      import gc  # pylint: disable=g-import-not-at-top
      gc.collect(0)  # Run GC only on the youngest generation.


class UmfpackSolver(LinearSolver):
  """Wrapper around UMFPACK (via scikit-umfpack) for LU factorization.

  Unlike ScipySolver (scipy SuperLU), UMFPACK separates symbolic and numeric
  factorization phases. Symbolic analysis runs only on the first call since the
  sparsity pattern is fixed across IPM iterations; subsequent calls redo only
  the cheaper numeric factorization.
  """

  def __init__(self):
    import scikits.umfpack as umfpack  # pylint: disable=g-import-not-at-top

    self._umfpack = umfpack
    self._ctx = umfpack.UmfpackContext("di")
    self._symbolic_done = False

  def factorize(self):
    if not self._symbolic_done:
      self._ctx.symbolic(self._kkt)
      self._symbolic_done = True
    self._ctx.numeric(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self._ctx.solve(self._umfpack.UMFPACK_A, self._kkt, rhs, autoTranspose=True)

  def format(self) -> Literal["csc"]:
    return "csc"


class ScipyDenseSolver(LinearSolver):
  """Dense LU solver via LAPACK for small KKT systems.

  For small problems (n+m < ~200), avoids the overhead of sparse data
  structures (symbolic analysis, CSC index arrays, indirect memory access)
  that dominates over actual arithmetic in sparse solvers.

  Calls LAPACK (dgetrf/dgetrs) directly with cached function handles to avoid
  per-call Python wrapper overhead (input validation, lapack func lookup) in
  scipy.linalg.lu_factor/lu_solve, which dominates for tiny matrices.

  Two Fortran-order buffers are kept: _kkt_dense holds the original matrix
  values for the matvec; _lu_dense is a copy that dgetrf overwrites in-place
  with the LU factors (overwrite_a=True avoids a per-call allocation).
  """

  def __init__(self):
    from scipy.linalg import lapack  # pylint: disable=g-import-not-at-top

    # Cache function handles to avoid per-call get_lapack_funcs lookup.
    self._dgetrf = lapack.dgetrf
    self._dgetrs = lapack.dgetrs
    self._piv = None
    self._kkt_dense: np.ndarray | None = None
    self._lu_dense: np.ndarray | None = None

  def factorize(self) -> None:
    if self._kkt_dense is None:
      # Fortran order so dgetrf can overwrite in-place without an internal copy.
      self._kkt_dense = np.asfortranarray(self._kkt.toarray())
      self._lu_dense = self._kkt_dense.copy(order="F")
    else:
      # Only the diagonal changes each IPM iteration; off-diagonal blocks are fixed.
      np.fill_diagonal(self._kkt_dense, self._kkt.diagonal())
      np.copyto(self._lu_dense, self._kkt_dense)
    self._lu_dense, self._piv, _ = self._dgetrf(self._lu_dense, overwrite_a=True)

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._kkt_dense @ x

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    x, _ = self._dgetrs(self._lu_dense, self._piv, rhs)
    return x

  def format(self) -> Literal["csr"]:
    return "csr"


class DenseLdltSolver(LinearSolver):
  """Dense symmetric indefinite (Bunch-Kaufman LDLT) solver via LAPACK.

  The KKT matrix is symmetric indefinite, so LDLT (dsytrf/dsytrs) needs
  roughly half the flops of general LU (dgetrf/dgetrs).  Otherwise follows
  the same buffer-reuse strategy as ScipyDenseSolver.
  """

  def __init__(self):
    from scipy.linalg import lapack  # pylint: disable=g-import-not-at-top

    self._dsytrf = lapack.dsytrf
    self._dsytrs = lapack.dsytrs
    self._piv = None
    self._kkt_dense: np.ndarray | None = None
    self._ldl_dense: np.ndarray | None = None

  def factorize(self) -> None:
    if self._kkt_dense is None:
      self._kkt_dense = np.asfortranarray(self._kkt.toarray())
      self._ldl_dense = self._kkt_dense.copy(order="F")
    else:
      np.fill_diagonal(self._kkt_dense, self._kkt.diagonal())
      np.copyto(self._ldl_dense, self._kkt_dense)
    self._ldl_dense, self._piv, _ = self._dsytrf(
        self._ldl_dense, lower=True, overwrite_a=True
    )

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._kkt_dense @ x

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    x, _ = self._dsytrs(self._ldl_dense, self._piv, rhs, lower=True)
    return x

  def format(self) -> Literal["csr"]:
    return "csr"


class CupyDenseSolver(LinearSolver):
  """Dense LU solver on GPU via cupy (cuSOLVER).

  Transfers the KKT matrix to GPU once and updates only the diagonal each
  iteration.  Uses cupyx.lapack for direct getrf/getrs calls.  All
  reusable GPU buffers (pivot array, vector temporaries) are allocated
  once and reused across iterations to avoid per-call allocations.
  """

  def __init__(self):
    import cupy  # pylint: disable=g-import-not-at-top
    import cupyx.lapack  # pylint: disable=g-import-not-at-top

    self._cp = cupy
    self._lapack = cupyx.lapack
    self._kkt_gpu: cupy.ndarray | None = None
    self._lu_gpu: cupy.ndarray | None = None
    self._piv_gpu: cupy.ndarray | None = None
    self._diag_idx: cupy.ndarray | None = None
    # Pre-allocated GPU buffers for matvec and solve.
    self._x_gpu: cupy.ndarray | None = None
    self._rhs_gpu: cupy.ndarray | None = None

  def factorize(self) -> None:
    cp = self._cp
    if self._kkt_gpu is None:
      self._kkt_gpu = cp.asfortranarray(cp.asarray(self._kkt.toarray()))
      self._lu_gpu = self._kkt_gpu.copy(order="F")
      n = self._kkt_gpu.shape[0]
      self._diag_idx = cp.arange(n)
      self._piv_gpu = cp.empty(n, dtype=cp.int32)
      self._x_gpu = cp.empty(n, dtype=cp.float64)
      self._rhs_gpu = cp.empty((n, 1), dtype=cp.float64, order="F")
    else:
      self._kkt_gpu[self._diag_idx, self._diag_idx] = cp.asarray(
          self._kkt.diagonal()
      )
      cp.copyto(self._lu_gpu, self._kkt_gpu)
    self._lapack.getrf(self._lu_gpu, self._piv_gpu)

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    self._x_gpu.set(x)
    return (self._kkt_gpu @ self._x_gpu).get()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    self._rhs_gpu[:, 0].set(rhs)
    self._lapack.getrs(self._lu_gpu, self._piv_gpu, self._rhs_gpu)
    return self._rhs_gpu[:, 0].get()

  def format(self) -> Literal["csr"]:
    return "csr"


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

    # Build the KKT scaffold once. NaN is used as a sentinel to mark the
    # diagonal positions that must be overwritten each iteration (because they
    # depend on mu, s, y). All off-diagonal and constant entries are fixed here.
    # After construction, kkt_nan_idxs caches exactly which entries in the
    # sparse data array are NaN, giving O(nnz_diag) updates instead of a full
    # matrix rebuild.
    n_nans = sp.diags(np.full(self.n, np.nan, dtype=np.float64), format="csc")
    m_nans = sp.diags(np.full(self.m, np.nan, dtype=np.float64), format="csc")

    self._kkt = sp.bmat(
        [[p + n_nans, a.T], [a, m_nans]],
        format=self._solver.format(),
        dtype=np.float64,
    )
    self._kkt_nan_idxs = np.isnan(self._kkt.data)  # Sentinel positions to update.

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
    self._kkt.data[self._kkt_nan_idxs] = self._reg_diags
    self._solver.set_kkt(self._kkt)
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
