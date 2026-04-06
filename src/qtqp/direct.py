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
      # Recommended iparms for IPMs from Pardiso docs.
      # These only affect the analysis step so should be set before __init__,
      # but this is not currently possible with the current interface.
      self.factorization.set_iparm(10, 1)
      self.factorization.set_iparm(12, 1)
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
  """Wrapper for MUMPS solver (via petsc4py).

  Creates a single PETSc Mat/KSP pair on the first factorize call and reuses
  them across iterations.  Only the numeric values are updated each call;
  the sparsity pattern (and MUMPS symbolic analysis) is computed once.
  """

  def __init__(self):
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top

    self._PETSc = petsc4py.PETSc
    self._mat = None
    self._ksp = None
    self._b = None
    self._x = None

  def factorize(self):
    PETSc = self._PETSc
    kkt = self._kkt

    if self._mat is None:
      # First call: build PETSc Mat from CSR, configure KSP + MUMPS.
      # createAIJWithArrays shares the scipy data buffer, so
      # DirectKktSolver's in-place diagonal updates are visible to PETSc
      # without any copy.  On subsequent factorize calls we just bump the
      # state counter and refactorize.
      self._mat = PETSc.Mat().createAIJWithArrays(
          kkt.shape, (kkt.indptr, kkt.indices, kkt.data)
      )
      self._mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
      self._mat.setOption(PETSc.Mat.Option.SPD, False)
      self._mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
      self._mat.assemble()

      self._ksp = PETSc.KSP().create()
      self._ksp.setType(PETSc.KSP.Type.PREONLY)
      self._pc = self._ksp.getPC()
      self._pc.setType(PETSc.PC.Type.LU)
      self._pc.setFactorSolverType("mumps")
      self._ksp.setOperators(self._mat)

      # MUMPS ICNTL parameters must be set before setUp (before symbolic
      # analysis).  We use PETSc options which are read during setUp.
      opts = PETSc.Options()
      # ICNTL(14): percentage increase in estimated working space.
      # Quasidefinite KKT matrices need generous headroom; MUMPS may
      # silently produce poor-quality factors when space is tight.
      opts.setValue("-mat_mumps_icntl_14", "2000")
      # ICNTL(24): null pivot detection — important for near-singular
      # systems that arise during infeasibility / unboundedness detection.
      opts.setValue("-mat_mumps_icntl_24", "1")
      # Allow further command-line customization to override the above.
      self._ksp.setFromOptions()

      # Trigger symbolic analysis + first numeric factorization.
      self._ksp.setUp()
      self._F = self._pc.getFactorMatrix()
      self._mumps_icntl_14 = self._F.getMumpsIcntl(14)

      # Pre-allocate RHS and solution vectors.
      self._b = self._mat.createVecRight()
      self._x = self._mat.createVecRight()
      self._sol = np.empty(kkt.shape[0], dtype=np.float64)
    else:
      # Subsequent calls: the shared data array already has the new values
      # (DirectKktSolver updates the scipy matrix in-place).  We just need
      # to tell PETSc the values changed and redo the numeric factorization.
      self._mat.stateIncrease()
      self._ksp.setUp()

    # If MUMPS ran out of working space (INFOG(1) == -9), double ICNTL(14)
    # and retry until it succeeds.
    while self._F.getMumpsInfog(1) == -9:
      self._mumps_icntl_14 *= 2
      self._F.setMumpsIcntl(14, self._mumps_icntl_14)
      self._mat.stateIncrease()
      self._ksp.setUp()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    self._b.array[:] = rhs
    self._ksp.solve(self._b, self._x)
    np.copyto(self._sol, self._x.array)
    return self._sol

  def format(self) -> Literal["csr"]:
    return "csr"

  def free(self):
    """Frees the solver resources."""
    if self._ksp is not None:
      self._ksp.destroy()
      self._ksp = None
    if self._mat is not None:
      self._mat.destroy()
      self._mat = None
    self._b = None
    self._x = None


class CuDssSolver(LinearSolver):
  """Wrapper around Nvidia's CuDSS for GPU-accelerated solving.

  Maintains a single GPU sparse matrix used for both nvmath (factorize/solve)
  and cupy matvec.  On the first call, the CPU matrix is converted to a cupy
  GPU sparse matrix which is passed to nvmath's DirectSolver.  On subsequent
  calls the GPU data array is updated in-place via .set(); nvmath wraps the
  data pointer so it sees the new values without needing reset_operands
  (which would invalidate the plan).
  """

  def __init__(self):
    import cupy  # pylint: disable=g-import-not-at-top
    import cupyx.scipy.sparse  # pylint: disable=g-import-not-at-top
    import nvmath  # pylint: disable=g-import-not-at-top

    self._cp = cupy
    self._cp_sparse = cupyx.scipy.sparse
    self.nvmath = nvmath
    self._solver: nvmath.sparse.advanced.DirectSolver | None = None
    # Single GPU sparse matrix for both nvmath and matvec.
    self._kkt_gpu = None
    self._x_gpu: cupy.ndarray | None = None
    self._rhs_gpu: cupy.ndarray | None = None

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    """Transfers KKT data to GPU; does not retain the CPU matrix."""
    if self._kkt_gpu is None:
      self._kkt_gpu = self._cp_sparse.csr_matrix(kkt)
    else:
      self._kkt_gpu.data.set(kkt.data)

  def factorize(self):
    cp = self._cp
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
      n = self._kkt_gpu.shape[1]
      self._x_gpu = cp.empty(n, dtype=cp.float64)
      self._rhs_gpu = cp.empty(n, order="F", dtype=cp.float64)
      self._solver = self.nvmath.sparse.advanced.DirectSolver(
          self._kkt_gpu, self._rhs_gpu, options=options
      )
      self._solver.plan()
    # No reset_operands: nvmath wraps _kkt_gpu's data pointer, so
    # in-place updates via .set() in set_kkt are visible directly.

    self._solver.factorize()

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    self._x_gpu.set(x)
    return (self._kkt_gpu @ self._x_gpu).get()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    self._rhs_gpu.set(rhs)
    result = self._solver.solve()
    return self._cp.asnumpy(result)

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

  Converts the sparse KKT to a dense GPU matrix via set_kkt and re-transfers
  the full dense matrix each iteration.  Uses cupyx.scipy.linalg.lu_factor/
  lu_solve for LU factorization.  lu_factor overwrites its input, so a copy
  is made each iteration while _kkt_gpu stays pristine for matvec.
  """

  def __init__(self):
    import cupy  # pylint: disable=g-import-not-at-top
    import cupyx.scipy.linalg  # pylint: disable=g-import-not-at-top

    self._cp = cupy
    self._linalg = cupyx.scipy.linalg
    self._kkt_gpu: cupy.ndarray | None = None
    self._lu_and_piv = None  # (lu, piv) tuple from lu_factor
    # Pre-allocated GPU buffers for matvec and solve.
    self._x_gpu: cupy.ndarray | None = None
    self._rhs_gpu: cupy.ndarray | None = None

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    """Transfers KKT to GPU; does not retain the CPU matrix."""
    if self._kkt_gpu is None:
      # First call: convert full sparse matrix to dense on GPU.
      self._kkt_gpu = self._cp.asarray(kkt.toarray(), dtype=self._cp.float64)
    else:
      # Re-transfer the full dense matrix.  Only n+m diagonal entries
      # change between iterations but toarray() is the simplest way to
      # get a consistent dense view from the sparse scaffold.
      self._kkt_gpu.set(kkt.toarray())

  def factorize(self) -> None:
    cp = self._cp
    if self._x_gpu is None:
      n = self._kkt_gpu.shape[0]
      self._x_gpu = cp.empty(n, dtype=cp.float64)
      self._rhs_gpu = cp.empty(n, dtype=cp.float64)
    # lu_factor overwrites its input, so pass a copy.
    self._lu_and_piv = self._linalg.lu_factor(
        self._kkt_gpu.copy(), overwrite_a=True
    )

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    self._x_gpu.set(x)
    return (self._kkt_gpu @ self._x_gpu).get()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    self._rhs_gpu.set(rhs)
    result = self._linalg.lu_solve(self._lu_and_piv, self._rhs_gpu)
    return self._cp.asnumpy(result)

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
