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

  def set_dims(self, n: int, m: int, z: int) -> None:
    """Stores problem dimensions; called once before set_kkt.

    Dense solvers override this to pre-allocate buffers sized by n and m.
    Sparse solvers ignore it (the default is a no-op).
    """
    pass

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
      # pydiso uses 0-based C indexing: set_iparm(i, v) sets C iparm[i],
      # which is Fortran iparm(i+1).  Intel recommends for symmetric
      # indefinite IPM/saddle-point systems:
      #   Fortran iparm(11) = 1: scaling (C index 10)
      #   Fortran iparm(13) = 1: weighted matching (C index 12)
      #   Fortran iparm(24) = 1: two-level parallel factorization (C index 23)
      #   Fortran iparm(10) = 8: pivot perturbation 10^-8 (C index 9)
      #     Default is 13 (10^-13); 8 is recommended for IPM systems.
      # Note: these are set after __init__ (which calls analyze+factor),
      # so they only take effect from the second factorization onward.
      self.factorization.set_iparm(9, 8)   # pivot perturbation
      self.factorization.set_iparm(10, 1)  # scaling
      self.factorization.set_iparm(12, 1)  # matching
      self.factorization.set_iparm(23, 1)  # two-level parallel factorization
    else:
      self.factorization.refactor(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    try:
      return self.factorization.solve(rhs)
    except self.mkl_solver.PardisoError as e:
      logging.warning("PardisoError: %s", e)
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
  """Dense Cholesky solver via Gram/Schur-complement reduction.

  Instead of factorizing the full (n+m)x(n+m) KKT system, eliminates y to
  obtain an n x n SPD system solved by Cholesky (dpotrf).  Reduces
  factorization cost from O((n+m)^3) to O(n^3), a large win when m >> n
  (typical for QPs).

  **Derivation.**  The KKT system is:

      [ H   A'] [x]   [r_x]
      [ A  -D ] [y] = [r_y]

  where H = P + diag(R_x) (n x n, SPD) and D = diag(R_y) (m x m, D > 0).
  From the second row:  y = D^{-1} (A x - r_y).
  Substituting into the first row gives the Gram (normal-equations) system:

      G x = r_x + A' D^{-1} r_y,   where  G = H + A' D^{-1} A.

  G is SPD (H is SPD from regularization, A' D^{-1} A is PSD).

  **Implementation.**  The KKT diagonal mixes P's diagonal with R_x (the
  regularization), so we store P_offdiag (P with diagonal zeroed) once and
  read the combined diagonal each iteration:

      G = P_offdiag + diag(kkt_diag[:n]) + A' D^{-1} A
        = P_offdiag + diag(P_diag + R_x)  + A' D^{-1} A
        = P + diag(R_x) + A' D^{-1} A     = H + A' D^{-1} A.

  The A' D^{-1} A term is formed via dsyrk on A_scaled = diag(1/sqrt(R_y)) A
  so that A_scaled' A_scaled = A' D^{-1} A, exploiting BLAS3 symmetry.
  """

  def __init__(self):
    from scipy.linalg import lapack, blas  # pylint: disable=g-import-not-at-top

    self._dpotrf = lapack.dpotrf
    self._dpotrs = lapack.dpotrs
    self._dsyrk = blas.dsyrk
    self._dsymv = blas.dsymv
    self._n = 0
    self._m = 0

  def set_dims(self, n: int, m: int, z: int) -> None:
    self._n = n
    self._m = m
    # Blocks extracted once from the KKT scaffold (populated in first set_kkt).
    self._A: np.ndarray | None = None         # (m, n) dense, C-order
    self._P_offdiag: np.ndarray | None = None  # (n, n) P with diagonal zeroed, F-order

    # Pre-allocate all per-iteration buffers.
    self._R_x = np.empty(n, dtype=np.float64)
    self._R_y = np.empty(m, dtype=np.float64)
    self._G = np.empty((n, n), dtype=np.float64, order="F")
    self._chol = np.empty((n, n), dtype=np.float64, order="F")
    self._A_scaled = np.empty((m, n), dtype=np.float64, order="F")
    self._diag_idx = np.diag_indices(n)
    self._result = np.empty(n + m, dtype=np.float64)
    self._g = np.empty(n, dtype=np.float64)

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    n, m = self._n, self._m
    if self._A is None:
      kkt_dense = kkt.toarray()
      self._A = np.ascontiguousarray(kkt_dense[n:, :n], dtype=np.float64)
      P_block = kkt_dense[:n, :n].copy()
      np.fill_diagonal(P_block, 0.0)
      self._P_offdiag = np.asfortranarray(P_block)
    diag = kkt.diagonal()
    np.copyto(self._R_x, diag[:n])
    np.negative(diag[n:], out=self._R_y)

  def factorize(self) -> None:
    # G = P_offdiag + diag(R_x) + A' diag(1/R_y) A
    np.copyto(self._G, self._P_offdiag)
    self._G[self._diag_idx] += self._R_x
    np.multiply(self._A, (1.0 / np.sqrt(self._R_y))[:, None], out=self._A_scaled)
    # Symmetric rank-k update: G += A_scaled.T @ A_scaled via BLAS dsyrk.
    self._dsyrk(1.0, self._A_scaled, beta=1.0, c=self._G, trans=1,
                lower=1, overwrite_c=True)
    # G is theoretically SPD but the rank-k update can introduce roundoff
    # that makes it very slightly indefinite (eigenvalue ~ -1e-8) when 1/R_y
    # spans many orders of magnitude.  A tiny relative perturbation fixes
    # this; iterative refinement (which uses the exact block matvec in
    # __matmul__) corrects for any factorization-level perturbation.
    self._G[self._diag_idx] += 1e-14 * np.max(self._G[self._diag_idx])
    np.copyto(self._chol, self._G)
    self._chol, info = self._dpotrf(self._chol, lower=True, overwrite_a=True)
    if info != 0:
      raise np.linalg.LinAlgError(f"Cholesky failed (dpotrf info={info})")

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    n = self._n
    x_x, x_y = x[:n], x[n:]
    result = self._result
    # result[:n] = P_offdiag @ x_x + R_x * x_x + A.T @ x_y
    self._dsymv(1.0, self._P_offdiag, x_x, 0.0, result[:n], lower=1,
                overwrite_y=1)
    result[:n] += self._R_x * x_x
    result[:n] += self._A.T @ x_y
    result[n:] = self._A @ x_x - self._R_y * x_y
    return result

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    n = self._n
    inv_R_y = 1.0 / self._R_y
    # Reduced RHS: g = rhs_x + A' (R_y^{-1} rhs_y)
    g = self._g
    np.copyto(g, rhs[:n])
    g += self._A.T @ (inv_R_y * rhs[n:])
    x, _ = self._dpotrs(self._chol, g, lower=True)
    # Back-substitute: y = R_y^{-1} (A x - rhs_y)
    result = self._result
    result[:n] = x
    np.multiply(inv_R_y, self._A @ x - rhs[n:], out=result[n:])
    return result

  def format(self) -> Literal["csr"]:
    return "csr"


class CupyDenseSolver(LinearSolver):
  """GPU Cholesky solver via Gram/Schur-complement reduction (cupy).

  GPU counterpart of ScipyDenseSolver.  See that class's docstring for the
  Gram derivation.  Forms G = H + A' D^{-1} A on the GPU and factorizes
  with Cholesky via cupyx.scipy.linalg.cho_factor.
  """

  def __init__(self):
    import cupy  # pylint: disable=g-import-not-at-top
    import cupyx.scipy.linalg  # pylint: disable=g-import-not-at-top

    self._cp = cupy
    self._cupyx_linalg = cupyx.scipy.linalg
    self._n = 0
    self._m = 0

  def set_dims(self, n: int, m: int, z: int) -> None:
    cp = self._cp
    self._n = n
    self._m = m
    self._A_gpu = None
    self._P_offdiag_gpu = None
    self._R_x_gpu = cp.empty(n, dtype=cp.float64)
    self._R_y_gpu = cp.empty(m, dtype=cp.float64)
    self._G_gpu = cp.empty((n, n), dtype=cp.float64)
    self._diag_idx = cp.arange(n)
    self._result_gpu = cp.empty(n + m, dtype=cp.float64)
    self._L = None  # Lower-triangular Cholesky factor

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    cp = self._cp
    n, m = self._n, self._m
    if self._A_gpu is None:
      kkt_dense = kkt.toarray()
      self._A_gpu = cp.asarray(kkt_dense[n:, :n], dtype=cp.float64)
      P_block = kkt_dense[:n, :n].copy()
      np.fill_diagonal(P_block, 0.0)
      self._P_offdiag_gpu = cp.asarray(P_block, dtype=cp.float64)
    diag = kkt.diagonal()
    self._R_x_gpu.set(diag[:n])
    self._R_y_gpu.set(-diag[n:])

  def factorize(self) -> None:
    cp = self._cp
    idx = self._diag_idx
    cp.copyto(self._G_gpu, self._P_offdiag_gpu)
    self._G_gpu[idx, idx] += self._R_x_gpu
    A_scaled = self._A_gpu * (1.0 / cp.sqrt(self._R_y_gpu))[:, None]
    self._G_gpu += A_scaled.T @ A_scaled
    # Same numerical perturbation as ScipyDenseSolver.factorize.
    self._G_gpu[idx, idx] += 1e-14 * cp.max(self._G_gpu[idx, idx])
    self._L = cp.linalg.cholesky(self._G_gpu)

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    cp = self._cp
    n = self._n
    x_gpu = cp.asarray(x)
    x_x, x_y = x_gpu[:n], x_gpu[n:]
    result = self._result_gpu
    result[:n] = self._P_offdiag_gpu @ x_x + self._R_x_gpu * x_x + self._A_gpu.T @ x_y
    result[n:] = self._A_gpu @ x_x - self._R_y_gpu * x_y
    return cp.asnumpy(result)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    cp = self._cp
    n = self._n
    rhs_gpu = cp.asarray(rhs)
    inv_R_y = 1.0 / self._R_y_gpu
    g = rhs_gpu[:n] + self._A_gpu.T @ (inv_R_y * rhs_gpu[n:])
    # Solve L L' x = g via triangular solves.
    x = self._cupyx_linalg.solve_triangular(self._L, g, lower=True)
    x = self._cupyx_linalg.solve_triangular(self._L, x, lower=True, trans='C')
    result = self._result_gpu
    result[:n] = x
    cp.multiply(inv_R_y, self._A_gpu @ x - rhs_gpu[n:], out=result[n:])
    return cp.asnumpy(result)

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
    self._solver.set_dims(n=self.n, m=self.m, z=self.z)

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
