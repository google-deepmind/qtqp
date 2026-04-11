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
"""Sparse KKT linear system solver backends."""

import logging
from typing import Literal

import numpy as np
import scipy.sparse as sp

from .direct import LinearSolver
from .direct import diag_data_indices


def _full_symmetric_from_upper(kkt: sp.spmatrix, format: Literal["csc", "csr"]) -> sp.spmatrix:
  """Reconstruct a full symmetric matrix from stored upper-triangular entries."""
  diag = sp.diags(kkt.diagonal(), format=format)
  return (kkt + kkt.T - diag).asformat(format)


class MklPardisoSolver(LinearSolver):
  """Wrapper around pymklpardiso.PardisoSolver."""

  def __init__(self):
    import pymklpardiso  # pylint: disable=g-import-not-at-top

    self._pymklpardiso = pymklpardiso
    self._solver: pymklpardiso.PardisoSolver | None = None

  def factorize(self):
    triu = self._kkt
    if self._solver is None:
      # Initial analysis is pattern-only (cheap). On error recovery we
      # escalate to value-dependent analysis via iparm[10]/iparm[12].
      #   iparm[9]  = 8: pivot perturbation 10^-8 (default 13 ie 10^-13)
      #   iparm[23] = 1: two-level parallel factorization
      self._solver = self._pymklpardiso.PardisoSolver(
          triu,
          mtype=self._pymklpardiso.MTYPE_REAL_SYM_INDEF,
          iparms={9: 8, 23: 1},
      )
    else:
      self._solver.refactor(triu.data)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    try:
      return self._solver.solve(rhs)
    except RuntimeError as e:
      # Escalate to value-dependent analysis (scaling + weighted matching)
      # and re-analyze + re-factor from scratch.
      logging.warning("PARDISO error: %s", e)
      logging.warning("Re-analyzing with value-dependent scaling/matching.")
      self._solver.set_iparm(10, 1)
      self._solver.set_iparm(12, 1)
      self._solver.factor(self._kkt.data)
      return self._solver.solve(rhs)

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
      self.factorization = self.qdldl.Solver(self._kkt, upper=True)
    else:
      self.factorization.update(self._kkt, upper=True)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class ScipySolver(LinearSolver):
  """Wrapper around scipy.linalg.factorized."""

  def __init__(self):
    self.factorization = None

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    super().set_kkt(kkt)
    self._full_kkt = _full_symmetric_from_upper(kkt, "csc")
    self._full_diag_idxs = diag_data_indices(self._full_kkt)

  def update_diag(self, diag: np.ndarray) -> None:
    super().update_diag(diag)
    self._full_kkt.data[self._full_diag_idxs] = diag

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._full_kkt @ x

  def factorize(self):
    self.factorization = sp.linalg.factorized(self._full_kkt)

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
      # Must use simplicial mode: the KKT matrix is indefinite (has negative
      # eigenvalues from the -(D+mu*I) block), so we need LDL factorization.
      # Supernodal mode only supports positive-definite Cholesky and will fail
      # with CholmodNotPositiveDefiniteError on the indefinite KKT system.
      self.factorization = self.cholmod.CholeskyFactor(
          self._kkt, supernodal_mode="simplicial", lower=False
      )
    self.factorization.factorize(self._kkt, ldl=True)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


class EigenSolver(LinearSolver):
  """Wrapper around Eigen Simplicial LDL^T."""

  def __init__(self):
    import nanoeigenpy  # pylint: disable=g-import-not-at-top

    self.nanoeigenpy = nanoeigenpy
    self._solver: nanoeigenpy.SimplicialLDLT | None = None

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    # Eigen itself supports either triangle, but nanoeigenpy's Python module
    # exposes only the default Lower-flavored SimplicialLDLT class, so adapt
    # the shared upper-triangular KKT into the lower triangle here.  The base
    # symmetric matvec works with either stored triangle, so we only keep the
    # lower-triangular view.
    super().set_kkt(kkt.T.tocsc())

  def update_diag(self, diag: np.ndarray) -> None:
    self._kkt.data[self._kkt_diag_idxs] = diag
    np.copyto(self._kkt_diag, diag)

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
      # First call: build a PETSc AIJ matrix from the stored upper triangle,
      # mark it symmetric, and tell PETSc/MUMPS to ignore the absent lower
      # triangle structurally.
      # createAIJWithArrays shares the scipy data buffer, so
      # DirectKktSolver's in-place diagonal updates are visible to PETSc
      # without any copy.  On subsequent factorize calls we just bump the
      # state counter and refactorize.
      self._mat = PETSc.Mat().createAIJWithArrays(
          kkt.shape, (kkt.indptr, kkt.indices, kkt.data)
      )
      self._mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
      self._mat.setOption(PETSc.Mat.Option.SPD, False)
      self._mat.setOption(PETSc.Mat.Option.IGNORE_LOWER_TRIANGULAR, True)
      self._mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
      self._mat.assemble()

      self._ksp = PETSc.KSP().create()
      self._ksp.setType(PETSc.KSP.Type.PREONLY)
      self._pc = self._ksp.getPC()
      # PETSc exposes MUMPS's symmetric-indefinite LDL^T path through the
      # "cholesky" factor interface. That name is a misnomer here: because
      # MAT_SPD remains false above, MUMPS treats the matrix as general
      # symmetric (quasidefinite KKT), not SPD.
      self._pc.setType(PETSc.PC.Type.CHOLESKY)
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
    # Some PETSc conda builds use complex scalars, so the solution
    # vector may be complex128.  Take the real part.
    x = self._x.array
    np.copyto(self._sol, x.real if np.iscomplexobj(x) else x)
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


class AccelerateSolver(LinearSolver):
  """Wrapper around macldlt for Apple Accelerate sparse LDL^T (macOS only)."""

  def __init__(self):
    import macldlt  # pylint: disable=g-import-not-at-top

    self._macldlt = macldlt
    self._solver = None

  def factorize(self):
    if self._solver is None:
      self._solver = self._macldlt.LDLTSolver(self._kkt, triangle="upper")
    else:
      self._solver.refactor(self._kkt.data)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self._solver.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"


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
    self._ctx.control[umfpack.UMFPACK_STRATEGY] = umfpack.UMFPACK_STRATEGY_SYMMETRIC
    self._symbolic_done = False

  def set_kkt(self, kkt: sp.spmatrix) -> None:
    super().set_kkt(kkt)
    self._full_kkt = _full_symmetric_from_upper(kkt, "csc")
    self._full_diag_idxs = diag_data_indices(self._full_kkt)

  def update_diag(self, diag: np.ndarray) -> None:
    super().update_diag(diag)
    self._full_kkt.data[self._full_diag_idxs] = diag

  def __matmul__(self, x: np.ndarray) -> np.ndarray:
    return self._full_kkt @ x

  def factorize(self):
    if not self._symbolic_done:
      self._ctx.symbolic(self._full_kkt)
      self._symbolic_done = True
    self._ctx.numeric(self._full_kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self._ctx.solve(
        self._umfpack.UMFPACK_A, self._full_kkt, rhs, autoTranspose=True
    )

  def format(self) -> Literal["csc"]:
    return "csc"
