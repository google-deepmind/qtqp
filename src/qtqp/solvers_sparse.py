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


class MklPardisoSolver(LinearSolver):
  """Wrapper around pydiso.mkl_solver.MKLPardisoSolver."""

  def __init__(self):
    import pydiso.mkl_solver  # pylint: disable=g-import-not-at-top

    self.mkl_solver = pydiso.mkl_solver
    self.factorization: pydiso.mkl_solver.MKLPardisoSolver | None = None

  def factorize(self):
    if self.factorization is None:
      # Subclass to intercept the start of the analysis phase
      class HackedMKLPardisoSolver(self.mkl_solver.MKLPardisoSolver):
        def _analyze(self):
          # pydiso uses 0-based C indexing: set_iparm(i, v) sets C iparm[i],
          # which is Fortran iparm(i+1).  Intel recommends for symmetric
          # indefinite IPM/saddle-point systems:
          #   Fortran iparm(10) = 8: pivot perturbation 10^-8 (C index 9)
          #     Default is 13 (10^-13); 8 is recommended for IPM systems.
          #   Fortran iparm(11) = 1: scaling (C index 10)
          #   Fortran iparm(13) = 1: weighted matching (C index 12)
          #   Fortran iparm(24) = 1: two-level parallel factorization (C index 23)
          # 1. Inject parameters. The Cython object is created, memory is
          # allocated, but the heavy math hasn't started yet.
          self.set_iparm(9, 8)   # pivot perturbation
          self.set_iparm(10, 1)  # scaling
          self.set_iparm(12, 1)  # matching
          self.set_iparm(23, 1)  # two-level parallel factorization

          # 2. Proceed with the actual analysis using the new parameters
          super()._analyze()

      # When this instantiates, it will automatically call our overridden _analyze
      self.factorization = HackedMKLPardisoSolver(
          self._kkt, matrix_type="real_symmetric_indefinite"
      )
    else:
      self.factorization.refactor(self._kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    try:
      return self.factorization.solve(rhs)
    except self.mkl_solver.PardisoError as e:
      # We tried loosening iparm(10) (pivot perturbation) on retry, but
      # even 10^-2 still triggered zero-pivot errors on some unbounded
      # problems.  A plain re-analyze + re-factor is more reliable.
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
