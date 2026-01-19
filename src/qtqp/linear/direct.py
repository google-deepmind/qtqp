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
from typing import Any, Literal, Protocol

import numpy as np
import scipy.sparse as sp
import scipy
from . import LinearSolver 

class MklPardisoSolver(LinearSolver):
  """Wrapper around pydiso.mkl_solver.MKLPardisoSolver."""

  def __init__(self):
    import pydiso.mkl_solver  # pylint: disable=g-import-not-at-top

    self.module = pydiso.mkl_solver
    self.factorization: pydiso.mkl_sovler.MKLPardisoSolver | None = None

  def update(self, kkt: sp.spmatrix):
    if self.factorization is None:
      self.factorization = self.module.MKLPardisoSolver(
          kkt, matrix_type="real_symmetric_indefinite"
      )
      # Recommended iparms for IPMs from Pardiso docs.
      # These only affect the analysis step so should be set before __init__,
      # but this is not currently possible with the current interface.
      self.factorization.set_iparm(10, 1)
      self.factorization.set_iparm(12, 1)
    else:
      self.factorization.refactor(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    try:
      return self.factorization.solve(rhs)
    except self.module.PardisoError as e:
      logging.warning("PardisoError: %s", e)
      logging.warning("Performing analysis and factorization steps again.")
      self.factorization._analyze()  # pylint: disable=protected-access
      self.factorization._factor()  # pylint: disable=protected-access
      return self.factorization.solve(rhs)

  def format(self) -> Literal["csr"]:
    return "csr"

  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"


class QdldlSolver(LinearSolver):
  """Wrapper around qdldl.Solver for quasi-definite LDL factorization."""

  def __init__(self):
    import qdldl  # pylint: disable=g-import-not-at-top

    self.module = qdldl
    self.factorization: qdldl.Solver | None = None

  def update(self, kkt: sp.spmatrix):
    if self.factorization is None:
      self.factorization = self.module.Solver(kkt)
    else:
      self.factorization.update(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"

  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"


class ScipySolver(LinearSolver):
  """Wrapper around scipy.linalg.factorized."""

  def __init__(self):
    self.factorization = None

  def update(self, kkt: sp.spmatrix):
    # Use to_csc() to ensure correct format, though usually it's a cheap view.
    self.factorization = sp.linalg.factorized(kkt.tocsc())

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"

  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"


class CholModSolver(LinearSolver):
  """Wrapper around sksparse.cholmod for Cholesky LDLt factorization."""

  def __init__(self):
    from sksparse import cholmod  # pylint: disable=g-import-not-at-top

    self.module = cholmod
    self.factorization: cholmod.CholeskyFactor | None = None

  def update(self, kkt: sp.spmatrix):
    if self.factorization is None:
      self.factorization = self.module.cholesky(kkt, mode="simplicial")
    else:
      self.factorization.cholesky_inplace(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.factorization(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"

  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"


class EigenSolver(LinearSolver):
  """Wrapper around Eigen Simplicial LDL^T."""

  def __init__(self):
    import nanoeigenpy  # pylint: disable=g-import-not-at-top

    self.module = nanoeigenpy
    self.solver: nanoeigenpy.SimplicialLDLT | None = None

  def update(self, kkt: sp.spmatrix):
    if self.solver is None:
      self.solver = self.module.SimplicialLDLT()
      self.solver.analyzePattern(kkt)

    self.solver.factorize(kkt)

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    return self.solver.solve(rhs)

  def format(self) -> Literal["csc"]:
    return "csc"

  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"

class MumpsSolver(LinearSolver):
  """Wrapper for MUMPS solver (via petsc4py)."""

  def __init__(self):
    import petsc4py.PETSc  # pylint: disable=g-import-not-at-top

    self.module = petsc4py.PETSc
    self.ksp = self.module.KSP().create()

    # Configure as a direct solver (apply preconditioner only)
    self.ksp.setType(self.module.KSP.Type.PREONLY)
    self.ksp.getPC().setType(self.module.PC.Type.LU)
    self.ksp.getPC().setFactorSolverType("mumps")

    # Allow command-line customization (eg, -mat_mumps_icntl_14 20).
    self.ksp.setFromOptions()

  def update(self, kkt: sp.spmatrix):
    kkt_wrapper = self.module.Mat().createAIJ(
        size=kkt.shape, csr=(kkt.indptr, kkt.indices, kkt.data)
    )
    kkt_wrapper.setOption(self.module.Mat.Option.SYMMETRIC, True)
    kkt_wrapper.setOption(self.module.Mat.Option.SPD, False)
    kkt_wrapper.assemble()

    # Check if KSP already has a matrix defined to determine the flag
    already_factorized = self.ksp.getOperators()[0] is not None
    if already_factorized:
      flag = self.module.Mat.Structure.SAME_NONZERO_PATTERN
    else:
      flag = self.module.Mat.Structure.DIFFERENT_NONZERO_PATTERN

    try:
      self.ksp.setOperators(kkt_wrapper, kkt_wrapper, flag)
    except TypeError:
      # Fallback for older petsc4py API; usually auto-detects reuse
      self.ksp.setOperators(kkt_wrapper, kkt_wrapper)

    # Force factorization (symbolic first time, numeric every time)
    self.ksp.setUp()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    b = self.module.Vec().createWithArray(rhs.ravel())
    x = self.module.Vec().createSeq(rhs.size)
    self.ksp.solve(b, x)
    return x.getArray().real.reshape(rhs.shape)

  def format(self) -> Literal["csr"]:
    return "csr"

  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"

class CuDssSolver(LinearSolver):
  """Wrapper around Nvidia's CuDSS for GPU-accelerated solving."""

  def __init__(self):
    import nvmath.sparse  # pylint: disable=g-import-not-at-top

    self.module = nvmath.sparse
    self.solver: nvmath.sparse.advanced.DirectSolver | None = None

  def update(self, kkt: sp.spmatrix):
    if self.solver is None:
      sparse_system_type = self.module.advanced.DirectSolverMatrixType.SYMMETRIC
      # Turn off annoying logs by default.
      logger = logging.getLogger("null")
      logger.disabled = True
      options = self.module.advanced.DirectSolverOptions(
          sparse_system_type=sparse_system_type, logger=logger
      )
      # RHS must be in column major order (Fortran) for cuDSS.
      dummy_rhs = np.empty(kkt.shape[1], order="F", dtype=np.float64)
      self.solver = self.module.advanced.DirectSolver(
          kkt, dummy_rhs, options=options
      )
      self.solver.plan()
    else:
      self.solver.reset_operands(a=kkt)

    self.solver.factorize()

  def solve(self, rhs: np.ndarray) -> np.ndarray:
    # Ensure RHS is Fortran contiguous for cuDSS expected input format
    rhs_fortran = np.asfortranarray(rhs, dtype=np.float64)
    self.solver.reset_operands(b=rhs_fortran)
    return self.solver.solve()

  def format(self) -> Literal["csr"]:
    return "csr"

  def __del__(self):
    """Frees the solver resources."""
    if self.solver is not None:
      self.solver.free()
  
  def type(self) -> Literal["direct"]:
    """Returns the type of the solver."""
    return "direct"

