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
"""Linear solvers for KKT systems."""

from typing import Literal, Protocol

import numpy as np
import scipy.sparse as sp


class LinearSolver(Protocol):
  """Protocol defining the interface for linear solvers."""

  def update(self, kkt: sp.spmatrix) -> None:
    """Factorizes or refactorizes the KKT matrix."""
    ...

  def solve(
      self, rhs: np.ndarray, warm_start: np.ndarray | None = None
  ) -> np.ndarray | tuple[np.ndarray, int]:
    """Solves the linear system."""
    ...

  def format(self) -> str:
    """Returns the expected sparse matrix format (eg, 'csc' or 'csr')."""
    ...

  def type(self) -> Literal["direct", "indirect"]:
    """Returns solver type: 'direct' (needs refinement) or 'indirect'."""
    ...


# Import concrete implementations for convenient access
from qtqp.linear.direct import (
    CholModSolver,
    CuDssSolver,
    EigenSolver,
    MklPardisoSolver,
    MumpsSolver,
    QdldlSolver,
    ScipySolver,
)
from qtqp.linear.indirect import PetscGMRES
