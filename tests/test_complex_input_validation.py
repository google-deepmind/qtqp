# Copyright 2026 Google LLC
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

"""QTQP must not silently discard imaginary components from QP data."""

import numpy as np
import pytest
from scipy import sparse

import qtqp


def _problem_data():
  return {
      "a": sparse.csc_matrix([[1.0], [-1.0]]),
      "b": np.array([1.0, 0.0]),
      "c": np.array([-1.0]),
      "p": sparse.csc_matrix([[0.5]]),
      "z": 0,
  }


@pytest.mark.parametrize("name,complex_value", [
    ("a", sparse.csc_matrix([[1.0 + 2.0j], [-1.0]])),
    ("b", np.array([1.0 + 2.0j, 0.0])),
    ("c", np.array([-1.0 + 3.0j])),
    ("p", sparse.csc_matrix([[0.5 + 4.0j]])),
])
def test_complex_data_is_rejected_before_discarding_imaginary_part(
    name, complex_value
):
  data = _problem_data()
  data[name] = complex_value
  with pytest.raises(ValueError, match=name + ".*real"):
    qtqp.QTQP(**data)


def test_complex_dtype_is_not_implicitly_reinterpreted_as_real():
  data = _problem_data()
  data["c"] = np.array([-1.0 + 0.0j], dtype=np.complex128)
  with pytest.raises(ValueError, match="c.*real"):
    qtqp.QTQP(**data)


def test_real_qp_solves_as_before():
  problem = qtqp.QTQP(**_problem_data())
  result = problem.solve(
      verbose=False, linear_solver=qtqp.LinearSolver.SCIPY
  )
  assert result.status is qtqp.SolutionStatus.SOLVED
  np.testing.assert_allclose(result.x, [1.0], atol=1e-5)


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__]))
