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

"""Public solver option checks also run under python -O and -OO."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
from scipy import sparse

import qtqp


_INVALID_OPTIONS = [
    ("tol_feas", -1.), ("tol_gap_abs", -1.), ("tol_gap_rel", -1.),
    ("tol_infeas_abs", -1.), ("tol_infeas_rel", -1.),
    ("certificate_ktratio", 0.), ("max_iter", 0),
    ("step_size_scale", 0.), ("step_size_scale", 1.),
    ("min_static_regularization", -1.),
    ("max_iterative_refinement_steps", 0),
    ("linear_solver_atol", -1.), ("linear_solver_rtol", -1.),
    ("tol_feas", float("nan")), ("step_size_scale", float("nan")),
]


def _problem():
  return qtqp.QTQP(a=sparse.csc_matrix([[1.], [-1.]]),
                   b=np.array([1., 0.]), c=np.array([-1.]), z=0)


@pytest.mark.parametrize("name,value", _INVALID_OPTIONS)
def test_invalid_options_fail_before_backend_creation(name, value):
  problem = _problem()
  with mock.patch.object(qtqp, "_resolve_linear_solver") as resolve:
    with pytest.raises(ValueError, match=name):
      problem.solve(verbose=False, **{name: value})
    resolve.assert_not_called()


@pytest.mark.parametrize("optimization", [0, 1, 2])
def test_validation_survives_interpreter_optimization(optimization, tmp_path):
  script = """
import json
import sys
import numpy as np
from scipy import sparse
import qtqp

if sys.flags.optimize != int(sys.argv[1]):
    raise RuntimeError("Wrong optimization mode")
problem = qtqp.QTQP(a=sparse.csc_matrix([[1.], [-1.]]),
                    b=np.array([1., 0.]), c=np.array([-1.]), z=0)
failures = []
for name, value in json.loads(sys.argv[2]):
    try:
        problem.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY,
                      **{name: value})
    except ValueError as error:
        if name not in str(error):
            failures.append((name, str(error)))
    except Exception as error:
        failures.append((name, type(error).__name__))
    else:
        failures.append((name, "accepted invalid option"))
if failures:
    raise RuntimeError(failures)
solution = problem.solve(verbose=False, linear_solver=qtqp.LinearSolver.SCIPY)
if solution.status is not qtqp.SolutionStatus.SOLVED:
    raise RuntimeError("Valid solve failed after rejected options")
np.testing.assert_allclose(solution.x, [1.], atol=1e-7)
"""
  env = dict(os.environ, PYTHONOPTIMIZE=str(optimization))
  package_root = str(Path(qtqp.__file__).resolve().parent.parent)
  env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
  result = subprocess.run(
      [sys.executable, "-c", script, str(optimization), json.dumps(_INVALID_OPTIONS)],
      cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
  )
  assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name,value", [
    ("tol_feas", 0.), ("tol_gap_abs", 0.), ("tol_gap_rel", 0.),
    ("tol_infeas_abs", 0.), ("tol_infeas_rel", 0.),
    ("certificate_ktratio", 1.), ("max_iter", 1),
    ("min_static_regularization", 0.),
    ("max_iterative_refinement_steps", 1),
    ("linear_solver_atol", 0.), ("linear_solver_rtol", 0.),
])
def test_existing_allowed_boundaries_are_preserved(name, value):
  solution = _problem().solve(
      verbose=False, linear_solver=qtqp.LinearSolver.SCIPY, **{name: value}
  )
  assert isinstance(solution, qtqp.Solution)
  assert np.all(np.isfinite(solution.x))
