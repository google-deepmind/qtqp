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
"""direct.py must not import its backend modules."""

import ast
import inspect

from qtqp import direct

_BACKEND_MODULES = {"solvers_dense", "solvers_gpu", "solvers_sparse"}


def test_direct_has_no_backend_import():
  """The import cycle is gone: direct.py never pulls a backend module in.

  Every import statement in the file is checked, not only the module-level
  ones, so a lazy re-export inside a function cannot quietly reintroduce
  the dependency later.
  """
  tree = ast.parse(inspect.getsource(direct))
  imported = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
      imported.add(node.module.lstrip("."))
    elif isinstance(node, ast.Import):
      imported.update(alias.name for alias in node.names)
  assert not imported & _BACKEND_MODULES
