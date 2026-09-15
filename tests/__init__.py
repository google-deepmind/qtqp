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

"""Tests for the QTQP solver.

Reaching for a private member is allowed here, but only deliberately. A test
that asserts a *behaviour* of the solver goes through the public surface --
`QTQP.solve` and the `Solution` it returns -- so that refactoring the
internals cannot break it. A test that covers an *internal seam* names that
seam in its own name (`test_max_step_size`, `test_normalize_invariant`,
`test_equivalent_equilibration`) and may reach for it directly, because the
seam is the thing under test and no public surface reports it.

Where a seam needs coverage often enough to be worth supporting, the fix is
to give it a public surface rather than to keep reaching: see
`DirectKktSolver.__matmul__`, which is the true-KKT operator the refinement
residuals are measured against, promoted out of three copies of the same
expression so tests can assert against it.

The third case is a spy or a fake -- patching `_newton_step` to fail, or
passing a backend that corrupts its second solve. Those set up a condition
rather than assert on one, and the assertion that follows should still be
public.
"""
