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

"""Runtime of QTQP against cvx-quadprog across problem families and sizes.

QTQP is a primal-dual interior point method on sparse KKT systems; cvx-quadprog
implements the Goldfarb/Idnani dual active-set method on dense matrices. The two
have different asymptotics, so "which is faster" has no single answer -- it
depends on the size, on how much of the constraint set ends up active, and on
whether the data is sparse. This script measures where each one wins.

Correctness is cross-checked in `tests/test_cvx_quadprog.py`; here it is only
re-verified per measurement, so that a fast wrong answer cannot be reported as
a fast one.

Four things to keep in mind when reading the numbers.

1.  cvx-quadprog takes dense `G` and `C`. On a sparse problem the dense
    conversion is not overhead the benchmark adds -- it is what using the
    solver costs -- so it is timed as part of the solve. This dominates the
    sparse families and is the honest reason for the gap there.

2.  The active-set method's iteration count grows with the size of the active
    set. A problem whose unconstrained minimum is already feasible is solved in
    one iteration and tells us nothing, so every family here is calibrated to
    put a substantial fraction of the constraints on their bounds; `nact` in the
    output reports how many.

3.  The two stop on different criteria. QTQP stops at `tol_feas = 1e-8`;
    Goldfarb/Idnani is an exact method and stops on the active set, reaching
    machine precision. QTQP is therefore being asked for less accuracy than it
    delivers, which flatters it slightly. `rel dobj` in the output is the
    relative objective disagreement against whichever solver answered first,
    so the reader can see what accuracy each is being paid for. A family whose
    `rel dobj` is not near machine precision is not comparing equal-accuracy
    solves and its timings should be distrusted.

4.  Both are ultimately calling into BLAS, so the numbers move with the BLAS
    build and its thread count. Medians over repetitions damp the noise but not
    a systematic difference between machines.

Usage:
    python experiments/benchmark_cvx_quadprog.py
    python experiments/benchmark_cvx_quadprog.py --families portfolio \
        --sizes 100,200,400
    python experiments/benchmark_cvx_quadprog.py --csv results.csv --reps 5
"""

import argparse
import statistics
import time

import numpy as np
from scipy import sparse

import qtqp

try:
  from cvx.quadprog import solve_qp
except ImportError:  # pragma: no cover - the script is not part of the suite.
  raise SystemExit(
      "cvx-quadprog is required: pip install 'qtqp[test]'"
  ) from None


# ---------------------------------------------------------------------------
# Problem families.
#
# Each returns (p, a, b, c, z) in QTQP's convention -- p and a sparse, so the
# sparse families stay sparse until a solver that needs dense asks for it.
#
# Every family is calibrated so that a large part of the constraint set binds;
# see note 2 in the module docstring for why that matters.
# ---------------------------------------------------------------------------


def _tight_bound(p, c, fraction):
  """Return a symmetric bound that a `fraction` of coordinates violate.

  The bound is set from the unconstrained minimiser's own scale, so it stays
  meaningful as `n` grows and the family does not silently drift towards
  "no constraints active" or "every constraint active".

  Args:
    p: The QP matrix.
    c: The cost vector.
    fraction: Multiple of the mean absolute unconstrained minimiser to bound at.

  Returns:
    The scalar bound.
  """
  if sparse.issparse(p):
    unconstrained = sparse.linalg.spsolve(p.tocsc(), -c)
  else:
    unconstrained = np.linalg.solve(p, -c)
  return fraction * float(np.abs(unconstrained).mean())


def box_dense(n, seed=0):
  """Dense QP with box constraints: the active-set method's home ground."""
  rng = np.random.default_rng(seed)
  factor = rng.standard_normal((n, n)) / np.sqrt(n)
  p = factor @ factor.T + 0.05 * np.eye(n)
  c = rng.standard_normal(n)
  bound = _tight_bound(p, c, 0.25)
  a = sparse.vstack(
      [sparse.eye(n), -sparse.eye(n)], format="csc"
  )
  b = np.full(2 * n, bound)
  return sparse.csc_matrix(p), a, b, c, 0


def portfolio(n, seed=0):
  """Long-only capped minimum-variance: dense factor covariance, sparse rows.

  The cap is `2/n`, so about half the positions sit on it and the active set
  is nearly the whole problem -- the worst case for an active-set walk.
  """
  rng = np.random.default_rng(seed)
  loadings = rng.standard_normal((n, 8))
  p = loadings @ loadings.T + 0.5 * np.eye(n)
  c = -rng.uniform(0.0, 0.1, n)
  a = sparse.vstack(
      [sparse.csc_matrix(np.ones((1, n))), sparse.eye(n), -sparse.eye(n)],
      format="csc",
  )
  b = np.concatenate([[1.0], np.full(n, 2.0 / n), np.zeros(n)])
  return sparse.csc_matrix(p), a, b, c, 1


def random_ineq_dense(n, seed=0):
  """Dense QP with dense general inequality rows, `m = 2n`.

  The right-hand side is built from an interior point `x0`, which keeps the
  problem feasible at every `n`: tightening rows towards the unconstrained
  minimum instead -- the obvious way to force an active set -- makes `2n`
  random half-spaces in `R^n` inconsistent as `n` grows, and both solvers
  then correctly refuse the problem rather than timing it.

  The slack is small and random, so the feasible region is a thin shell
  around `x0` and the optimum pushes against a large share of the rows.
  """
  rng = np.random.default_rng(seed)
  factor = rng.standard_normal((n, n)) / np.sqrt(n)
  p = factor @ factor.T + 0.05 * np.eye(n)
  c = rng.standard_normal(n)
  a = rng.standard_normal((2 * n, n))
  x0 = rng.standard_normal(n) / np.sqrt(n)
  b = a @ x0 + rng.uniform(0.0, 0.05, 2 * n)
  return sparse.csc_matrix(p), sparse.csc_matrix(a), b, c, 0


def band_sparse(n, seed=0):
  """Tridiagonal P with box constraints: `nnz(P) = 3n - 2`.

  This is the regime QTQP's sparse KKT factorisation exists for, and the one
  where cvx-quadprog has to materialise an `n x n` dense `G` to start:
  200MB of it at `n = 5000`.
  """
  rng = np.random.default_rng(seed)
  diagonal = rng.uniform(1.0, 2.0, n)
  off = -0.4 * np.ones(n - 1)
  p = sparse.diags(
      [diagonal, off, off], [0, -1, 1], format="csc"
  )
  c = rng.standard_normal(n)
  bound = _tight_bound(p, c, 0.3)
  a = sparse.vstack([sparse.eye(n), -sparse.eye(n)], format="csc")
  b = np.full(2 * n, bound)
  return p, a, b, c, 0


FAMILIES = {
    "box_dense": box_dense,
    "portfolio": portfolio,
    "random_ineq": random_ineq_dense,
    "band_sparse": band_sparse,
}

DENSE_FAMILIES = frozenset({"box_dense", "portfolio", "random_ineq"})


# ---------------------------------------------------------------------------
# The solvers under test, each a callable (problem) -> (x, objective).
# ---------------------------------------------------------------------------


def _objective(p, c, x):
  return 0.5 * x @ (p @ x) + c @ x


def _relative_gap(objective, reference):
  """Return the objective disagreement, relative where that is meaningful.

  Reported relative rather than absolute so one number stays readable across
  families whose objectives differ by orders of magnitude -- the portfolio's
  is ~1e-2, the dense box's ~1e2.
  """
  return abs(objective - reference) / max(1.0, abs(reference))


def _run_qtqp(problem, linear_solver):
  """Construct and solve with QTQP.

  Setup is inside the timed region on purpose: `solve_qp` does its own setup
  per call, so charging QTQP only for `solve()` would compare a warm solver
  against a cold one.
  """
  p, a, b, c, z = problem
  solution = qtqp.QTQP(p=p, a=a, b=b, c=c, z=z).solve(
      verbose=False, linear_solver=linear_solver
  )
  if solution.status != qtqp.SolutionStatus.SOLVED:
    raise RuntimeError(f"QTQP returned {solution.status.name}")
  return solution.x, _objective(p, c, solution.x)


def _run_goldfarb_idnani(problem, fast):
  """Densify and solve with cvx-quadprog.

  `toarray` is timed with the solve: it is the cost of handing this solver a
  problem QTQP holds sparse, not measurement overhead.
  """
  p, a, b, c, z = problem
  solution = solve_qp(
      G=p.toarray(), a=-c, C=-a.toarray().T, b=-b, meq=z, fast=fast
  )
  return solution.x, _objective(p, c, solution.x)


SOLVERS = {
    "qtqp": lambda problem: _run_qtqp(problem, qtqp.LinearSolver.AUTO),
    "qtqp_dense": lambda problem: _run_qtqp(
        problem, qtqp.LinearSolver.SCIPY_DENSE
    ),
    "gi": lambda problem: _run_goldfarb_idnani(problem, fast=False),
    "gi_fast": lambda problem: _run_goldfarb_idnani(problem, fast=True),
}


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


def _time_solver(fn, problem, reps):
  """Return (median seconds, result) over `reps` timed runs after a warmup.

  The median rather than the minimum: the question is what a caller typically
  waits for, not what the machine can be coaxed into once.
  """
  fn(problem)  # Warm up caches, imports and any first-call factory work.
  timings = []
  for _ in range(reps):
    start = time.perf_counter()
    result = fn(problem)
    timings.append(time.perf_counter() - start)
  return statistics.median(timings), result


def _active_count(problem, x):
  """Return how many constraints bind at `x`, for context.

  Counted from a solution the benchmark already has rather than from a fresh
  `solve_qp` call, whose dense factorisation on the large sparse families
  would cost more than the measurements it annotates. The tolerance is loose
  because an interior point method approaches its bounds rather than landing
  on them: a coordinate parked at a bound sits within `tol_feas` of it, while
  a genuinely interior one is orders of magnitude away.
  """
  _, a, b, _, _ = problem
  return int(np.sum(np.abs(a @ x - b) < 1e-6))


def benchmark(families, sizes, reps, budget):
  """Time every solver on every family and size, and return the rows.

  Args:
    families: Names of the problem families to run.
    sizes: Problem dimensions `n` to run at.
    reps: Timed repetitions per measurement.
    budget: Seconds above which a solver is dropped from larger sizes of the
      same family. Keeps a superlinear solver from dominating the wall clock
      once its verdict is already clear.

  Returns:
    A list of row dicts, one per (family, size, solver) measurement.
  """
  rows = []
  for family in families:
    generate = FAMILIES[family]
    retired = set()
    print(f"\n{family}:")
    for n in sizes:
      problem = generate(n)
      a = problem[1]
      nact = None
      reference = None

      for name, fn in SOLVERS.items():
        if name in retired:
          continue
        # qtqp and qtqp_dense are the same algorithm on the same data; running
        # both on a sparse family measures only the backend choice, which the
        # dense families already cover.
        if name == "qtqp_dense" and family not in DENSE_FAMILIES:
          continue

        try:
          seconds, (x, objective) = _time_solver(fn, problem, reps)
        except (RuntimeError, ValueError, MemoryError) as e:
          rows.append(dict(
              family=family, n=n, m=a.shape[0], nact=nact or -1, solver=name,
              seconds=float("nan"), dobj=float("nan"), note=type(e).__name__,
          ))
          continue

        # The first solver to answer sets the reference objective, so `dobj`
        # reads as disagreement with it rather than as anyone's error.
        if reference is None:
          reference = objective
          nact = _active_count(problem, x)
        rows.append(dict(
            family=family, n=n, m=a.shape[0], nact=nact, solver=name,
            seconds=seconds, dobj=_relative_gap(objective, reference), note="",
        ))
        if seconds > budget:
          retired.add(name)

      _report_size(rows, family, n)
  return rows


def _report_size(rows, family, n):
  """Print the rows for one (family, size) as soon as they are measured."""
  here = [r for r in rows if r["family"] == family and r["n"] == n]
  if not here:
    return
  head = here[0]
  print(f"  n={head['n']:<6d} m={head['m']:<7d} nact={head['nact']:<7d}", end="")
  for row in here:
    if row["note"]:
      cell = f"{row['note']}"
    else:
      cell = f"{row['seconds'] * 1e3:.1f}ms"
    print(f"  {row['solver']}={cell}", end="")
  worst = max((r["dobj"] for r in here if r["dobj"] == r["dobj"]), default=0.0)
  print(f"   (max rel dobj {worst:.1e})")


def _print_summary(rows, sizes):
  """Print one table per family: milliseconds by solver and size."""
  families = sorted({r["family"] for r in rows}, key=lambda f: list(FAMILIES).index(f))
  for family in families:
    solvers = [s for s in SOLVERS if any(
        r["solver"] == s and r["family"] == family for r in rows)]
    print(f"\n=== {family} (median ms over reps) ===")
    header = f"{'n':>7} {'m':>8} {'nact':>7}" + "".join(
        f"{s:>13}" for s in solvers) + f"{'best':>13}"
    print(header)
    print("-" * len(header))
    for n in sizes:
      here = {r["solver"]: r for r in rows
              if r["family"] == family and r["n"] == n}
      if not here:
        continue
      any_row = next(iter(here.values()))
      line = f"{n:>7} {any_row['m']:>8} {any_row['nact']:>7}"
      timed = {s: r["seconds"] for s, r in here.items()
               if r["seconds"] == r["seconds"]}
      for s in solvers:
        row = here.get(s)
        if row is None:
          line += f"{'-':>13}"
        elif row["note"]:
          line += f"{row['note']:>13}"
        else:
          line += f"{row['seconds'] * 1e3:>13.1f}"
      if timed:
        best = min(timed, key=timed.get)
        runner = sorted(timed.values())
        ratio = runner[1] / runner[0] if len(runner) > 1 else 1.0
        line += f"{best + f' {ratio:.1f}x':>13}"
      print(line)


def main():
  parser = argparse.ArgumentParser(
      description=__doc__.splitlines()[0],
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--families", default=",".join(FAMILIES),
      help=f"Comma-separated subset of: {', '.join(FAMILIES)}",
  )
  parser.add_argument(
      "--sizes", default="50,100,200,400,800",
      help="Comma-separated problem dimensions n.",
  )
  parser.add_argument("--reps", type=int, default=3,
                      help="Timed repetitions per measurement (median taken).")
  parser.add_argument("--budget", type=float, default=2.0,
                      help="Seconds above which a solver is dropped from "
                           "larger sizes of the same family.")
  parser.add_argument("--csv", help="Also write the raw rows to this path.")
  args = parser.parse_args()

  families = [f.strip() for f in args.families.split(",") if f.strip()]
  unknown = set(families) - set(FAMILIES)
  if unknown:
    raise SystemExit(f"Unknown families: {', '.join(sorted(unknown))}")
  sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

  print(f"reps={args.reps}, budget={args.budget}s, "
        f"sizes={','.join(str(s) for s in sizes)}")
  rows = benchmark(families, sizes, args.reps, args.budget)

  _print_summary(rows, sizes)

  if args.csv:
    import csv
    with open(args.csv, "w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
  main()
