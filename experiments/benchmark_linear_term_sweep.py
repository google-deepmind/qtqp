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

"""Solve 100 QPs that differ only in the linear term, and time the reuse.

A frontier sweep, a rolling rebalance and a scenario grid all solve one problem
many times with a slightly different linear term while `p`, `a`, `b` and `z`
stay put. Both libraries have a mechanism for exactly that, and they are not the
same mechanism, which is what makes the comparison interesting:

*   QTQP takes `warm_start=(x, y, s)` from a nearby problem. It screens the
    point -- four positivity floors, six matrix-vector products, no
    factorisation -- and accepts it when the guarded local score comes in under
    `warm_start_threshold`. What is saved is interior point iterations; every
    solve still equilibrates and factorises.

*   cvx-quadprog's `Sweep` caches the factorisation itself. `J` depends only on
    `G` and `R` only on `G` and the active set, so both survive a change of
    linear term verbatim. When the cached active set still satisfies the KKT
    conditions the answer is recovered in `O(n^2)` with no factorisation and no
    iteration at all; when it does not, the set is repaired rather than
    abandoned.

So QTQP amortises iterations and `Sweep` amortises the factorisation. Which one
wins should depend on how near consecutive problems actually are, and that is
the axis this script sweeps: `path` walks the linear term along a ray, `jitter`
perturbs it by 1% independently each step, and `independent` redraws it from
scratch. The last one is the control -- nothing is being reused, so any
mechanism that still claims a speed-up there is measuring something else.

Both mechanisms report whether the reuse actually took, and the script reports
it back as a hit rate: `warm_accepted` for QTQP, and for `Sweep` an
`iterations` of `(0, 0)`, which is how it says the cached factorisation was
used outright.

Correctness is not assumed. Every one of the 100 solves is compared across all
four configurations, and the largest relative objective disagreement is
reported, so a configuration cannot look fast by returning something else.

Usage:
    python experiments/benchmark_linear_term_sweep.py
    python experiments/benchmark_linear_term_sweep.py --n 400 --steps 100
    python experiments/benchmark_linear_term_sweep.py --family box --csv out.csv
"""

import argparse
import time

import numpy as np
from scipy import sparse

import qtqp

try:
  from cvx.quadprog import Sweep, solve_qp
except ImportError:  # pragma: no cover - the script is not part of the suite.
  raise SystemExit(
      "cvx-quadprog is required: pip install 'qtqp[test]'"
  ) from None


# ---------------------------------------------------------------------------
# The fixed part of the problem, and the sequence of linear terms.
# ---------------------------------------------------------------------------


def portfolio_base(n, seed=0):
  """Long-only capped minimum-variance: `sum(x) == 1`, `0 <= x <= 2/n`."""
  rng = np.random.default_rng(seed)
  loadings = rng.standard_normal((n, 8))
  p = loadings @ loadings.T + 0.5 * np.eye(n)
  a = sparse.vstack(
      [sparse.csc_matrix(np.ones((1, n))), sparse.eye(n), -sparse.eye(n)],
      format="csc",
  )
  b = np.concatenate([[1.0], np.full(n, 2.0 / n), np.zeros(n)])
  c0 = -rng.uniform(0.0, 0.1, n)
  return sparse.csc_matrix(p), a, b, c0, 1


def box_base(n, seed=0):
  """Dense `p` with box constraints tight enough to bind on about half."""
  rng = np.random.default_rng(seed)
  factor = rng.standard_normal((n, n)) / np.sqrt(n)
  p = factor @ factor.T + 0.05 * np.eye(n)
  c0 = rng.standard_normal(n)
  bound = 0.25 * float(np.abs(np.linalg.solve(p, -c0)).mean())
  a = sparse.vstack([sparse.eye(n), -sparse.eye(n)], format="csc")
  b = np.full(2 * n, bound)
  return sparse.csc_matrix(p), a, b, c0, 0


BASES = {"portfolio": portfolio_base, "box": box_base}


def linear_terms(c0, steps, regime, seed=1):
  """Return the sequence of `steps` linear terms to solve for.

  Args:
    c0: The base linear term.
    steps: How many problems in the sequence.
    regime: `path` walks along a fixed ray, so consecutive problems are as
      close as a sweep ever gets; `jitter` applies an independent 1% relative
      perturbation, which is the perturbation size `Sweep`'s own documentation
      quotes; `independent` redraws the term from scratch and is the control.
    seed: Seed for the perturbations.

  Returns:
    A list of `steps` linear-term vectors.
  """
  rng = np.random.default_rng(seed)
  n = c0.shape[0]
  scale = float(np.abs(c0).mean())

  if regime == "path":
    # A ray, traversed in `steps` equal increments totalling 20% of the base
    # term's scale -- an efficient-frontier sweep, in effect.
    direction = rng.standard_normal(n)
    direction *= scale / float(np.abs(direction).mean())
    return [c0 + 0.2 * (k / max(1, steps - 1)) * direction
            for k in range(steps)]
  if regime == "jitter":
    return [c0 * (1.0 + 0.01 * rng.standard_normal(n)) for _ in range(steps)]
  if regime == "independent":
    return [-rng.uniform(0.0, 2.0 * scale, n) for _ in range(steps)]
  raise ValueError(f"Unknown regime: {regime}")


REGIMES = ("path", "jitter", "independent")


# ---------------------------------------------------------------------------
# The four configurations. Each consumes the whole sequence and returns
# (objectives, seconds, hits, detail), where `hits` counts the solves whose
# reuse mechanism actually engaged.
# ---------------------------------------------------------------------------


def _objective(p, c, x):
  return 0.5 * x @ (p @ x) + c @ x


def run_qtqp(problem, terms, warm):
  """Solve the sequence with QTQP, optionally chaining the warm start.

  A fresh `QTQP` is constructed per problem because `c` is a constructor
  argument. That construction is validation and presolve, not a
  factorisation, and it is timed separately so its share is visible rather
  than folded into the comparison.
  """
  p, a, b, _, z = problem
  objectives, iterations, hits = [], [], 0
  construct_seconds = 0.0
  previous = None

  start = time.perf_counter()
  for c in terms:
    tick = time.perf_counter()
    solver = qtqp.QTQP(p=p, a=a, b=b, c=c, z=z)
    construct_seconds += time.perf_counter() - tick

    kwargs = {"warm_start": previous} if (warm and previous is not None) else {}
    solution = solver.solve(verbose=False, **kwargs)
    if solution.status != qtqp.SolutionStatus.SOLVED:
      raise RuntimeError(f"QTQP returned {solution.status.name}")
    if warm:
      if getattr(solver, "warm_accepted", False):
        hits += 1
      previous = (solution.x, solution.y, solution.s)
    iterations.append(solution.iterations)
    objectives.append(_objective(p, c, solution.x))
  seconds = time.perf_counter() - start

  return objectives, seconds, hits, {
      "construct_ms": construct_seconds / len(terms) * 1e3,
      "mean_iters": float(np.mean(iterations)),
  }


def run_goldfarb_idnani(problem, terms, sweep):
  """Solve the sequence with cvx-quadprog, optionally reusing the factors.

  The dense conversion is outside the timed region here, unlike in
  `benchmark_cvx_quadprog.py`: it depends only on `p` and `a`, which do not
  change across the sequence, so a caller doing this for real converts once.
  Charging it 100 times would measure a mistake rather than the mechanism.
  """
  p, a, b, _, z = problem
  dense_g = p.toarray()
  dense_c = -a.toarray().T
  neg_b = -b

  objectives, iterations, hits = [], [], 0
  start = time.perf_counter()
  if sweep:
    sweeper = Sweep(dense_g, dense_c, neg_b, meq=z)
    for c in terms:
      solution = sweeper.solve(-c)
      # `Sweep` reports (0, 0) exactly when the cached factorisation answered
      # without any active-set iteration.
      if not solution.iterations.any():
        hits += 1
      iterations.append(solution.iterations.sum())
      objectives.append(_objective(p, c, solution.x))
  else:
    for c in terms:
      solution = solve_qp(G=dense_g, a=-c, C=dense_c, b=neg_b, meq=z)
      iterations.append(solution.iterations.sum())
      objectives.append(_objective(p, c, solution.x))
  seconds = time.perf_counter() - start

  return objectives, seconds, hits, {"mean_iters": float(np.mean(iterations))}


CONFIGS = {
    "qtqp_cold": lambda problem, terms: run_qtqp(problem, terms, warm=False),
    "qtqp_warm": lambda problem, terms: run_qtqp(problem, terms, warm=True),
    "gi_cold": lambda problem, terms: run_goldfarb_idnani(
        problem, terms, sweep=False),
    "gi_sweep": lambda problem, terms: run_goldfarb_idnani(
        problem, terms, sweep=True),
}


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


def _max_relative_gap(objectives, reference):
  """Return the largest relative objective disagreement over the sequence."""
  worst = 0.0
  for value, base in zip(objectives, reference):
    worst = max(worst, abs(value - base) / max(1.0, abs(base)))
  return worst


def benchmark(family, n, steps, regimes, seed):
  """Run every configuration over every regime and return the rows."""
  base = BASES[family](n, seed=seed)
  # `c` comes from the sequence, so the fixed part carries a placeholder.
  problem = (base[0], base[1], base[2], None, base[4])
  c0 = base[3]

  rows = []
  for regime in regimes:
    terms = linear_terms(c0, steps, regime, seed=seed + 1)
    reference = None
    print(f"\n{family} n={n} {regime}, {len(terms)} solves:")

    for name, run in CONFIGS.items():
      objectives, seconds, hits, detail = run(problem, terms)
      if reference is None:
        reference = objectives
      gap = _max_relative_gap(objectives, reference)
      rows.append(dict(
          family=family, n=n, regime=regime, steps=len(terms), config=name,
          total_s=seconds, per_solve_ms=seconds / len(terms) * 1e3,
          hit_rate=hits / len(terms), max_rel_dobj=gap,
          construct_ms=detail.get("construct_ms", 0.0),
          mean_iters=detail.get("mean_iters", float("nan")),
      ))
      hit = f" hits={hits / len(terms):.0%}" if hits else ""
      print(f"  {name:<11} {seconds:7.3f}s total"
            f"  {seconds / len(terms) * 1e3:8.2f} ms/solve"
            f"  iters={detail.get('mean_iters', float('nan')):5.1f}{hit}"
            f"   (max rel dobj {gap:.1e})")
  return rows


def _print_summary(rows):
  """Print one table per regime: ms per solve and speed-up over the cold run."""
  regimes = sorted({r["regime"] for r in rows}, key=REGIMES.index)
  configs = list(CONFIGS)
  print("\n=== ms per solve, and speed-up over the same library's cold run ===")
  header = f"{'regime':>13}" + "".join(f"{c:>13}" for c in configs)
  print(header)
  print("-" * len(header))
  for regime in regimes:
    here = {r["config"]: r for r in rows if r["regime"] == regime}
    line = f"{regime:>13}"
    for config in configs:
      row = here.get(config)
      line += f"{row['per_solve_ms']:>13.2f}" if row else f"{'-':>13}"
    print(line)

    speedups = f"{'':>13}"
    for config in configs:
      row, cold = here.get(config), here.get(
          "qtqp_cold" if config.startswith("qtqp") else "gi_cold")
      if row and cold and row is not cold:
        speedups += f"{cold['per_solve_ms'] / row['per_solve_ms']:>12.2f}x"
      else:
        speedups += f"{'(base)' if row else '-':>13}"
    print(speedups)

  print("\n=== mean iterations per solve "
        "(QTQP: IPM steps; gi: active-set adds + removes) ===")
  header = f"{'regime':>13}" + "".join(f"{c:>13}" for c in configs)
  print(header)
  print("-" * len(header))
  for regime in regimes:
    here = {r["config"]: r for r in rows if r["regime"] == regime}
    line = f"{regime:>13}"
    for config in configs:
      row = here.get(config)
      line += f"{row['mean_iters']:>13.1f}" if row else f"{'-':>13}"
    print(line)

  print("\n=== reuse hit rate (cached factorisation used outright) ===")
  header = f"{'regime':>13}" + "".join(
      f"{c:>13}" for c in ("qtqp_warm", "gi_sweep"))
  print(header)
  print("-" * len(header))
  for regime in regimes:
    here = {r["config"]: r for r in rows if r["regime"] == regime}
    line = f"{regime:>13}"
    for config in ("qtqp_warm", "gi_sweep"):
      row = here.get(config)
      line += f"{row['hit_rate']:>12.0%} " if row else f"{'-':>13}"
    print(line)


def main():
  parser = argparse.ArgumentParser(
      description=__doc__.splitlines()[0],
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--family", default="portfolio",
                      choices=sorted(BASES),
                      help="Which fixed problem the sequence perturbs.")
  parser.add_argument("--n", type=int, default=200,
                      help="Problem dimension.")
  parser.add_argument("--steps", type=int, default=100,
                      help="How many problems in the sequence.")
  parser.add_argument("--regimes", default=",".join(REGIMES),
                      help=f"Comma-separated subset of: {', '.join(REGIMES)}")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--csv", help="Also write the raw rows to this path.")
  args = parser.parse_args()

  regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
  unknown = set(regimes) - set(REGIMES)
  if unknown:
    raise SystemExit(f"Unknown regimes: {', '.join(sorted(unknown))}")

  rows = benchmark(args.family, args.n, args.steps, regimes, args.seed)
  _print_summary(rows)

  if args.csv:
    import csv
    with open(args.csv, "w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
  main()
