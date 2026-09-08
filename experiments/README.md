# Experiments

Scripts here measure QTQP rather than test it. They are not part of the test
suite, are not run in CI, and their numbers are specific to the machine that
produced them.

## `benchmark_cvx_quadprog.py`

Runtime of QTQP against [cvx-quadprog](https://github.com/jebel-quant/quadprog),
an independent implementation of the Goldfarb/Idnani dual active-set method.
Correctness of the same pairing is covered by `tests/test_cvx_quadprog.py`.

```bash
python experiments/benchmark_cvx_quadprog.py
python experiments/benchmark_cvx_quadprog.py --families band_sparse \
    --sizes 1200,2000,5000
python experiments/benchmark_cvx_quadprog.py --csv results.csv --reps 5
```

Four solvers are timed: `qtqp` (the `AUTO` backend, sparse KKT), `qtqp_dense`
(`SCIPY_DENSE`), `gi` (`solve_qp`), and `gi_fast` (`solve_qp(fast=True)`).
Setup is inside the timed region for both, and the dense conversion
cvx-quadprog requires is charged to it, because that is what calling it costs.

### Results

macOS arm64, Accelerate, Python 3.14, qtqp 0.0.7, cvx-quadprog 0.4.2, medians
of 3. Milliseconds; `nact` is how many of the `m` constraints bind.

`box_dense` — dense `P`, box constraints:

| n | m | nact | qtqp | qtqp_dense | gi | gi_fast |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 200 | 53 | 7.9 | 5.2 | 1.1 | **0.4** |
| 400 | 800 | 190 | 48.2 | 22.3 | 11.8 | **2.3** |
| 1200 | 2400 | 565 | 526.1 | 216.1 | 152.5 | **50.2** |

`portfolio` — factor covariance, `sum(x)==1` plus caps at `2/n`:

| n | m | nact | qtqp | qtqp_dense | gi | gi_fast |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 201 | 82 | 8.1 | 5.3 | 2.2 | **0.7** |
| 400 | 801 | 380 | 48.3 | 23.8 | 29.3 | **15.8** |
| 1200 | 2401 | 1171 | 518.1 | **205.6** | 465.9 | 707.3 |

`random_ineq` — dense `P`, dense general inequality rows, `m = 2n`:

| n | m | nact | qtqp | qtqp_dense | gi | gi_fast |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 200 | 95 | 14.7 | 6.4 | **6.2** | 6.6 |
| 400 | 800 | 399 | 483.3 | **51.5** | 230.5 | 226.4 |
| 1200 | 2400 | 1190 | (retired) | **479.5** | 3099.4 | 3195.0 |

`band_sparse` — tridiagonal `P` (`nnz = 3n-2`), box constraints:

| n | m | nact | qtqp | gi | gi_fast |
|---:|---:|---:|---:|---:|---:|
| 200 | 400 | 159 | 5.9 | 4.4 | **0.6** |
| 1200 | 2400 | 910 | **18.2** | 208.6 | 40.5 |
| 2000 | 4000 | 1523 | **29.8** | 908.8 | 95.6 |
| 5000 | 10000 | 3751 | **68.2** | 11546.9 | 2203.7 |

### What the numbers say

**Small and dense belongs to the active-set method.** Below roughly `n = 200`,
`gi_fast` leads by 2-7x on the three families with structured constraint rows.
QTQP pays a fixed setup cost — equilibration, symbolic factorisation, the first
numeric factorisation — that a few hundred microseconds of dual walking does not
justify. `random_ineq` is the exception even at this size: with dense rows it is
already a tie at `n = 100` (`gi` 6.2ms against `qtqp_dense` 6.4ms).

**Sparse belongs to QTQP, by margins nothing else here reaches.** On
`band_sparse` at `n = 5000`, QTQP is 169x faster than the exact walk and 32x
faster than its fast path. QTQP's own time grows about linearly across that
sweep (18ms → 68ms for a 4x size increase) because it never leaves the sparse
factorisation; cvx-quadprog has to materialise a dense `n x n` `G` before it
starts, and that alone is 200MB at `n = 5000`.

**Density of the constraint rows decides the backend, and the default gets it
wrong on dense data.** On `random_ineq` the `AUTO` backend is 5-17x slower than
`SCIPY_DENSE` over `n = 200..800` (3222ms against 187ms at `n = 800`) because
it runs a sparse KKT factorisation over a matrix with no zeros in it. Anyone
solving dense-`A`
problems should pass `linear_solver=qtqp.LinearSolver.SCIPY_DENSE` explicitly.
This is the largest single effect measured here, and it is a QTQP configuration
question rather than a comparison between the two libraries.

**The crossover is set by the active set, not by `n`.** `portfolio` and
`box_dense` are the same size and both dense, but the portfolio caps put
`nact ≈ n` where the box puts `nact ≈ n/2`, and only the portfolio crosses over
by `n = 1200`. The active-set method's iteration count grows with the active
set; QTQP's does not.

**`fast=True` is not uniformly faster**, exactly as its documentation says. It
wins by up to 5x on most of the sweep, and loses on `portfolio` at `n = 1200`
(707ms against the exact walk's 466ms), where the active set is nearly the whole
problem.

### Caveats

- The two stop on different criteria: QTQP at `tol_feas = 1e-8`,
  Goldfarb/Idnani exactly. QTQP is being asked for less accuracy than it can
  deliver, which flatters it slightly. The `rel dobj` column reports objective
  disagreement so this stays visible; every measurement above is at 1e-9 or
  tighter.
- Only strictly convex, feasible, bounded problems are timed, because that is
  cvx-quadprog's problem class. Singular `P`, LPs, and infeasible problems are
  QTQP-only and are recorded in `tests/test_cvx_quadprog.py` instead.
- Both call into BLAS, so results move with the BLAS build and thread count.
- `--budget` drops a solver from larger sizes of a family once it exceeds the
  limit, which is why `qtqp` shows `(retired)` on `random_ineq` at `n = 1200`.

## `benchmark_linear_term_sweep.py`

Solves 100 QPs that differ only in the linear term, with `p`, `a`, `b` and `z`
fixed — a frontier sweep, a rolling rebalance, a scenario grid. Both libraries
have a mechanism for this and they are not the same mechanism, which is the
point of the comparison:

- **QTQP** takes `warm_start=(x, y, s)`, screens it, and saves *interior point
  iterations*. Every solve still equilibrates and factorises.
- **cvx-quadprog's `Sweep`** caches the *factorisation*. `J` depends only on
  `G` and `R` only on `G` and the active set, so both survive a change of
  linear term; a still-valid active set is recovered in `O(n^2)` with no
  iteration at all, and a stale one is repaired rather than abandoned.

```bash
python experiments/benchmark_linear_term_sweep.py
python experiments/benchmark_linear_term_sweep.py --n 400 --steps 100
python experiments/benchmark_linear_term_sweep.py --family box --csv out.csv
```

Three regimes set how near consecutive problems are: `path` walks the term
along a ray, `jitter` applies an independent 1% relative perturbation, and
`independent` redraws it from scratch. The last is the control — nothing is
being reused there, so a mechanism that still claims a speed-up is measuring
something else.

### Results

`portfolio`, `n = 200`, 100 solves per regime, same machine as above. Milliseconds
per solve, with the speed-up over the *same library's* cold run:

| regime | qtqp_cold | qtqp_warm | gi_cold | gi_sweep |
|---|---:|---:|---:|---:|
| `path` | 15.02 | 7.82 (**1.9x**) | 6.48 | 0.29 (**22.0x**) |
| `jitter` | 16.29 | 12.01 (**1.4x**) | 7.30 | 0.40 (**18.1x**) |
| `independent` | 15.29 | 14.49 (1.1x) | 6.62 | 52.06 (**0.13x**) |

Mean iterations per solve — QTQP's IPM steps, cvx-quadprog's active-set adds
plus removes — which is where the wall-clock comes from:

| regime | qtqp_cold | qtqp_warm | gi_cold | gi_sweep |
|---|---:|---:|---:|---:|
| `path` | 8.2 | 3.2 | 192.1 | 2.6 |
| `jitter` | 9.0 | 6.2 | 198.8 | 4.7 |
| `independent` | 8.4 | 8.4 | 191.4 | 131.4 |

Reuse hit rate (`warm_accepted`; for `Sweep`, an `iterations` of `(0,0)`):

| regime | qtqp_warm | gi_sweep |
|---|---:|---:|
| `path` | 99% | 58% |
| `jitter` | 99% | 0% |
| `independent` | 99% | 0% |

### What the numbers say

**`Sweep` is the bigger win by an order of magnitude, when the problems really
are near.** 22x on `path` at `n = 200`, and 36x at `n = 400`. QTQP's warm start
tops out near 2x. That gap is structural rather than incidental: `Sweep` is
amortising the factorisation, which is most of a solve, while `warm_start`
amortises only the iterations and leaves equilibration and factorisation to be
paid 100 times. QTQP's iterations drop 2.6x on `path` (8.2 → 3.2) but its wall
clock only 1.9x, and the difference is exactly that fixed per-solve cost.

**Almost none of `Sweep`'s win comes from cache hits.** On `jitter` the outright
hit rate is **0%** and it is still 18x faster, because a stale active set is
repaired from the cached factors instead of walked from cold: 199 active-set
operations become 4.7. The hit-rate column badly understates the mechanism, and
reading it alone would give the wrong picture.

**The control earns its place: `Sweep` on unrelated problems is 8x *slower*
than solving cold** (52.06ms against 6.62ms), and 7x slower at `n = 400`. It
still cuts iterations (191 → 131), so it is not that repair fails — it is that
each repaired iteration carries an `O(n^2)` recovery and a full KKT
verification, and against genuinely unrelated data that overhead is paid for
nothing. `Sweep` is the right tool only when consecutive problems are actually
adjacent, and the wrong one otherwise by a wide margin.

**`warm_accepted` is not a useful signal at the default threshold.** It reads
99% in *every* regime, including the control where the warm start buys exactly
zero iterations (8.4 cold, 8.4 warm). The underlying `warm_lambda` score does
discriminate — median 1.7 on `path` against 3.04 on `independent` — but the
default `warm_start_threshold` of 100 sits far above both, so everything is
accepted. This matches what QTQP's own README says about the screen ("acceptance
does not certify distance to the path or guarantee fewer iterations"); worth
knowing that in practice it accepts essentially always. Acceptance is not
*harmful* when it does not help (8.32 against 8.43 iterations when rejected),
and tightening the threshold to 1.0 is actively worse: it rejects the `path`
warm starts too, doubling iterations from 4.05 to 8.18.

**Geometry moves the constants, not the conclusions.** On `box` at `n = 200`
the same pattern holds with `gi_sweep` at 20.8x on `path`, 15.3x on `jitter`,
and 0.25x on the control.

### Caveats

- The dense conversion for cvx-quadprog is *outside* the timed region here,
  unlike in `benchmark_cvx_quadprog.py`: it depends only on `p` and `a`, which
  do not change across the sequence, so a real caller converts once. Charging
  it 100 times would measure a mistake rather than the mechanism.
- QTQP is constructed fresh per problem because `c` is a constructor argument.
  That is validation and presolve, not a factorisation; it is timed separately
  and recorded in the CSV as `construct_ms`.
- The warm start is chained — each solve starts from the previous solution —
  which is what a sweep would do, and which makes the `independent` regime a
  genuine control rather than a repeat of the same start.
- All four configurations are compared on every one of the 100 solves. The
  largest relative objective disagreement was 7.9e-09 on `portfolio` at
  `n = 200`, 6.5e-09 at `n = 400`, and 2.3e-08 on `box` — all consistent with
  QTQP's `tol_feas = 1e-8`, so no configuration is fast by being wrong.
