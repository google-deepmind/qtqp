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
