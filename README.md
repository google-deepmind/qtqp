# QTQP

[![Build Status](https://github.com/google-deepmind/qtqp/actions/workflows/ci.yml/badge.svg)](https://github.com/google-deepmind/qtqp/actions/workflows/ci.yml)

The cutie QP solver is a primal-dual interior point method for solving
convex quadratic programs (QPs), implemented in pure python. It solves the
primal QP:

```
    min. (1/2) x.T @ p @ x + c.T @ x
    s.t. a @ x + s = b
         s[:z] == 0
         s[z:] >= 0
```

With dual:

```
    max. -(1/2) x.T @ p @ x - b.T @ y
    s.t. p @ x + a.T @ y = -c
         y[z:] >= 0
```

with data `a, b, c, p, z` and variables `x, y, s`. It returns a primal-dual
solution when one exists, or a certificate of primal or dual infeasibility
otherwise.

## Installation

QTQP is available via pip:

```bash
python -m pip install qtqp
```

On supported platforms this also installs the recommended sparse CPU backend
automatically:

- Linux / Windows `x86_64`: `py-mkl-pardiso`
- macOS `arm64`: `macldlt`

To install from source, first clone the repository:

```bash
git clone https://github.com/google-deepmind/qtqp.git
cd qtqp
```

Then, assuming conda is installed, create a new conda environment:

```bash
conda create -n tmp python=3.12
conda activate tmp
```

Finally, install the package:

```bash
python -m pip install .
```

To run the tests, inside the qtqp directory:

```bash
python -m pytest .
```

Tests for optional linear solvers are skipped when the corresponding
dependencies are not installed.

## Quick start

Here is an example usage (taken from
[here](https://www.cvxgrp.org/scs/examples/python/basic_qp.html#py-basic-qp)):

```python
import qtqp
import scipy
import numpy as np

# Set up the problem data
p = scipy.sparse.csc_matrix([[3.0, -1.0], [-1.0, 2.0]])
a = scipy.sparse.csc_matrix([[-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([-1, 0.3, -0.5])
c = np.array([-1.0, -1.0])

# Initialize solver
solver = qtqp.QTQP(p=p, a=a, b=b, c=c, z=1)
# Solve!
sol = solver.solve()
print(f'{sol.x=}')
print(f'{sol.y=}')
print(f'{sol.s=}')
```

You should see output similar to

```
| QTQP v0.0.6: m=3, n=2, z=1, nnz(A)=4, nnz(P)=4, linear_solver=ACCELERATE, equilibration=RUIZ
|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
| iter |      pcost |      dcost |     pres |     dres |      gap |   infeas |       mu |  q, p, c |     time |
|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
|    0 |  1.340e+00 |  1.131e+00 | 1.28e-01 | 1.67e-01 | 2.09e-01 | 1.05e+00 | 1.78e-01 |  1, 1, 1 | 1.76e-01 |
|    1 |  1.221e+00 |  1.226e+00 | 1.60e-02 | 7.08e-03 | 4.23e-03 | 8.21e-01 | 3.93e-03 |  1, 1, 1 | 1.76e-01 |
|    2 |  1.235e+00 |  1.235e+00 | 1.65e-04 | 7.28e-05 | 3.80e-05 | 8.09e-01 | 4.73e-05 |  1, 1, 1 | 1.77e-01 |
|    3 |  1.235e+00 |  1.235e+00 | 4.14e-08 | 1.83e-08 | 9.53e-09 | 8.09e-01 | 1.26e-08 |  1, 1, 1 | 1.77e-01 |
|    4 |  1.235e+00 |  1.235e+00 | 5.08e-12 | 2.38e-12 | 8.62e-13 | 8.09e-01 | 1.26e-12 |  1, 2, 2 | 1.77e-01 |
|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
| Solved
sol.x=array([ 0.3, -0.7])
sol.y=array([2.70000000e+00, 2.10000000e+00, 1.47264885e-11])
sol.s=array([0.00000000e+00, 2.72772187e-13, 2.00000000e-01])
```

## API reference

Once installed QTQP is imported using

```python
import qtqp
```

This exposes the main solver class `qtqp.QTQP` with constructor:

```python
QTQP(
    *,
    a: scipy.sparse.csc_matrix,
    b: np.ndarray,
    c: np.ndarray,
    z: int,
    p: scipy.sparse.csc_matrix | None = None,
)
```

Arguments:

-   `a`: (m×n) Constraint matrix.
-   `b`: (m) RHS vector. For inequality rows, values at or above
    `1e20 * (1 - 1e-9)` are reserved: they are treated as `+inf` (the row is
    unbounded and removed by presolve, with its dual fixed to 0). Finite data
    in that range is therefore discarded by design — rescale any genuine
    constraint whose RHS approaches `1e20` before calling the solver.
-   `c`: (n) Cost vector.
-   `z`: Number of equality constraints (size of the zero cone). Must satisfy
    `0 ≤ z ≤ m`; an all-equality problem (`z == m`) is solved by the
    initialization itself, which solves that KKT system exactly, and is
    graded by the standard criteria at iteration 0 (a singular system is
    reported as `FAILED`, or `ALMOST_SOLVED` if the reduced tolerances
    hold). Problems with no variables, or with no constraints left
    after presolve, are rejected.
-   `p`: (n×n) QP matrix. If None, treated as the zero matrix (i.e., LP).

This class has a single API method `solve`:

```python
solve(
    *,
    tol_feas: float = 1e-8,
    tol_gap_abs: float = 1e-8,
    tol_gap_rel: float = 1e-8,
    tol_infeas_abs: float = 1e-8,
    tol_infeas_rel: float = 1e-8,
    certificate_ktratio: float = 1e9,
    max_iter: int = 100,
    step_size_scale: float = 0.99,
    min_static_regularization: float = 1e-8,
    max_iterative_refinement_steps: int = 20,
    linear_solver_atol: float = 1e-12,
    linear_solver_rtol: float = 1e-12,
    linear_solver: qtqp.LinearSolver = qtqp.LinearSolver.AUTO,
    verbose: bool = True,
    equilibration_strategy: qtqp.EquilibrationStrategy = (
        qtqp.EquilibrationStrategy.RUIZ
    ),
    collect_stats: bool = False,
    refinement_strategy: qtqp.RefinementStrategy = (
        qtqp.RefinementStrategy.GMRES
    ),
    gmres_restart: int = 20,
    max_centrality_correctors: int = 1,
    adaptive_step_size: bool = True,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    warm_start_threshold: float = 100.0,
) -> qtqp.Solution
```

Key parameters:

-   `tol_feas`, `tol_gap_abs`, `tol_gap_rel`: Stopping tolerances for
    optimality (Clarabel's definitions; see below).
-   `tol_infeas_abs`, `tol_infeas_rel`: Thresholds for (primal/dual)
    infeasibility detection.
-   `certificate_ktratio`: Embedding ratio `kappa / tau` above which
    certificates are considered (default `1e9`, as in Clarabel).
-   `max_iter`: Iteration cap.
-   `step_size_scale` (0,1): Scale for line search step size to stay strictly
    interior.
-   `min_static_regularization`: Diagonal regularization on KKT for robustness.
    Dense Gram backends (`SCIPY_DENSE`, `CUPY_DENSE`) require a positive value
    when initializing a problem with equality rows (`z > 0`); zero raises
    `ValueError` because Gram elimination cannot invert a zero equality diagonal.
-   `max_iterative_refinement_steps`, `linear_solver_atol/rtol`: Control
    iterative refinement of the linear solve. The default is 20 refinement
    steps, counting the initial solve.
-   `linear_solver`: (`qtqp.LinearSolver`) Choose the KKT solver backend (see
    below).
-   `verbose`: Print per-iteration table with key metrics.
-   `equilibration_strategy`: Choose how problem data is scaled before the IPM
    iterations. Defaults to `qtqp.EquilibrationStrategy.RUIZ`.
-   `collect_stats`: If True, populate `Solution.stats` with per-iteration
    diagnostics (sy, s/y statistics, complementarity, etc.). Defaults to False
    for faster throughput.
    `Solution.iterations` always reports completed IPM steps, excluding
    initialization and failed step attempts, even with `collect_stats=False`.
    A solution found at initialization reports zero. In collected rows, `iter`
    remains the zero-based label and `iterations` is the completed-step count
    at that row; the final log footer reports the total completed count.
    Collected rows also include `delta_path`, an estimate of
    `||T_mu(u)|| / mu`. In exact arithmetic with positive `mu`, this bounds
    the distance to the central-path point by strong monotonicity. The
    recorded value uses floating-point denominator guards, so it is a
    diagnostic rather than a certified numerical bound. Its local-norm
    companion `delta_path_local` approximates the residual norm
    `lambda = ||T_mu(u)||_(H^-1)`, where
    `H = mu*I + mu*hess(Phi)`. This score also uses denominator guards and
    is not itself a distance bound. `lambda_init` (also an attribute on
    the solver) is the same guarded local score at the chosen initial
    point, warm or cold, before the first step. For an accepted warm
    start, it reuses the screening score.
-   `refinement_strategy`: Choose the iterative-refinement method used for KKT
    solves. Defaults to `qtqp.RefinementStrategy.GMRES`.
-   `gmres_restart`: Restart length for `qtqp.RefinementStrategy.GMRES`.
    Defaults to `20`: one uninterrupted Krylov cycle spanning the full
    refinement budget, avoiding restart stagnation. Ignored by Richardson
    refinement.
-   `max_centrality_correctors`: Maximum Gondzio-style centrality correctors
    per iteration, each one extra back-solve on the existing factorization,
    recentering the aspirational trial point's outlier complementarity
    products and accepted only when the step size improves. Default `1`
    (validated on Maros-Meszaros + NETLIB + MIPLIB: 14-16% fewer median
    iterations on every dataset with unchanged robustness); `0` disables.
-   `adaptive_step_size`: If True (the default), once `mu < 1e-3` the
    fraction-to-boundary scale follows `min(0.9999, max(step_size_scale,
    1 - 10*mu))`: the margin to the cone boundary shrinks proportionally
    to `mu`, unlocking the superlinear endgame that a constant haircut
    caps at a linear rate. Set False for the constant legacy schedule.
-   `warm_start`: Optional `(x, y, s)` from a nearby problem (original scale,
    e.g. a previous solution's arrays). The point is equilibrated into the
    operating scale, its inequality duals and slacks are floored at each
    of `1e-6`, `1e-4`, `1e-2`, and `1.0`, and each candidate is normalized.
    The candidate with the smallest guarded local score is accepted when
    `lambda <= warm_start_threshold`; otherwise the solver uses the
    standard initialization. This is empirical screening: acceptance does
    not certify distance to the path or guarantee fewer iterations or less
    time than a cold solve. Screening uses no factorization and six
    matrix-vector products across the four candidates: shared `A @ x` and
    `P @ x`, plus one `A.T @ y` per candidate. An accepted start skips the
    cold initialization factorization and reuses its score as `lambda_init`;
    a rejected start adds screening overhead before the cold solve. After
    `solve`, the measured `warm_lambda` and the `warm_accepted` decision are
    attributes on the solver. A warm start is ignored on an all-equality
    problem, which the initialization solves outright.
-   `warm_start_threshold`: Empirical screening threshold (default `100.0`),
    not a certified distance tolerance. For the exact local score, the
    theoretical distance bound instead requires
    `eta = lambda / sqrt(mu) < 1` and is `eta / (1 - eta)`. The solver does
    not use this test for acceptance, and its guarded score must not be
    substituted into that bound as a numerical certificate.

#### Equilibration strategies

Choose one with the `equilibration_strategy` argument:

-   `qtqp.EquilibrationStrategy.RUIZ`: Default. Ruiz equilibration on the KKT
    block `[P, A'; A, 0]`, plus the two scalars a QP admits freely: a joint
    scale on `b` and `c` (a rescale of the solution) taking `||b||_inf` to 1,
    and a scale on the objective `P`, `c` (a rescale of the duals) taking
    `max(||c||_inf, max |P_ij|)` to 1, both kept within `[1e-4, 1e4]`. `b`
    and `c` never enter the row/column scalings, so the factorized block
    keeps unit rows and columns.
-   `qtqp.EquilibrationStrategy.AUGMENTED`: Ruiz equilibration on the symmetric
    augmented matrix containing `P`, `A`, `b`, and `c`, so that `b` and `c`
    also inform the row and column scalings. Kept for experimentation: it
    splits the magnitude of large entries of `b` and `c` into the factorized
    block, and RUIZ solved more problems on every benchmark collection.
-   `qtqp.EquilibrationStrategy.NONE`: Disable equilibration.

#### Initialization

The initial iterate is Clarabel's initialization: one solve of
`[P, A'; A, -H][x; y] = [-c; b]` for QPs (two, with a shared
factorization, for LPs - primal from feasibility, dual from optimality),
with `H` the identity on inequality rows and zero on equality rows, so
equality rows are satisfied exactly by the initial point. It runs through
the same linear-solver backend, ordering, static regularization, and
iterative refinement as the main loop, then the inequality components are
shifted into the strict interior. The initial point is graded before the
first step, so an exact initialization or an accepted warm start can
terminate at iteration 0. If the initialization solve produces
non-finite values, the solver falls back to a trivial unit start.

#### Refinement strategies

Choose one with the `refinement_strategy` argument:

-   `qtqp.RefinementStrategy.GMRES`: Default. Restarted right-preconditioned
    GMRES on the true KKT system. Each Arnoldi step consumes one factor-solve,
    and `gmres_restart` controls the restart length.
-   `qtqp.RefinementStrategy.RICHARDSON`: Classical iterative refinement using
    the factorized regularized KKT matrix as a preconditioner (a smoother; its
    best-effort iterates behave differently from GMRES's residual-optimal ones
    in the deep endgame).

#### Advanced numerical options

-   Termination criteria are Clarabel's, evaluated on the returned point,
    so results are directly comparable: `SOLVED` requires
    `||Ax + s - b|| / max(1, ||b||_inf + ||x|| + ||s||) < tol_feas`,
    `||Px + A'y + c|| / max(1, ||c||_inf + ||x|| + ||y||) < tol_feas`
    (2-norms), and a duality gap `|pcost - dcost|` below `tol_gap_abs` or
    below `tol_gap_rel * max(1, min(|pcost|, |dcost|))`, with the iterate on
    the solution side of the embedding (`kappa / tau < 1`). A certificate
    requires `kappa / tau > certificate_ktratio`, an objective slope
    (`b'y` or `c'x`) below `-tol_infeas_abs`, and violations relative to
    `max(1, ||ray||)` below `tol_infeas_rel * |slope|`.
-   `x`: (n) Primal variable or certificate of unboundedness.
-   `y`: (m) Dual variable or certificate of infeasibility.
-   `s`: (m) Slack variable or certificate of unboundedness.
-   `status`: (`qtqp.SolutionStatus`) One of `SOLVED`, `INFEASIBLE`,
    `UNBOUNDED`, `ALMOST_SOLVED`, `HIT_MAX_ITER`, `FAILED`.

    Inequality rows removed by presolve (RHS at or above the `1e20`
    infinity sentinel) are restored in `s` as `+inf` with dual `0` on
    solution outputs, and as `NaN` on certificates; verifiers should mask
    them.
    `ALMOST_SOLVED` is returned in place of `HIT_MAX_ITER` (or of a
    numerical breakdown of the linear solver, which never raises) when
    the best iterate over the trajectory (by max normalized residual)
    meets the solved criteria at Clarabel's reduced tolerances (`1e-4`
    feasibility, `5e-5` gap): the returned
    solution is that best iterate, honestly labeled as not meeting the
    full `SOLVED` contract. `SOLVED` semantics are unchanged. A
    breakdown whose best iterate does not qualify returns `FAILED`.
-   `stats`: (list of dicts) Per-iteration diagnostics. Empty unless
    `collect_stats=True`. When enabled, includes primal/dual objective,
    residuals, gap, mu, elapsed time, and complementarity statistics.

## Linear solvers

The backend linear system solver can be changed by passing a `qtqp.LinearSolver`
to the `solve` method via the `linear_solver` argument. By default
`linear_solver=qtqp.LinearSolver.AUTO`. AUTO resolves to
`qtqp.LinearSolver.PARDISO` first on Linux / Windows and to
`qtqp.LinearSolver.ACCELERATE` first on macOS, then falls back through the
other sparse CPU backends before finally using `qtqp.LinearSolver.SCIPY`.
The enum
`qtqp.LinearSolver` contains values corresponding to the following backend
solvers:

Recommended starting points:

| System / problem type | Recommended solver |
| --- | --- |
| Default choice | `qtqp.LinearSolver.AUTO` |
| Linux / Windows | `qtqp.LinearSolver.PARDISO` |
| macOS | `qtqp.LinearSolver.ACCELERATE` |
| NVIDIA GPU available | `qtqp.LinearSolver.CUDSS` |
| Dense data | `qtqp.LinearSolver.SCIPY_DENSE` |
| Tiny problems (`n + m < 50`) | `qtqp.LinearSolver.QDLDL` |

#### Automatic selection: `qtqp.LinearSolver.AUTO`

Runtime selection for sparse CPU backends.

- Linux / Windows preference order starts with `PARDISO`.
- macOS preference order starts with `ACCELERATE`.
- The default install brings in `py-mkl-pardiso` on Linux / Windows `x86_64`
  and `macldlt` on macOS `arm64`.
- If the preferred backend is unavailable, QTQP tries the remaining sparse CPU
  backends and finally falls back to `SCIPY`.

#### scipy SuperLU: `qtqp.LinearSolver.SCIPY`

Baseline sparse CPU backend using `scipy.sparse.linalg.factorized`.
No additional dependencies required.

#### MKL Pardiso: `qtqp.LinearSolver.PARDISO`

Recommended sparse CPU backend on Linux and Windows. Available via the
py-mkl-pardiso package (Linux and Windows, x86_64). To install

```bash
python -m pip install py-mkl-pardiso
```

#### Accelerate: `qtqp.LinearSolver.ACCELERATE`

Apple Accelerate sparse LDL^T factorization via
[macldlt](https://github.com/bodono/macldlt) (macOS only). Recommended sparse
CPU backend on macOS. Published wheels are currently Apple Silicon only. To
install

```bash
python -m pip install macldlt
```

#### Nvidia cuDSS: `qtqp.LinearSolver.CUDSS`

Recommended sparse GPU backend when an NVIDIA GPU is available. To install

```bash
python -m pip install nvidia-cudss-cu12
python -m pip install nvmath-python[cu12]
python -m pip install cupy-cuda12x
```

#### Dense Cholesky: `qtqp.LinearSolver.SCIPY_DENSE`

Recommended backend for dense data. Uses a dense Schur-complement / Cholesky
factorization. No additional dependencies required.

#### QDLDL: `qtqp.LinearSolver.QDLDL`

Sparse LDL^T backend via `qdldl`. To install

```bash
python -m pip install qdldl
```

#### UMFPACK: `qtqp.LinearSolver.UMFPACK`

Sparse LU backend via scikit-umfpack. To install

```bash
conda install scikit-umfpack -c conda-forge
```

#### CHOLMOD: `qtqp.LinearSolver.CHOLMOD`

Sparse Cholesky / LDL^T backend via scikit-sparse. To install

```bash
conda install suitesparse -c conda-forge
python -m pip install 'scikit-sparse>=0.5'
```

#### Eigen: `qtqp.LinearSolver.EIGEN`

Sparse LDL^T backend via nanoeigenpy. To install

```bash
conda install nanoeigenpy -c conda-forge
```

#### MUMPS: `qtqp.LinearSolver.MUMPS`

Sparse direct solver backend via petsc4py / MUMPS. To install

```bash
conda install petsc4py -c conda-forge
```

#### cupy dense GPU: `qtqp.LinearSolver.CUPY_DENSE`

GPU counterpart of `SCIPY_DENSE`: dense Schur-complement / Cholesky on GPU via
cupy/cuSOLVER. To install

```bash
python -m pip install cupy-cuda12x
```

## Citing this work

Coming soon, in the meantime the closest work is:

```
@article{odonoghue:21,
    author       = {Brendan O'Donoghue},
    title        = {Operator Splitting for a Homogeneous Embedding of the Linear Complementarity Problem},
    journal      = {{SIAM} Journal on Optimization},
    month        = {August},
    year         = {2021},
    volume       = {31},
    issue        = {3},
    pages        = {1999-2023},
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
