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
| QTQP v0.0.5: m=3, n=2, z=1, nnz(A)=4, nnz(P)=4, linear_solver=SCIPY
|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
| iter |      pcost |      dcost |     pres |     dres |      gap |   infeas |       mu |  q, p, c |     time |
|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
|    0 |  1.205e+00 |  1.298e+00 | 2.18e-01 | 6.17e-01 | 9.36e-02 | 1.67e+00 | 1.09e+00 |  1, 1, 1 | 1.61e-02 |
|    1 |  1.161e+00 |  1.211e+00 | 3.16e-02 | 5.23e-02 | 5.01e-02 | 1.35e+00 | 1.04e-01 |  1, 1, 1 | 1.66e-02 |
|    2 |  1.234e+00 |  1.235e+00 | 3.77e-04 | 8.61e-04 | 6.64e-04 | 1.30e+00 | 7.67e-03 |  1, 1, 1 | 1.70e-02 |
|    3 |  1.235e+00 |  1.235e+00 | 3.78e-06 | 8.62e-06 | 6.65e-06 | 1.30e+00 | 1.25e-04 |  1, 1, 1 | 1.74e-02 |
|    4 |  1.235e+00 |  1.235e+00 | 3.78e-08 | 8.62e-08 | 6.65e-08 | 1.30e+00 | 1.25e-06 |  1, 1, 1 | 1.78e-02 |
|------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
| Solved
sol.x=array([ 0.29999999, -0.69999997])
sol.y=array([2.69999964e+00, 2.09999968e+00, 3.86572055e-07])
sol.s=array([0.00000000e+00, 7.13141634e-09, 1.99999944e-01])
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
    `0 ≤ z ≤ m`; `z == m` (all-equality) is solved by a single direct KKT
    solve, where `SOLVED` certifies the primal and dual residuals (the gap
    is reported in stats but not tested) and a singular system is reported
    as `FAILED`. Problems with no variables, or with no constraints left
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
    certificate_ktratio: float = 1.0,
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
    certificates are considered (Clarabel uses `1e9`).
-   `max_iter`: Iteration cap.
-   `step_size_scale` (0,1): Scale for line search step size to stay strictly
    interior.
-   `min_static_regularization`: Diagonal regularization on KKT for robustness.
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
    Every iteration also logs `delta_path`, a rigorous a posteriori upper
    bound on the distance from the iterate to the exact central-path point
    at the current `mu` (from the strong monotonicity of the regularized path
    map); it is informative when small, conservative for aggressively
    centered iterates, and saturates at the floating-point floor in the
    final iterations. Its local-norm companion `delta_path_local` measures
    the same residual in the barrier metric `H = mu*I + mu*hess(Phi)`
    (a Newton-decrement analogue): it weights each component by the
    curvature resisting it and remains informative for aggressive
    iterates. `lambda_init` (also an attribute on the solver) is the same
    measure at the deterministic initial point, before the first step:
    healthy problems measure small, while pathologically scaled data
    announces itself by tens of orders of magnitude — this diagnostic is
    how corrupted infinity-sentinel bounds were found in the
    Maros-Meszaros benchmark files.
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
    operating scale, embedded interior at a few centering shifts, and the
    best embedding is accepted only when the distance-to-path certificate
    measures `lambda <= warm_start_threshold`; a vetoed point falls back to
    the standard initialization, so warm starting is never worse than a
    cold solve by more than the certificate evaluation (three matvecs per
    shift). After `solve`, the measured `warm_lambda` and the `warm_accepted`
    decision are attributes on the solver. The `z == m` direct solve
    ignores a warm start.
-   `warm_start_threshold`: Acceptance threshold for the certified warm
    start (default `100.0`). Poisoned or mis-scaled points measure orders of
    magnitude above it; same-problem re-entries orders of magnitude below.

#### Equilibration strategies

Choose one with the `equilibration_strategy` argument:

-   `qtqp.EquilibrationStrategy.RUIZ`: Default. Ruiz equilibration on `A` and
    `P`; `b` and `c` are scaled passively by the accumulated row/column
    scalings.
-   `qtqp.EquilibrationStrategy.AUGMENTED`: Ruiz equilibration on the symmetric
    augmented matrix containing `P`, `A`, `b`, and `c`. This lets `b` and `c`
    participate directly in the scaling and can improve reliability on
    ill-scaled instances.
-   `qtqp.EquilibrationStrategy.NONE`: Disable equilibration.

#### Initialization

The initial iterate is a least-squares initialization in the CVXOPT /
Clarabel style: one saddle-point solve for QPs (two, with a shared
factorization, for LPs - primal from feasibility, dual from optimality),
run through the same linear-solver backend, ordering, static
regularization, and iterative refinement as the main loop, then shifted
into the strict interior. If the initialization solve produces
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
