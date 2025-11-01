# QTQP

[![Build Status](https://github.com/google-deepmind/qtqp/actions/workflows/ci.yml/badge.svg)](https://github.com/google-deepmind/qtqp/actions/workflows/ci.yml)

The cutie QP solver is a primal-dual interior point method for solving
convex quadratic programs (QPs), implemented in pure python. It solves
primal QP problem:

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

With data `a, b, c, p, z` and variables `x, y, s`. It will return a primal-dual
solution should one exist, or a certificate of primal or dual infeasibility
otherwise.

## Installation

QTQP is available via pip:

```bash
python -m pip install qtqp
```

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

Note tests will fail for linear solvers that are not installed on your system.

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

You should see something like

```
| QTQP v0.0.3: m=3, n=2, z=1, nnz(A)=4, nnz(P)=4, linear_solver=SCIPY
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
-   `b`: (m) RHS vector.
-   `c`: (n) Cost vector.
-   `z`: Number of equality constraints (size of the zero cone). Must satisfy
    `0 ≤ z < m`.
-   `p`: (n×n) QP matrix. If None, treated as the zero matrix (i.e., LP).

This class has a single API method `solve`:

```python
solve(
    *,
    atol: float = 1e-7,
    rtol: float = 1e-8,
    atol_infeas: float = 1e-8,
    rtol_infeas: float = 1e-9,
    max_iter: int = 100,
    step_size_scale: float = 0.99,
    min_static_regularization: float = 1e-7,
    max_iterative_refinement_steps: int = 50,
    linear_solver_atol: float = 1e-12,
    linear_solver_rtol: float = 1e-12,
    linear_solver: qtqp.LinearSolver = qtqp.LinearSolver.SCIPY,
    verbose: bool = True,
    equilibrate: bool = True,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    s: np.ndarray | None = None,
) -> qtqp.Solution
```

Key parameters:

-   `atol`, `rtol`: Absolute/relative stopping tolerances for optimality.
-   `atol_infeas`, `rtol_infeas`: Thresholds for (primal/dual) infeasibility
    detection.
-   `max_iter`: Iteration cap.
-   `step_size_scale` (0,1): Scale for line search step size to stay strictly
    interior.
-   `min_static_regularization`: Diagonal regularization on KKT for robustness.
-   `max_iterative_refinement_steps`, `linear_solver_atol/rtol`: Control
    iterative refinement of the linear solve.
-   `linear_solver`: (`qtqp.LinearSolver`) Choose the KKT solver backend (see
    below).
-   `verbose`: Print per-iteration table with key metrics.
-   `equilibrate`: Scale/equilibrate data for numerical stability.
-   `x`, `y`, `s`: Optional warm-starts (must satisfy conic interiority).

This method will return a `qtqp.Solution` object, with fields:

-   `x`: (n) Primal variable or certificate of unboundedness.
-   `y`: (m) Dual variable or certificate of infeasibility.
-   `s`: (m) Slack variable or certificate of unboundedness.
-   `status`: (`qtqp.SolutionStatus`) One of `SOLVED`, `INFEASIBLE`,
    `UNBOUNDED`, `FAILED`.
-   `stats`: (list of dicts) Includes primal/dual objective, residuals, gap, mu,
    elapsed time, etc, for each iteration.

## Linear solvers

The backend linear system solver can be changed by passing a `qtqp.LinearSolver`
to the `solve` method via the `linear_solver` argument. By default
`linear_solver=qtqp.LinearSolver.SCIPY` which uses `scipy.linalg.factorized`.
QTQP supports several other linear solvers that may be faster or more reliable
for your problem. The enum `qtqp.LinearSolver` contains values corresponding
to the following backend solvers:

#### MKL Pardiso: `qtqp.LinearSolver.PARDISO`

Pardiso is available via the pydiso package (only available for Intel CPUs). To
install

```bash
conda install pydiso -c conda-forge
```

#### Eigen: `qtqp.LinearSolver.EIGEN`

Available via nanoeigenpy. To install

```bash
conda install nanoeigenpy -c conda-forge
```

#### MUMPS: `qtqp.LinearSolver.MUMPS`

Available via petsc4py. To install

```bash
conda install petsc4py -c conda-forge
```

#### QDLDL: `qtqp.LinearSolver.QDLDL`

To install

```bash
python -m pip install qdldl
```

#### CHOLMOD: `qtqp.LinearSolver.CHOLMOD`

Cholmod is available in the scikit sparse package. To install

```bash
conda install scikit-sparse -c conda-forge
```

#### Nvidia cuDSS: `qtqp.LinearSolver.CUDSS`

cuDSS uses a GPU accelerated direct solver (requires a GPU). To install

```bash
python -m pip install nvidia-cudss-cu12
python -m pip install nvmath-python[cu12]
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

