# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QTQP is a pure Python primal-dual interior point solver for convex quadratic programs (QPs). It solves:

```
min (1/2) x^T @ P @ x + c^T @ x
s.t. A @ x + s = b
     s[:z] == 0  (equality constraints)
     s[z:] >= 0  (inequality constraints)
```

Developed by Google DeepMind. Can detect infeasibility and unboundedness via homogeneous embedding.

## Commands

```bash
# Install in dev mode
python -m pip install -e .

# Run all tests
python -m pytest .

# Run a single test
python -m pytest tests/test_qtqp.py::test_name -v

# Build distribution
python -m build
```

## Architecture

The solver lives in two files under `src/qtqp/`:

**`__init__.py`** — The `QTQP` class and solve loop. Key phases per iteration:
1. Normalization (keeps iterates on central path)
2. KKT system solve via `DirectKktSolver`
3. Affine (predictor) step with `mu_target=0`
4. Sigma computation via Mehrotra's heuristic
5. Corrector step with `mu_target=sigma*mu`
6. Step size computation (maintains `y, s > 0`)
7. Termination check (optimality, infeasibility, unboundedness)

Preprocessing: Ruiz equilibration (`_equilibrate`) scales A and P rows/columns for numerical stability. The `tau` variable enables homogeneous embedding for infeasible/unbounded detection.

**`direct.py`** — `DirectKktSolver` wraps multiple backend linear solvers via a Protocol interface. Default is scipy; optional backends require extra packages: `qdldl`, `scikit-sparse` (CHOLMOD), `nanoeigenpy` (Eigen), `pydiso` (MKL Pardiso), `petsc4py` (MUMPS), `nvidia-cudss` + `cupy` (CUDSS GPU sparse), `cupy` (CUPY_DENSE GPU dense). The solver builds the augmented KKT matrix, factorizes it, and performs iterative refinement each call to `.solve()`.

**`tests/test_qtqp.py`** — Integration tests that call the full solver. `conftest.py` provides a fixture that collects per-iteration stats. Tests check solution accuracy with `np.testing.assert_allclose`.

## Key Constraints

- Input matrices must be in CSC (compressed sparse column) format
- Python 3.9+ required; core deps are numpy ≥1.23 and scipy ≥1.9
- CI runs on Ubuntu/macOS/Windows with Python 3.10–3.13
