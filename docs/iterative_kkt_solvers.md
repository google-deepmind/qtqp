# Iterative KKT Solvers for Interior Point Methods: Findings

This document records the experiments and findings from investigating iterative
(matrix-free) alternatives to direct factorization for solving KKT systems in
QTQP's primal-dual interior point method.

## Background

QTQP solves QPs via a primal-dual IPM. Each iteration solves multiple KKT systems
of the form:

```
[P + mu*I      A^T    ] [x]   [r1 ]
[   A     -(D + mu*I) ] [y] = [-r2]
```

where `D = diag(s/y)` for inequality constraints (and `D = 0` for equalities),
`mu` is the barrier parameter that decreases toward zero, and `P` is the QP
cost matrix (PSD).

Direct solvers (LDL^T factorization) work well but have O(nnz^{1.5}) fill-in
cost that dominates for large sparse problems. Iterative solvers need only
matrix-vector products at O(nnz) per iteration, potentially offering large
speedups for n+m > ~1000.

### Key property of QTQP's KKT system

QTQP adds `mu*I` to both diagonal blocks (not just the (1,1) block). This means:
- The (1,1) block `P + mu*I` has eigenvalues >= mu (SPD)
- The (2,2) block `-(D + mu*I)` has eigenvalues <= -mu (negative definite)
- The full KKT is symmetric indefinite with no zero eigenvalues

The condition number of the KKT matrix is O(||A||^2 / mu^2), which grows as
mu -> 0. This is the central challenge for iterative methods.

## Approaches Tested

### 1. CG on Normal Equations (WORKS -- shipped as `LinearSolver.CG`)

**Idea**: Eliminate y from the KKT system by substitution. The KKT system:
```
H x + A^T y = r1       (H = P + mu*I)
A x - D y   = r2       (D = diag(s/y) + mu*I)
```
gives `y = D^{-1}(Ax - r2)`, substituting into the first equation:
```
N x = r1 + A^T D^{-1} r2
```
where `N = H + A^T D^{-1} A` is the normal equations matrix (n x n, SPD).

**Implementation** (`CgNormalEqSolver` in `indirect.py`):
- N is applied matrix-free: `N v = (P + mu*I) v + A^T (D^{-1} (A v))`, costing
  O(nnz(A)) per CG iteration
- Diagonal preconditioner: `M = diag(P) + mu + colsum(A^2 / d_vec)`, which
  approximates `diag(N)` cheaply via precomputed `A^2` (element-wise squared)
- After CG converges on x, y is recovered as `y = D^{-1}(Ax - r2)`
- Tolerance: `atol + rtol * ||rhs||`, converted to scipy CG's relative form

**Result**: All 218 test cases pass (feasible, infeasible, unbounded, LP, all-inequality)
with `atol=rtol=1e-4` for the IPM and `linear_solver_atol=linear_solver_rtol=1e-8`
for the inner CG solve.

**Why it works**: The diagonal preconditioner captures both the `diag(P) + mu` and
the `A^T D^{-1} A` contributions to N's diagonal, making it effective across all
mu values. CG on SPD systems has well-understood convergence, and the preconditioner
keeps iteration counts bounded.

**Known limitation**: y recovery amplifies the CG solve error by O(1/mu) because
`y = D^{-1}(...)` and `D` has entries as small as mu. In practice, the CG tolerance
(1e-8) is tight enough that even after amplification the KKT residual stays small.

### 2. MINRES on Full KKT with Diagonal Scaling (FAILED -- `LinearSolver.MINRES_DIAG`)

**Idea**: Solve the full (n+m) x (n+m) KKT system directly with MINRES, avoiding
the normal equations reduction and its O(1/mu) y-recovery amplification. Use
explicit diagonal scaling `S = diag(1/sqrt(|diag(KKT)|))` to improve conditioning.

**Approach 2a: Passing diagonal preconditioner to scipy.minres**

First attempt: pass `M = diag(1/|diag(KKT)|)` as a preconditioner to
`scipy.sparse.linalg.minres`.

Result: **Failed**. scipy's MINRES checks the **preconditioned** residual norm
`||M r||`, not the true residual `||r||`. The preconditioner makes the
preconditioned residual look small while the true residual grows. Diagnostic
comparison showed:
```
mu=1e-6: preconditioned MINRES residual = 4e-5 (looks converged)
         true residual = 4.7e+07 (catastrophically wrong)
```

**Approach 2b: Unpreconditioned MINRES**

Remove the preconditioner entirely: `scipy.sparse.linalg.minres(KKT, rhs, rtol=...)`.

Result: **Failed**. Without preconditioning, the condition number O(||A||^2/mu^2)
prevents convergence at small mu. At mu=1e-6 with condition number ~1e7, MINRES
cannot reach 1e-8 relative tolerance.

**Approach 2c: Explicit diagonal scaling**

Instead of passing M to scipy, explicitly transform the system:
```
(S @ KKT @ S) z = S @ rhs,   then   sol = S @ z
```
where `S = diag(1/sqrt(|diag(KKT)|))`. This makes the scaled system's diagonal
entries all +/-1, reducing the condition number from O(1/mu^2) to O(1/mu).

Standalone test (known solution, random A, P):
```
mu      | kappa(KKT)  | kappa(S KKT S)  | MINRES rel_err  | MINRES rel_res
--------|-------------|-----------------|-----------------|---------------
1e+00   | 7e+00       | 7e+00           | 1e-09           | 1e-09
1e-02   | 1e+03       | 8e+01           | 2e-09           | 1e-09
1e-04   | 1e+05       | 8e+02           | 6e-06           | 3e-08
1e-06   | 1e+07       | 8e+03           | 2e-04           | 8e-09
1e-08   | 1e+09       | 8e+04           | 8e-02           | 2e-08
```

The scaling reduces kappa dramatically (1e9 -> 8e4 at mu=1e-8) and the **scaled
residual** converges to ~1e-8. But the **solution error** grows as O(kappa_scaled),
reaching 8% at mu=1e-8.

**The unscaling amplification problem**: When MINRES solves the scaled system with
residual e, the true residual after unscaling is `r_true = S^{-1} @ e`. Since
`S^{-1} = sqrt(|diag(KKT)|)` has entries up to O(sqrt(s/y)) ~ O(1/sqrt(mu))
for inactive constraints, the unscaling amplifies the residual by O(1/sqrt(mu)).

**Approach 2d: Explicit scaling + iterative refinement**

Added an outer iterative refinement loop:
1. Compute true residual `r = rhs - KKT @ sol`
2. Scale: solve `(S KKT S) dz = S r` with MINRES
3. Unscale: `sol += S @ dz`
4. Repeat until `||r||_inf < tolerance`

Result: **Failed**. Each refinement step has the same S^{-1} amplification. The
MINRES solution to the correction equation has scaled residual `e`, giving
`r_new = S^{-1} @ e` -- the same amplification factor at every step. Refinement
stalls rather than converging.

IPM diagnostic (150 x 100 problem):
```
MINRES calls  | scaled_res  | true_res    | Notes
1-3           | 1e-06       | 2e-06       | Works fine at large mu
10-12         | 5e-04       | 6e-03       | Degrading at mu ~ 1e-3
16-18         | 3e-02       | 2.6e-01     | True residual >> tolerance
19-21         | 1e+01       | 1.2e+03     | IPM diverging
25+           | 4e+01       | 1.8e+05     | Complete breakdown
```

**Conclusion**: Diagonal scaling on the full KKT is fundamentally limited. It
reduces kappa from O(1/mu^2) to O(1/mu), but the unscaling amplifies residuals
by O(1/sqrt(mu)), and these effects compound rather than cancel. Iterative
refinement cannot fix this because each refinement step suffers the same
amplification.

### 3. MINRES on Full KKT via IndirectKktSolver (FAILED -- `LinearSolver.MINRES`)

**Idea**: Use the `IndirectKktSolver` wrapper with `ScipyMinresSolver` backend.
No preconditioning, just MINRES on the raw KKT matrix with adaptive tolerance.

Result: **Failed**. Same as approach 2b -- without preconditioning, MINRES cannot
converge at small mu. The `IndirectKktSolver` wrapper adds no magic; the
fundamental conditioning problem remains.

### 4. PETSc GMRES + Block Fieldsplit (FAILED -- `LinearSolver.MINRES_PETSC`)

**Idea**: Use PETSc's GMRES with a block fieldsplit preconditioner:
- (1,1) block `P + mu*I`: ICC (incomplete Cholesky, SPD block)
- (2,2) block `-(D + mu*I)`: Jacobi (exact since diagonal)
- Multiplicative fieldsplit composition

Uses separate `amat` (true KKT) and `pmat` (regularized KKT) so GMRES converges
to the true solution while ICC sees a well-conditioned preconditioner matrix.

Convergence checks use unpreconditioned residual norm, and divergence tolerance
is disabled to allow warm-starting with large initial residuals.

**Implementation** (`PetscFieldSplitSolver` in `indirect.py`).

**Result**: Converges well for early IPM iterations (large mu), then breaks down:
```
mu ~ 1.0:   24 iterations, converged (reason=2)
mu ~ 1e-2:  58 iterations, converged
mu ~ 1e-3: 193 iterations, converged (slow)
mu ~ 1e-4: breakdown (reason=-5, KSP_DIVERGED_BREAKDOWN)
```

**Root cause**: Block-diagonal preconditioning ignores the off-diagonal A/A^T
coupling. At small mu, the coupling dominates the system (the off-diagonal
blocks scale as O(||A||) while the diagonals scale as O(mu) for equality
constraints). The preconditioner provides no information about this coupling,
so GMRES stalls or breaks down.

### 5. Systematic PETSc Configuration Sweep (ALL FAILED)

Tested all combinations of KSP type, preconditioner, and fieldsplit variant
on a controlled problem (n=100, m=150, z=10) with IPM-realistic s/y ratios
(half constraints active with s/y ~ mu, half inactive with s/y ~ 1/mu).

#### Block fieldsplit variants (all fail at mu <= 1e-2):

| Configuration                     | mu=1.0 | mu=1e-2 | mu=1e-4 | mu=1e-8 |
|-----------------------------------|--------|---------|---------|---------|
| GMRES + ICC/Jacobi multiplicative | 30 it  | fail    | fail    | fail    |
| GMRES + ILU/Jacobi multiplicative | 30 it  | fail    | fail    | fail    |
| GMRES + ICC/Jacobi additive       | 66 it  | fail    | fail    | fail    |
| FGMRES + ICC/Jacobi multiplicative| 30 it  | fail    | fail    | fail    |
| MINRES + ICC/Jacobi additive      | fail*  | fail    | fail    | fail    |
| MINRES + ILU/Jacobi additive      | fail*  | fail    | fail    | fail    |

*MINRES requires SPD preconditioner; block-diagonal with -(D+mu*I) block is
indefinite, so MINRES rejects it immediately (reason=-8, NaN/Inf detected).

#### Full-matrix preconditioners (all fail at mu <= 1e-2):

| Configuration           | mu=1.0 | mu=1e-2    | mu=1e-4 | mu=1e-8       |
|-------------------------|--------|------------|---------|---------------|
| GMRES + full ILU        | 36 it  | fail (500) | fail    | breakdown     |
| FGMRES + full ILU       | 35 it  | fail (500) | fail    | fail (500)    |
| Direct LU (baseline)    | exact  | exact      | exact   | exact         |

ILU quality degrades because the KKT matrix becomes increasingly ill-conditioned
as mu -> 0, making the incomplete factorization a poor approximation.

#### Schur complement fieldsplit (fails at mu <= 1e-4):

| Configuration                              | mu=1.0 | mu=1e-2     | mu=1e-4    | mu=1e-8 |
|--------------------------------------------|--------|-------------|------------|---------|
| Schur DIAG + LU/(GMRES no-pc)             | 30 it  | 29 it       | breakdown  | fail    |
| Schur LOWER + LU/(GMRES no-pc)            | 2 it   | 9 it        | breakdown  | fail    |
| Schur UPPER + LU/(GMRES no-pc)            | 2 it   | 8 it        | breakdown  | fail    |
| Schur FULL + LU/(GMRES no-pc)             | 2 it   | 6 it        | breakdown  | fail    |
| Schur FULL + LU/(CG + Jacobi on Schur)    | 30 in  | 371 inner   | fail (500) | fail    |

The Schur complement of the (1,1) block is `S = -(D+mu*I) - A(P+mu*I)^{-1}A^T`,
which is the (negative) normal equations matrix. The Jacobi preconditioner on the
Schur complement only uses -(D+mu*I) (the (2,2) block), missing the
`A(P+mu*I)^{-1}A^T` term that dominates at small mu. A proper preconditioner
for the Schur complement would need `diag(A(P+mu*I)^{-1}A^T)` -- which is
exactly what CgNormalEqSolver already computes.

## Why CG on Normal Equations is the Right Approach

The experiments above show a consistent pattern: all iterative methods on the
full (n+m) x (n+m) KKT system fail at small mu because:

1. **Block preconditioners** ignore the A/A^T coupling that dominates at small mu
2. **Full ILU** quality degrades with the condition number
3. **Diagonal scaling** reduces kappa but introduces unscaling amplification
4. **Schur complement** needs a good inner preconditioner, which is the normal
   equations diagonal -- exactly what CG on normal equations already uses

CG on normal equations succeeds because:

1. **SPD structure**: The reduction eliminates the indefinite structure, giving
   an n x n SPD system amenable to CG
2. **Effective preconditioner**: `diag(N) = diag(P) + mu + colsum(A^2/d)` captures
   both the P and A^T D^{-1} A contributions to N's diagonal
3. **Matrix-free**: N is never formed explicitly; each CG iteration costs
   O(nnz(A)) via `N v = (P+mu*I)v + A^T(D^{-1}(Av))`
4. **No fill-in**: Unlike direct methods, there is no sparse factorization

The Schur complement approach via PETSc is mathematically equivalent but with
more complexity and no advantage over the scipy CG implementation.

## Conditioning Analysis

Both the full KKT and normal equations N have condition number O(||A||^2/mu^2):

- **KKT**: eigenvalues in `[-max(d+mu), -mu] U [mu, max(diag(P)+mu) + ||A||^2/mu]`,
  giving `kappa(KKT) = O(||A||^2/mu^2)`
- **Normal equations**: `N = P + mu*I + A^T D^{-1} A` has
  `lambda_min >= mu` and `lambda_max >= ||A_eq||^2/mu`,
  giving `kappa(N) >= ||A_eq||^2/mu^2`

The diagonal preconditioner for N reduces the effective condition number
significantly in practice, making CG iteration counts manageable across all
mu values encountered in the IPM.

An initial claim that QTQP's mu*I regularization gives O(1/mu) conditioning
(vs O(1/mu^2) for standard IPMs) was incorrect. The mu*I terms bound the
smallest eigenvalue at O(mu), but the largest eigenvalue is dominated by the
off-diagonal coupling ||A||^2/mu, giving O(1/mu^2) regardless.

## Code in This Branch

The branch contains the full implementation of all approaches:

- `src/qtqp/indirect.py`:
  - `IterativeSolver` -- base class for iterative backends
  - `ScipyMinresSolver` -- scipy MINRES backend (used by MINRES)
  - `PetscFieldSplitSolver` -- PETSc GMRES+fieldsplit backend (used by MINRES_PETSC)
  - `IndirectKktSolver` -- generic wrapper for full-KKT iterative solvers
  - `CgNormalEqSolver` -- **working** CG on normal equations solver
  - `MinresSolver` -- MINRES with diagonal scaling + iterative refinement (failed)

- `src/qtqp/__init__.py`:
  - `LinearSolver` enum entries: CG, MINRES, MINRES_PETSC, MINRES_DIAG
  - Dispatch logic to construct the right solver type

- `tests/test_qtqp.py`:
  - All solvers added to parametrization
  - Iterative solver tolerance infrastructure (`_ITERATIVE_SOLVERS`, `_ITERATIVE_TOL`, etc.)
  - Direct-solver-specific tests excluded from iterative parametrization

Only `LinearSolver.CG` passes all tests. MINRES, MINRES_PETSC, and MINRES_DIAG
are included as experimental/reference implementations.

## Future Directions

If revisiting iterative solvers on the full KKT:

1. **Constraint preconditioners** (Keller, Gould, Wathen 2000): Use the
   exact constraint structure `[G, A^T; A, 0]` as a preconditioner, where G
   approximates the (1,1) block. Requires solving with G and with `A G^{-1} A^T`
   at each iteration, but guarantees bounded eigenvalues independent of mu.

2. **Augmented Lagrangian preconditioners**: Shift the system to improve
   conditioning at the cost of a modified operator. Works well when the shift
   parameter is tuned, but adds a tuning parameter.

3. **Multigrid / AMG on the (1,1) block**: If P has geometric structure (e.g.,
   discretized PDE), AMG can provide a scalable (1,1) block solve, making
   block preconditioners effective.

4. **Inexact interior point**: Allow the KKT solve tolerance to be O(mu), not
   O(mu^2), relaxing the accuracy requirement. This is theoretically justified
   (Bellavia, Gondzio, Morini 2012) and would make all iterative methods more
   viable, but requires careful IPM convergence analysis.
