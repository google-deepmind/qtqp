# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
QTQP is pre-1.0: the version number promises no API stability, and the public
surface is the one listed in `qtqp.__all__`.

Entries below 0.0.7 were reconstructed from the commit history at the time
`CHANGELOG.md` was introduced, so they summarise each release rather than
recording it contemporaneously.

## [Unreleased]

### Added

-   Optional solver backends are declared as installable extras: `qdldl`,
    `gpu-cu12` and `gpu-cu13` (#158).
-   Quadratic matrix dimensions are validated at construction (#143).

### Changed

-   `AUTO` chooses its backend from the problem data rather than the platform
    alone (#142).
-   The true-KKT operator is promoted to the public seam so tests no longer
    reach through to the backend (#159).
-   GPU refinement solves and matvecs are combined behind the backend
    interface, and the cuDSS transpose view and diagonal upload buffer are
    reused (#137, #138).
-   `direct.py` no longer imports its backend modules, breaking the import
    cycle; a test now enforces it (#152).
-   Docstring coverage raised on the solver backend modules (#153).
-   `CONTRIBUTING.md` documents the checks, and coverage is stated as reported
    rather than gated (#151, #157).

## [0.0.7] - 2026-09-06

### Changed

-   Invariant matrix products are reused across Newton substeps, and
    warm-start screening products are reused rather than recomputed
    (#129, #134).
-   Per-iteration Python overhead cut via a cached transpose, GMRES buffers
    and direct norms (#127).
-   Dense backend setup avoids densifying the full KKT matrix (#128).
-   The embedding gate uses working-frame complementarity (#132).
-   Completed IPM steps are reported separately from iteration labels (#131).

### Fixed

-   Initialization factorization failures are reported as `FAILED` rather than
    surfacing as a numeric error (#135).
-   Initialization statistics are reset for each solve (#133).
-   Zero equality regularization errors in the dense backends now explain
    themselves (#130).

## [0.0.6] - 2026-09-04

### Added

-   Gondzio multiple centrality correctors, defaulting to 1 (#81).
-   Certified warm starts via the distance-to-path certificate, with warm
    starts certified before the cold initialization (#86, #107, #119).
-   Path diagnostics `delta_path`, `delta_path_local` and `lambda_init`,
    gated on `collect_stats` (#84, #85, #106, #111).
-   `ALMOST_SOLVED` status, and `SolutionStatus.FAILED` made reachable by
    converting numeric failures into a status (#93, #97, #98).
-   A ruff lint step in CI (E9/F rules) (#94).

### Changed

-   Clarabel's initialization is matched exactly, and `z == m` (all-equality)
    problems are solved through the direct KKT path (#120, #123).
-   Clarabel's termination criteria adopted, judged by data-scaled backward
    error (#80, #105, #110, #122).
-   GMRES refinement is the default, with a full-budget restart length (#109).
-   The adaptive fraction-to-boundary step schedule is enabled by default,
    with the swept 0.9999 endgame cap restored (#82, #108).
-   Ruiz equilibration covers the `(b, c)` and objective scalars (#124).
-   The fused corrector slack update is unconditional (#113).

### Fixed

-   `NameError` in `CupyDenseSolver.update_diag` (#87).
-   GMRES Givens breakdown guarded; duplicate CSC entries canonicalized (#92).
-   `P` symmetry tolerance made relative to data scale (#91).
-   Finiteness of `a` and `c` validated at construction (#90).
-   Representation noise tolerated in the presolve infinity sentinel (#95).
-   Materially degrading corrections rolled back on Richardson stall (#88).

### Removed

-   `init_strategy`: the CVXOPT initialization is the only initialization
    (#112).
-   `central_path_exponent`, which was a false lever (#104).
-   The unproven factorization retry ladder (#121).

## [0.0.5] - 2026-04-13

### Added

-   `LinearSolver.AUTO` backend selection (#57).

### Changed

-   Sparse linear solver backends and the core IPM loop optimized (#59).

## [0.0.4] - 2026-04-10

### Added

-   `ACCELERATE` backend on macOS (#49).
-   `DENSE_LDLT` and `CUPY_DENSE` backends (#37).
-   `UMFPACK` and dense LU backends, with a cleaned-up `LinearSolver` API
    (#32).
-   Per-solver statistics and a solve-time histogram in CI output (#36).
-   A weekly CI schedule (#43).

### Changed

-   Solver backends split out of `direct.py` into separate modules (#42).
-   The direct solver uses triangular KKT storage, and `set_kkt` is split into
    `set_kkt` + `update_diag` (#54, #55).
-   Dense backends use a Gram/Cholesky reduction (#40).
-   `PARDISO` is provided by `py-mkl-pardiso` instead of `pydiso`, with iparms
    tuned for IPM numerical stability (#39, #52).
-   Python 3.10 dropped; scipy >= 1.16 required (#51).
-   Minimum static regularization reduced to 1e-8 (#20).

### Fixed

-   MUMPS crash with complex PETSc builds, and MUMPS tests re-enabled
    (#41, #50).
-   Sigma calculation (#25).
-   Equilibration for zero rows and columns (#21).
-   Test reproducibility (#24).

## [0.0.3] - 2025-10-30

### Added

-   MUMPS backend (#8).

### Changed

-   The Eigen backend is provided by `nanoeigenpy` instead of `eigenpy`.

## [0.0.2] - 2025-10-26

Initial public release.

### Added

-   The QTQP interior-point QP/LP solver, with the homogeneous self-dual
    embedding so primal/dual infeasibility is detected without a separate
    Phase I.
-   Eigen linear solver backend.
-   API reference and linear-solver guidance in `README.md`.
-   CI workflow and Dependabot configuration.

[Unreleased]: https://github.com/google-deepmind/qtqp/compare/v0.0.7...HEAD
[0.0.7]: https://github.com/google-deepmind/qtqp/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/google-deepmind/qtqp/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/google-deepmind/qtqp/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/google-deepmind/qtqp/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/google-deepmind/qtqp/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/google-deepmind/qtqp/releases/tag/v0.0.2
