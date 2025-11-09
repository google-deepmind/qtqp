import timeit

import numpy as np
from scipy.sparse import csc_matrix

from qtqp import QTQP, Solution, LinearSolver, SolutionStatus
from qtqp.direct import DirectKktSolver


class Clarabel(QTQP):
    def solve(
        self,
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
        linear_solver: LinearSolver = LinearSolver.SCIPY,
        verbose: bool = True,
        equilibrate: bool = True,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        s: np.ndarray | None = None,
    ) -> Solution:
        """Implement basic Clarabel routine.

        The `y` variable in this code corresponds to `z` in the Clarabel paper.
        """
        self.start_time = timeit.default_timer()
        self.atol = atol
        self.rtol = rtol
        self.atol_infeas = atol_infeas
        self.rtol_infeas = rtol_infeas
        self.equilibrate = equilibrate

        # TODO: Does Clarabel support user initialization?
        assert x is None and y is None and s is None

        if self.equilibrate:
            a, p, b, c, self.d, self.e = self._equilibrate()
        else:
            a, p, b, c, self.d, self.e = self.a, self.p, self.b, self.c, None, None

        linear_solver = DirectKktSolver(
            a=a,
            p=p,
            z=self.z,
            min_static_regularization=min_static_regularization,
            max_iterative_refinement_steps=max_iterative_refinement_steps,
            atol=linear_solver_atol,
            rtol=linear_solver_rtol,
            solver=linear_solver.value(),
        )
        # This is the second RHS in (12). When we call 'solve' the
        # b row will be negated (and hence positive, as in the paper)
        constant_rhs = np.concat([-c, -b])

        # --- Initialization ---
        x, y, s = self.init_iterates(linear_solver=linear_solver, c=c, b=b, ϵ=1e-8)
        τ = κ = 1.0

        # --- Verify strict interiority ---
        assert np.all(y[self.z :] > 0) and np.all(s[self.z :] > 0)
        assert np.all(s[: self.z] == 0.0)

        stats_list = []

        # --- Main iteration loop ---
        for self.it in range(max_iter):
            stats = {}

            # Eq. (9c)
            μ = s @ y / self.m

            # Pass mu=0 and let 'min_static_regularization' handle the
            # regularization they discuss in Section 3.3
            linear_solver.update(mu=0, s=s, y=y)

            # --- Step 0: Solve K @ Δ2 = q for Δ2 ---
            #   This calculation is half of Eq. (12)
            #   TODO: Should we use a different/better warm_start?
            Δ2, _ = linear_solver.solve(
                rhs=constant_rhs, warm_start=np.zeros(self.m + self.n)
            )
            Δx2, Δy2 = np.split(Δ2, [self.n])
            rx, ry, rτ = self.compute_residuals(
                x=x, y=y, s=s, τ=τ, κ=κ, a=a, b=b, p=p, c=c
            )

            # --- Step 1: Predictor (affine) step ---
            Δx_aff, Δy_aff, Δs_aff, Δτ_aff, Δκ_aff = self.clarabel_newton_step(
                x=x,
                y=y,
                s=s,
                τ=τ,
                κ=κ,
                linear_solver=linear_solver,
                Δx2=Δx2,
                Δy2=Δy2,
                dx=rx,
                dy=ry,
                ds=s,
                dτ=rτ,
                dκ=κ * τ,
            )
            α_aff = self._compute_step_size(y=y, s=s, d_y=Δy_aff, d_s=Δs_aff)
            if Δτ_aff < 0:
                α_aff = min(α_aff, τ / -Δτ_aff)
            if Δκ_aff < 0:
                α_aff = min(α_aff, κ / -Δκ_aff)
            assert 0 <= α_aff <= 1
            σ = (1 - α_aff) ** 3

            # --- Step 2: Corrector step ---
            η = Δs_aff * Δy_aff
            ds_cor = np.zeros(self.m, dtype="float")
            ds_cor[self.z:] = s[self.z:] + (η[self.z:] - σ * μ) / y[self.z:]
            Δx_cor, Δy_cor, Δs_cor, Δτ_cor, Δκ_cor = self.clarabel_newton_step(
                x=x,
                y=y,
                s=s,
                τ=τ,
                κ=κ,
                linear_solver=linear_solver,
                Δx2=Δx2,
                Δy2=Δy2,
                dx=(1 - σ) * rx,
                dy=(1 - σ) * ry,
                ds=ds_cor,
                dτ=(1 - σ) * rτ,
                dκ=κ * τ + Δκ_aff * Δτ_aff - σ * μ,
            )
            α_cor = self._compute_step_size(y=y, s=s, d_y=Δy_cor, d_s=Δs_cor)
            if Δτ_cor < 0:
                α_cor = min(α_cor, τ / -Δτ_cor)
            if Δκ_cor < 0:
                α_cor = min(α_cor, κ / -Δκ_cor)
            assert 0 <= α_cor <= 1

            # --- Step 3: Update iterates ---
            x += step_size_scale * α_cor * Δx_cor
            y += step_size_scale * α_cor * Δy_cor
            s += step_size_scale * α_cor * Δs_cor
            τ += step_size_scale * α_cor * Δτ_cor
            κ += step_size_scale * α_cor * Δκ_cor

            # --- Ensure strict feasibility is maintained ---
            y[self.z :] = np.maximum(y[self.z :], 1e-30)
            s[self.z :] = np.maximum(s[self.z :], 1e-30)
            τ = np.maximum(τ, 1e-30)

            # --- Termination criteria ---
            #   These are the QPQT termination criteria, which do not necessarily
            #   agree perfectly with the termination criteria in Clarabel
            status = self._check_termination(
                x=x, y=y, tau_arr=[τ], s=s, alpha=α_cor, mu=μ, sigma=σ, stats_i=stats
            )
            stats_list.append(stats)
            match status:
                case SolutionStatus.SOLVED:
                    if self.equilibrate:
                        x, y, s = self._unequilibrate_iterates(x, y, s)
                    return Solution(x / τ, y / τ, s / τ, stats_list, status)
                case SolutionStatus.INFEASIBLE:
                    if self.equilibrate:
                        x, y, s = self._unequilibrate_iterates(x, y, s)
                    x.fill(np.nan)
                    s.fill(np.nan)
                    return Solution(x, y / abs(self.b @ y), s, stats_list, status)
                case SolutionStatus.UNBOUNDED:
                    if self.equilibrate:
                        x, y, s = self._unequilibrate_iterates(x, y, s)
                    y.fill(np.nan)
                    abs_ctx = abs(self.c @ x)
                    return Solution(x / abs_ctx, y, s / abs_ctx, stats_list, status)
                case SolutionStatus.FAILED:
                    if self.equilibrate:
                        x, y, s = self._unequilibrate_iterates(x, y, s)
                    return Solution(
                        x / τ, y / τ, s / τ, stats_list, SolutionStatus.FAILED
                    )
                case SolutionStatus.UNFINISHED:
                    pass
                case _:
                    raise ValueError(f"Unknown convergence status: {status}")

        if self.equilibrate:
            x, y, s = self._unequilibrate_iterates(x, y, s)
        return Solution(x / τ, y / τ, s / τ, stats_list, SolutionStatus.FAILED)

    @staticmethod
    def compute_residuals(
        *,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        τ: float,
        κ: float,
        a: csc_matrix,
        p: csc_matrix,
        c: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        rx, ry, rτ = Clarabel.G(x=x, y=y, s=s, τ=τ, κ=κ, a=a, b=b, p=p, c=c)
        return rx, ry, rτ

    @staticmethod
    def G(
        *,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        τ: float,
        κ: float,
        a: csc_matrix,
        p: csc_matrix,
        c: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The non-linear operator G given in Eq. (4)."""
        row_1 = -p @ x - a.T @ y - τ * c
        row_2 = s + a @ x - τ * b
        row_3 = κ + c @ x + b @ y + x @ p @ x / τ
        return row_1, row_2, row_3

    def clarabel_newton_step(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        τ: float,
        κ: float,
        linear_solver: DirectKktSolver,
        Δx2: np.ndarray,
        Δy2: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        ds: np.ndarray,
        dτ: float,
        dκ: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Equation (12)."""
        Δ1, _ = linear_solver.solve(
            rhs=np.concat([dx, dy - ds]),
            warm_start=np.zeros(self.n + self.m),  # TODO: Better warm-start?
        )
        Δx1, Δy1 = np.split(Δ1, [self.n])

        ξ = x / τ
        Pξ = self.p @ ξ

        Δτ_num = dτ - dκ / τ + (2 * Pξ + self.c) @ Δx1 + self.b @ Δy1
        Δτ_den = κ / τ + ξ @ Pξ - (2 * Pξ + self.c) @ Δx2 - self.b @ Δy2
        Δτ = Δτ_num / Δτ_den
        Δκ = -(dκ + κ * Δτ) / τ

        Δx = Δx1 + Δτ * Δx2
        Δy = Δy1 + Δτ * Δy2
        Δs = np.zeros(self.m, dtype="float")
        Δs[self.z :] = -ds[self.z :] - s[self.z :] / y[self.z :] * Δy[self.z :]

        return Δx, Δy, Δs, Δτ, Δκ

    def init_iterates(
        self,
        *,
        linear_solver: DirectKktSolver,
        c: np.ndarray,
        b: np.ndarray,
        ϵ: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute v0 as described in Section 2.4.1."""
        linear_solver.update(mu=0, s=np.ones(self.m), y=np.ones(self.m))
        if self.p.nnz > 0:
            xy, _ = linear_solver.solve(
                rhs=np.concat([-c, -b]),
                warm_start=np.zeros(self.m + self.n),
            )
            x, y = np.split(xy, [self.n])

            s = np.zeros_like(y)
            s[self.z :] = -y[self.z :].copy()
            if s[self.z :].min() < ϵ:
                s[self.z :] += ϵ - s[self.z :].min()

            if y[self.z :].min() < ϵ:
                y[self.z :] += ϵ - y[self.z :].min()
        else:
            xs, _ = linear_solver.solve(
                rhs=np.concat([np.zeros_like(c), -b]),
                warm_start=np.zeros(self.m + self.n),
            )
            x, s = np.split(xs, [self.n])

            s = -s.copy()
            if s[self.z :].min() < ϵ:
                s[self.z :] += ϵ - s[self.z :].min()

            xy, _ = linear_solver.solve(
                rhs=np.concat([-c, np.zeros_like(b)]),
                warm_start=np.zeros(self.m + self.n),
            )
            _, y = np.split(xy, [self.n])

            if y[self.z :].min() < ϵ:
                y[self.z :] += ϵ - y[self.z :].min()

        return x, y, s
