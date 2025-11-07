import timeit

import numpy as np

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

        m, n, z = self.m, self.n, self.z

        # --- Initialization ---
        x = np.zeros(n) if x is None else x
        y = np.concatenate([np.zeros(z), np.ones(m - z)]) if y is None else y
        s = np.concatenate([np.zeros(z), np.ones(m - z)]) if s is None else s
        assert x.shape == (n,)
        assert y.shape == (m,)
        assert s.shape == (m,)
        τ = κ = 1.0

        # Store G(v0) used in Eq. (9a) as a three-element tuple corresponding
        # to the three rows of G
        Gv0 = self.G(x=x, y=y, s=s, τ=τ, κ=κ)

        # --- Verify strict interiority ---
        assert np.all(y[z:] > 0) and np.all(s[z:] > 0)
        assert np.all(s[:z] == 0.0)

        if self.equilibrate:
            a, p, b, c, self.d, self.e = self._equilibrate()
            x, y, s = self._equilibrate_iterates(x=x, y=y, s=s)
        else:
            a, p, b, c, self.d, self.e = self.a, self.p, self.b, self.c, None, None

        linear_solver = DirectKktSolver(
            a=a,
            p=p,
            z=z,
            min_static_regularization=min_static_regularization,
            max_iterative_refinement_steps=max_iterative_refinement_steps,
            atol=linear_solver_atol,
            rtol=linear_solver_rtol,
            solver=linear_solver.value(),
        )
        # This is the second RHS in (12). When we call 'solve' the
        # b row will be negated (and hence positive, as in the paper)
        constant_rhs = -np.concatenate([c, b])

        stats_list = []

        # --- Main iteration loop ---
        for self.it in range(max_iter):
            stats = {}

            # Eq. (9c)
            μ = s @ y / m

            # Pass mu=0 and let 'min_static_regularization' handle the
            # regularization they discuss in Section 3.3
            linear_solver.update(mu=0, s=s, y=y)

            # --- Step 0: Solve K @ Δ2 = q for Δ2 ---
            #   This calculation is half of Eq. (12)
            #   TODO: Should we use a different/better warm_start?
            Δ2, _ = linear_solver.solve(rhs=constant_rhs, warm_start=np.zeros(m + n))
            Δx2, Δy2 = np.split(Δ2, [n])
            rx, ry, rτ = self.compute_residuals(
                x=x, y=y, s=s, τ=τ, κ=κ, μ=μ, Gv0=Gv0
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
            σ = (1 - α_aff) ** 3
            σ = np.clip(σ, a_min=0, a_max=1)

            # --- Step 2: Corrector step ---
            η = Δs_aff * Δy_aff
            ds_cor = np.zeros(self.m, dtype="float")
            ds_cor[self.z :] = s[self.z :] / (
                s[self.z :] * y[self.z :] + η[self.z :] - σ * μ
            )
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
                dτ=(1 - σ) * τ,
                dκ=κ * τ + Δκ_aff * Δτ_aff - σ * μ,
            )
            α_cor = self._compute_step_size(y=y, s=s, d_y=Δy_cor, d_s=Δs_cor)

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
            #   These are the QTPT termination criteria, which do not necessarily
            #   agree perfectly with the termination criteria in Clarabel
            status = self._check_termination(x, y, [τ], s, α_cor, μ, σ, stats)
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

    def compute_residuals(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        τ: float,
        κ: float,
        μ: float,
        Gv0: tuple[np.ndarray, np.ndarray, float]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The residual G(v) - μG(v0) in Eq. (9a)."""
        Gv = self.G(x=x, y=y, s=s, τ=τ, κ=κ)
        rx = Gv[0] - μ * Gv0[0]
        ry = Gv[1] - μ * Gv0[1]
        rτ = Gv[2] - μ * Gv0[2]
        return rx, ry, rτ

    def G(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        τ: float,
        κ: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The non-linear operator in Eq. (4)."""
        row_1 = -self.p @ x - self.a.T @ y - τ * self.c
        row_2 = s + self.a @ x - τ * self.b
        row_3 = κ + self.c @ x + self.b @ y + x @ self.p @ x / τ
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
        Δ1, _ = linear_solver.solve(
            rhs=np.concatenate([dx, dy - ds]),
            warm_start=np.zeros(self.n + self.m),  # TODO: Better warm-start?
        )
        Δx1, Δy1 = np.split(Δ1, [self.n])

        ξ = x / τ
        Pξ = 2 * self.p @ ξ

        Δτ_num = dτ - dκ / τ + (2 * Pξ + self.c) @ Δx1 + self.b @ Δy1
        Δτ_den = κ / τ + ξ @ Pξ - (2 * Pξ + self.c) @ Δx2 - self.b @ Δy2
        Δτ = Δτ_num / Δτ_den
        Δκ = -(dκ + κ * Δτ) / τ

        Δx = Δx1 + Δτ * Δx2
        Δy = Δy1 + Δτ * Δy2
        Δs = np.zeros(self.m, dtype="float")
        Δs[self.z :] = -ds[self.z :] - s[self.z :] / y[self.z :] * Δy[self.z :]

        return Δx, Δy, Δs, Δτ, Δκ
