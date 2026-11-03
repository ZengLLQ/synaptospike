"""Morris-Lecar (1981) 2D Nonlinear Phase-Plane Model.

Simulates the excitability, nullclines, equilibrium states, and bifurcation
mechanisms of the barnacle giant muscle fiber with voltage-sensitive
calcium (Ca2+) and potassium (K+) conductances.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple, List, Optional
import numpy as np
from scipy import optimize


@dataclasses.dataclass(frozen=True)
class MLParameters:
    """Biophysical and kinetic parameters for Morris-Lecar dynamics."""

    # Membrane capacitance (uF / cm^2)
    c: float = 20.0

    # Maximal conductances (mS / cm^2)
    g_ca: float = 4.4
    g_k: float = 8.0
    g_l: float = 2.0

    # Reversal potentials (mV)
    v_ca: float = 120.0
    v_k: float = -84.0
    v_l: float = -60.0

    # Activation voltage parameters (mV)
    v1: float = -1.2
    v2: float = 18.0
    v3: float = 2.0
    v4: float = 30.0

    # Potassium kinetics temperature/scaling factor (ms^-1)
    phi: float = 0.04

    @classmethod
    def hopf_regime(cls) -> MLParameters:
        """Type II excitability preset (Subcritical Hopf bifurcation)."""
        return cls(v3=2.0, v4=30.0, phi=0.04, g_ca=4.4)

    @classmethod
    def snic_regime(cls) -> MLParameters:
        """Type I excitability preset (Saddle-Node on Invariant Circle bifurcation)."""
        return cls(v3=12.0, v4=17.4, phi=0.067, g_ca=4.0)


@dataclasses.dataclass
class MLState:
    """Instantaneous state of the Morris-Lecar oscillator."""

    v: float  # Membrane potential (mV)
    w: float  # Potassium activation recovery variable [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.w], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> MLState:
        return cls(v=float(arr[0]), w=float(arr[1]))


@dataclasses.dataclass
class MLPhasePlaneAnalysis:
    """Phase-plane topological data and stability characteristics."""

    v_grid: np.ndarray  # Voltage axis for nullclines (mV)
    v_nullcline: np.ndarray  # w(V) along V-nullcline
    w_nullcline: np.ndarray  # w(V) along w-nullcline
    fixed_points: List[Tuple[float, float]]  # (V*, w*) equilibria
    jacobians: List[np.ndarray]  # 2x2 Jacobian matrices at equilibria
    eigenvalues: List[np.ndarray]  # Complex eigenvalues at equilibria
    stability: List[str]  # Stability labels ("stable_node", "saddle", "unstable_spiral", etc.)


class MorrisLecar:
    """Morris-Lecar 2D dynamical system and phase-plane analyzer."""

    def __init__(self, params: MLParameters | None = None) -> None:
        self.params = params or MLParameters()

    def m_inf(self, v: float | np.ndarray) -> float | np.ndarray:
        """Fast calcium activation steady-state m_inf(V)."""
        return 0.5 * (1.0 + np.tanh((v - self.params.v1) / self.params.v2))

    def w_inf(self, v: float | np.ndarray) -> float | np.ndarray:
        """Potassium recovery activation steady-state w_inf(V)."""
        return 0.5 * (1.0 + np.tanh((v - self.params.v3) / self.params.v4))

    def tau_w(self, v: float | np.ndarray) -> float | np.ndarray:
        """Potassium relaxation time constant tau_w(V) (ms)."""
        return 1.0 / np.cosh((v - self.params.v3) / (2.0 * self.params.v4))

    def derivatives(self, state: MLState, i_inj: float) -> np.ndarray:
        """Vector field [dV/dt, dw/dt]."""
        v, w = state.v, state.w

        m_inf_v = self.m_inf(v)
        i_ca = self.params.g_ca * m_inf_v * (v - self.params.v_ca)
        i_k = self.params.g_k * w * (v - self.params.v_k)
        i_l = self.params.g_l * (v - self.params.v_l)

        dv_dt = (i_inj - i_ca - i_k - i_l) / self.params.c
        dw_dt = self.params.phi * (self.w_inf(v) - w) / self.tau_w(v)

        return np.array([dv_dt, dw_dt], dtype=np.float64)

    def compute_v_nullcline(self, v_grid: np.ndarray, i_inj: float) -> np.ndarray:
        """Solve dV/dt = 0 for w as a function of V."""
        m_inf_v = self.m_inf(v_grid)
        numerator = i_inj - self.params.g_ca * m_inf_v * (v_grid - self.params.v_ca) - self.params.g_l * (v_grid - self.params.v_l)
        denominator = self.params.g_k * (v_grid - self.params.v_k)

        # Handle singularity where V == V_K
        with np.errstate(divide="ignore", invalid="ignore"):
            w_null = numerator / denominator
        return w_null

    def compute_w_nullcline(self, v_grid: np.ndarray) -> np.ndarray:
        """Solve dw/dt = 0 for w as a function of V: w = w_inf(V)."""
        return self.w_inf(v_grid)

    def jacobian(self, v: float, w: float, i_inj: float) -> np.ndarray:
        """Analytical 2x2 Jacobian matrix evaluated at (V, w)."""
        m_inf_v = self.m_inf(v)
        # dm_inf/dV
        dm_dv = 0.5 / (self.params.v2 * (np.cosh((v - self.params.v1) / self.params.v2) ** 2))

        # d(dV/dt)/dV
        d_dv_dv = -(
            self.params.g_ca * dm_dv * (v - self.params.v_ca)
            + self.params.g_ca * m_inf_v
            + self.params.g_k * w
            + self.params.g_l
        ) / self.params.c

        # d(dV/dt)/dw
        d_dv_dw = -self.params.g_k * (v - self.params.v_k) / self.params.c

        # dw_inf/dV
        dw_inf_dv = 0.5 / (self.params.v4 * (np.cosh((v - self.params.v3) / self.params.v4) ** 2))

        # d(tau_w)/dV
        cosh_arg = (v - self.params.v3) / (2.0 * self.params.v4)
        d_tau_w_dv = -np.sinh(cosh_arg) / (2.0 * self.params.v4 * (np.cosh(cosh_arg) ** 2))

        tau = self.tau_w(v)
        w_inf_v = self.w_inf(v)

        # d(dw/dt)/dV via quotient rule: phi * (dw_inf/dV * tau - (w_inf - w) * d_tau/dV) / tau^2
        d_dw_dv = self.params.phi * (dw_inf_dv * tau - (w_inf_v - w) * d_tau_w_dv) / (tau**2)

        # d(dw/dt)/dw
        d_dw_dw = -self.params.phi / tau

        return np.array([[d_dv_dv, d_dv_dw], [d_dw_dv, d_dw_dw]], dtype=np.float64)

    def find_equilibria(
        self, i_inj: float, v_range: Tuple[float, float] = (-80.0, 50.0), num_guesses: int = 50
    ) -> List[Tuple[float, float]]:
        """Find all real equilibrium fixed points (V*, w*) within the voltage interval."""
        def root_func(v_val: float) -> float:
            w_target = float(self.w_inf(v_val))
            res = self.derivatives(MLState(v=v_val, w=w_target), i_inj)
            return float(res[0])

        v_guesses = np.linspace(v_range[0], v_range[1], num_guesses)
        found_roots: List[float] = []

        for guess in v_guesses:
            try:
                sol = optimize.root_scalar(root_func, x0=guess, bracket=[guess - 5.0, guess + 5.0], method="brentq")
                if sol.converged:
                    r = round(float(sol.root), 4)
                    if r not in [round(x, 4) for x in found_roots]:
                        found_roots.append(sol.root)
            except (ValueError, RuntimeError):
                # Search without brackets using hybr
                try:
                    sol_hybr = optimize.root(root_func, x0=guess, method="hybr")
                    if sol_hybr.success:
                        r = round(float(sol_hybr.x[0]), 4)
                        if v_range[0] <= r <= v_range[1] and r not in [round(x, 4) for x in found_roots]:
                            found_roots.append(float(sol_hybr.x[0]))
                except Exception:
                    pass

        equilibria: List[Tuple[float, float]] = []
        for v_eq in sorted(found_roots):
            w_eq = float(self.w_inf(v_eq))
            equilibria.append((v_eq, w_eq))

        return equilibria

    def analyze_phase_plane(
        self, i_inj: float, v_range: Tuple[float, float] = (-80.0, 50.0), grid_points: int = 300
    ) -> MLPhasePlaneAnalysis:
        """Comprehensive phase-plane nullcline and stability analysis."""
        v_grid = np.linspace(v_range[0], v_range[1], grid_points)
        v_null = self.compute_v_nullcline(v_grid, i_inj)
        w_null = self.compute_w_nullcline(v_grid)

        fixed_points = self.find_equilibria(i_inj, v_range)
        jacobians: List[np.ndarray] = []
        eigenvalues_list: List[np.ndarray] = []
        stability_labels: List[str] = []

        for v_star, w_star in fixed_points:
            j = self.jacobian(v_star, w_star, i_inj)
            eigs = np.linalg.eigvals(j)
            jacobians.append(j)
            eigenvalues_list.append(eigs)

            real_parts = np.real(eigs)
            imag_parts = np.imag(eigs)

            if np.all(real_parts < 0):
                label = "stable_spiral" if np.any(np.abs(imag_parts) > 1e-6) else "stable_node"
            elif np.all(real_parts > 0):
                label = "unstable_spiral" if np.any(np.abs(imag_parts) > 1e-6) else "unstable_node"
            elif real_parts[0] * real_parts[1] < 0:
                label = "saddle"
            else:
                label = "center_or_bifurcation"

            stability_labels.append(label)

        return MLPhasePlaneAnalysis(
            v_grid=v_grid,
            v_nullcline=v_null,
            w_nullcline=w_null,
            fixed_points=fixed_points,
            jacobians=jacobians,
            eigenvalues=eigenvalues_list,
            stability=stability_labels,
        )

    def step_rk4(self, state: MLState, i_inj: float, dt: float) -> MLState:
        """Runge-Kutta 4th-order integration step."""
        y0 = state.to_array()

        k1 = self.derivatives(state, i_inj)
        s2 = MLState.from_array(y0 + 0.5 * dt * k1)
        k2 = self.derivatives(s2, i_inj)
        s3 = MLState.from_array(y0 + 0.5 * dt * k2)
        k3 = self.derivatives(s3, i_inj)
        s4 = MLState.from_array(y0 + dt * k3)
        k4 = self.derivatives(s4, i_inj)

        y_next = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y_next[1] = np.clip(y_next[1], 0.0, 1.0)
        return MLState.from_array(y_next)

    def simulate(
        self,
        t_span: Tuple[float, float],
        dt: float,
        i_inj: float = 0.0,
        initial_state: MLState | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate Morris-Lecar dynamics, returning (time, v, w)."""
        t_start, t_end = t_span
        time_grid = np.arange(t_start, t_end + dt * 0.5, dt)
        num_steps = len(time_grid)

        v_arr = np.zeros(num_steps, dtype=np.float64)
        w_arr = np.zeros(num_steps, dtype=np.float64)

        if initial_state is None:
            # Default to equilibrium near resting potential
            eqs = self.find_equilibria(i_inj)
            initial_state = MLState(v=eqs[0][0], w=eqs[0][1]) if eqs else MLState(v=-60.0, w=float(self.w_inf(-60.0)))

        curr_state = initial_state
        for idx in range(num_steps):
            v_arr[idx] = curr_state.v
            w_arr[idx] = curr_state.w
            if idx < num_steps - 1:
                curr_state = self.step_rk4(curr_state, i_inj, dt)

        return time_grid, v_arr, w_arr
