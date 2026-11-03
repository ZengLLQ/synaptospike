"""Hodgkin-Huxley (1952) Biophysical Conductance Model.

Simulates the non-linear action potential dynamics of the squid giant axon
governed by voltage-dependent sodium (Na+), potassium (K+), and leak (L)
ionic currents across the lipid bilayer membrane.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal, Tuple, List
import numpy as np


@dataclasses.dataclass(frozen=True)
class HHParameters:
    """Biophysical parameters for Hodgkin-Huxley membrane dynamics."""

    # Membrane capacitance (uF / cm^2)
    c_m: float = 1.0

    # Maximal ionic conductances (mS / cm^2)
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3

    # Nernst reversal potentials (mV)
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.387

    # Resting potential baseline (mV)
    v_rest: float = -65.0

    # Threshold for action potential detection (mV)
    spike_threshold: float = 0.0


@dataclasses.dataclass
class HHState:
    """Instantaneous state of the Hodgkin-Huxley membrane."""

    v: float  # Membrane potential (mV)
    m: float  # Sodium activation gating variable [0, 1]
    h: float  # Sodium inactivation gating variable [0, 1]
    n: float  # Potassium activation gating variable [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.m, self.h, self.n], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> HHState:
        return cls(v=float(arr[0]), m=float(arr[1]), h=float(arr[2]), n=float(arr[3]))


@dataclasses.dataclass
class HHSolution:
    """Simulation trajectory and computed biophysical observables."""

    time: np.ndarray  # Time grid (ms)
    v: np.ndarray  # Membrane potential trajectory (mV)
    m: np.ndarray  # Na+ activation trajectory
    h: np.ndarray  # Na+ inactivation trajectory
    n: np.ndarray  # K+ activation trajectory
    i_na: np.ndarray  # Sodium current trajectory (uA / cm^2)
    i_k: np.ndarray  # Potassium current trajectory (uA / cm^2)
    i_l: np.ndarray  # Leak current trajectory (uA / cm^2)
    i_inj: np.ndarray  # Injected stimulus current (uA / cm^2)
    spike_times: np.ndarray  # Detected spike times (ms)
    firing_rate_hz: float  # Mean firing frequency (Hz)
    inter_spike_intervals: np.ndarray  # ISI intervals (ms)


class HodgkinHuxley:
    """Hodgkin-Huxley membrane dynamics solver."""

    def __init__(self, params: HHParameters | None = None) -> None:
        self.params = params or HHParameters()

    @staticmethod
    def _vtrap(x: float, y: float) -> float:
        """Singularity-safe evaluation of x / (exp(x / y) - 1)."""
        if abs(x / y) < 1e-7:
            return y * (1.0 - 0.5 * (x / y))
        return x / (np.exp(x / y) - 1.0)

    def alpha_m(self, v: float) -> float:
        """Sodium activation forward rate alpha_m(V) (ms^-1)."""
        # Modern shifted convention relative to resting potential
        v_rel = v - self.params.v_rest
        return 0.1 * self._vtrap(25.0 - v_rel, 10.0)

    def beta_m(self, v: float) -> float:
        """Sodium activation backward rate beta_m(V) (ms^-1)."""
        v_rel = v - self.params.v_rest
        return 4.0 * np.exp(-v_rel / 18.0)

    def alpha_h(self, v: float) -> float:
        """Sodium inactivation forward rate alpha_h(V) (ms^-1)."""
        v_rel = v - self.params.v_rest
        return 0.07 * np.exp(-v_rel / 20.0)

    def beta_h(self, v: float) -> float:
        """Sodium inactivation backward rate beta_h(V) (ms^-1)."""
        v_rel = v - self.params.v_rest
        return 1.0 / (np.exp((30.0 - v_rel) / 10.0) + 1.0)

    def alpha_n(self, v: float) -> float:
        """Potassium activation forward rate alpha_n(V) (ms^-1)."""
        v_rel = v - self.params.v_rest
        return 0.01 * self._vtrap(10.0 - v_rel, 10.0)

    def beta_n(self, v: float) -> float:
        """Potassium activation backward rate beta_n(V) (ms^-1)."""
        v_rel = v - self.params.v_rest
        return 0.125 * np.exp(-v_rel / 80.0)

    def steady_state_gating(self, v: float) -> Tuple[float, float, float]:
        """Compute asymptotic gating values (m_inf, h_inf, n_inf) at voltage v."""
        am, bm = self.alpha_m(v), self.beta_m(v)
        ah, bh = self.alpha_h(v), self.beta_h(v)
        an, bn = self.alpha_n(v), self.beta_n(v)

        m_inf = am / (am + bm)
        h_inf = ah / (ah + bh)
        n_inf = an / (an + bn)
        return m_inf, h_inf, n_inf

    def time_constants(self, v: float) -> Tuple[float, float, float]:
        """Compute relaxation time constants (tau_m, tau_h, tau_n) in ms."""
        tau_m = 1.0 / (self.alpha_m(v) + self.beta_m(v))
        tau_h = 1.0 / (self.alpha_h(v) + self.beta_h(v))
        tau_n = 1.0 / (self.alpha_n(v) + self.beta_n(v))
        return tau_m, tau_h, tau_n

    def equilibrium_state(self, v_init: float | None = None) -> HHState:
        """Generate resting equilibrium initial state."""
        v = v_init if v_init is not None else self.params.v_rest
        m_inf, h_inf, n_inf = self.steady_state_gating(v)
        return HHState(v=v, m=m_inf, h=h_inf, n=n_inf)

    def derivatives(self, state: HHState, i_inj: float) -> np.ndarray:
        """Evaluate d[V, m, h, n]/dt."""
        v, m, h, n = state.v, state.m, state.h, state.n

        # Ionic currents
        i_na = self.params.g_na * (m**3) * h * (v - self.params.e_na)
        i_k = self.params.g_k * (n**4) * (v - self.params.e_k)
        i_l = self.params.g_l * (v - self.params.e_l)

        # Membrane potential derivative: C_m * dV/dt = I_inj - I_ion
        dv_dt = (i_inj - i_na - i_k - i_l) / self.params.c_m

        # Gating kinetics: dx/dt = alpha_x*(1-x) - beta_x*x
        dm_dt = self.alpha_m(v) * (1.0 - m) - self.beta_m(v) * m
        dh_dt = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn_dt = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n

        return np.array([dv_dt, dm_dt, dh_dt, dn_dt], dtype=np.float64)

    def step_rk4(self, state: HHState, i_inj: float, dt: float) -> HHState:
        """Fourth-order Runge-Kutta numerical step."""
        y0 = state.to_array()

        k1 = self.derivatives(state, i_inj)

        s2 = HHState.from_array(y0 + 0.5 * dt * k1)
        k2 = self.derivatives(s2, i_inj)

        s3 = HHState.from_array(y0 + 0.5 * dt * k2)
        k3 = self.derivatives(s3, i_inj)

        s4 = HHState.from_array(y0 + dt * k3)
        k4 = self.derivatives(s4, i_inj)

        y_next = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Clamp gating probabilities to physical interval [0, 1]
        y_next[1:] = np.clip(y_next[1:], 0.0, 1.0)
        return HHState.from_array(y_next)

    def step_exponential_euler(self, state: HHState, i_inj: float, dt: float) -> HHState:
        """Exponential Euler step (exact integration for linear gating kinetics)."""
        v, m, h, n = state.v, state.m, state.h, state.n

        # Membrane current evaluated at current gating
        i_na = self.params.g_na * (m**3) * h * (v - self.params.e_na)
        i_k = self.params.g_k * (n**4) * (v - self.params.e_k)
        i_l = self.params.g_l * (v - self.params.e_l)
        dv_dt = (i_inj - i_na - i_k - i_l) / self.params.c_m
        v_next = v + dt * dv_dt

        # Exponential gating update
        m_inf, h_inf, n_inf = self.steady_state_gating(v)
        tau_m, tau_h, tau_n = self.time_constants(v)

        m_next = m_inf + (m - m_inf) * np.exp(-dt / tau_m)
        h_next = h_inf + (h - h_inf) * np.exp(-dt / tau_h)
        n_next = n_inf + (n - n_inf) * np.exp(-dt / tau_n)

        return HHState(
            v=float(v_next),
            m=float(np.clip(m_next, 0.0, 1.0)),
            h=float(np.clip(h_next, 0.0, 1.0)),
            n=float(np.clip(n_next, 0.0, 1.0)),
        )

    def simulate(
        self,
        t_span: Tuple[float, float],
        dt: float,
        stimulus: float | Callable[[float], float] = 0.0,
        initial_state: HHState | None = None,
        method: Literal["rk4", "exponential_euler"] = "rk4",
    ) -> HHSolution:
        """Run numerical simulation over time span (t_start, t_end) with step size dt (ms)."""
        t_start, t_end = t_span
        time_grid = np.arange(t_start, t_end + dt * 0.5, dt)
        num_steps = len(time_grid)

        # Injected stimulus array
        if callable(stimulus):
            i_inj_arr = np.array([stimulus(t) for t in time_grid], dtype=np.float64)
        else:
            i_inj_arr = np.full(num_steps, float(stimulus), dtype=np.float64)

        # Allocate trajectory arrays
        v_arr = np.zeros(num_steps, dtype=np.float64)
        m_arr = np.zeros(num_steps, dtype=np.float64)
        h_arr = np.zeros(num_steps, dtype=np.float64)
        n_arr = np.zeros(num_steps, dtype=np.float64)

        curr_state = initial_state if initial_state is not None else self.equilibrium_state()

        step_fn = self.step_rk4 if method == "rk4" else self.step_exponential_euler

        for idx, t in enumerate(time_grid):
            v_arr[idx] = curr_state.v
            m_arr[idx] = curr_state.m
            h_arr[idx] = curr_state.h
            n_arr[idx] = curr_state.n

            if idx < num_steps - 1:
                curr_state = step_fn(curr_state, i_inj_arr[idx], dt)

        # Compute ionic currents
        i_na_arr = self.params.g_na * (m_arr**3) * h_arr * (v_arr - self.params.e_na)
        i_k_arr = self.params.g_k * (n_arr**4) * (v_arr - self.params.e_k)
        i_l_arr = self.params.g_l * (v_arr - self.params.e_l)

        # Spike detection
        spike_indices = self.detect_spikes(v_arr, threshold=self.params.spike_threshold)
        spike_times = time_grid[spike_indices]

        # Inter-spike intervals and firing rate
        if len(spike_times) >= 2:
            isis = np.diff(spike_times)
            duration_s = (t_end - t_start) / 1000.0
            firing_rate = len(spike_times) / duration_s if duration_s > 0 else 0.0
        else:
            isis = np.array([], dtype=np.float64)
            firing_rate = 0.0

        return HHSolution(
            time=time_grid,
            v=v_arr,
            m=m_arr,
            h=h_arr,
            n=n_arr,
            i_na=i_na_arr,
            i_k=i_k_arr,
            i_l=i_l_arr,
            i_inj=i_inj_arr,
            spike_times=spike_times,
            firing_rate_hz=firing_rate,
            inter_spike_intervals=isis,
        )

    @staticmethod
    def detect_spikes(v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Identify local action potential maxima crossing the voltage threshold."""
        crossings = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
        spike_peaks: List[int] = []

        for idx in crossings:
            # Search forward in a 20-sample window for local peak
            window_end = min(idx + 25, len(v))
            local_max_idx = idx + int(np.argmax(v[idx:window_end]))
            # Prevent duplicate registration of the same spike
            if not spike_peaks or (local_max_idx - spike_peaks[-1] > 10):
                spike_peaks.append(local_max_idx)

        return np.array(spike_peaks, dtype=np.int64)
