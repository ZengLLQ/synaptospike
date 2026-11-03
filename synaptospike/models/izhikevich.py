"""Izhikevich (2003) Phenomenological Spiking Neuron & Population Model.

Combines the computational efficiency of integrate-and-fire mechanisms
with the rich neurocomputational firing repertoires of biophysical Hodgkin-Huxley
conductance systems.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Tuple, List, Dict
import numpy as np


class IzhikevichPreset(Enum):
    """Canonical cortical neurocomputational firing patterns."""

    REGULAR_SPIKING = "regular_spiking"
    INTRINSICALLY_BURSTING = "intrinsically_bursting"
    CHATTERING = "chattering"
    FAST_SPIKING = "fast_spiking"
    LOW_THRESHOLD_SPIKING = "low_threshold_spiking"
    THALAMO_CORTICAL = "thalamo_cortical"
    RESONATOR = "resonator"


@dataclasses.dataclass(frozen=True)
class IzhikevichParameters:
    """Izhikevich dimensionless four-parameter dynamical tuple (a, b, c, d)."""

    a: float  # Recovery variable decay rate
    b: float  # Sensitivity of recovery to subthreshold potential
    c: float  # After-spike voltage reset potential (mV)
    d: float  # After-spike recovery variable increment
    v_peak: float = 30.0  # Cutoff spike threshold (mV)

    @classmethod
    def from_preset(cls, preset: IzhikevichPreset) -> IzhikevichParameters:
        """Instantiate parameters from canonical neurobiological firing types."""
        preset_map: Dict[IzhikevichPreset, Tuple[float, float, float, float]] = {
            IzhikevichPreset.REGULAR_SPIKING: (0.02, 0.2, -65.0, 8.0),
            IzhikevichPreset.INTRINSICALLY_BURSTING: (0.02, 0.2, -55.0, 4.0),
            IzhikevichPreset.CHATTERING: (0.02, 0.2, -50.0, 2.0),
            IzhikevichPreset.FAST_SPIKING: (0.1, 0.2, -65.0, 2.0),
            IzhikevichPreset.LOW_THRESHOLD_SPIKING: (0.02, 0.25, -65.0, 2.0),
            IzhikevichPreset.THALAMO_CORTICAL: (0.02, 0.25, -65.0, 0.05),
            IzhikevichPreset.RESONATOR: (0.1, 0.26, -65.0, 2.0),
        }
        a, b, c, d = preset_map[preset]
        return cls(a=a, b=b, c=c, d=d)


@dataclasses.dataclass
class IzhikevichState:
    """State vector of a single Izhikevich neuron."""

    v: float  # Membrane potential (mV)
    u: float  # Membrane recovery variable

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.u], dtype=np.float64)


class IzhikevichNeuron:
    """Single Izhikevich spiking neuron simulator."""

    def __init__(self, params: IzhikevichParameters | None = None) -> None:
        self.params = params or IzhikevichParameters.from_preset(IzhikevichPreset.REGULAR_SPIKING)
        self.state = IzhikevichState(v=self.params.c, u=self.params.b * self.params.c)

    def reset(self, v_init: float | None = None) -> None:
        """Reset membrane state to resting equilibrium."""
        v = v_init if v_init is not None else self.params.c
        self.state = IzhikevichState(v=v, u=self.params.b * v)

    def step(self, i_inj: float, dt: float = 0.5) -> bool:
        """Advance neuron state by dt ms using two-step half-Euler update. Returns True if spiked."""
        v = self.state.v
        u = self.state.u

        # Two 0.5*dt steps for v to preserve numerical stability
        half_dt = 0.5 * dt
        v += half_dt * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_inj)
        v += half_dt * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_inj)
        u += dt * (self.params.a * (self.params.b * v - u))

        spiked = False
        if v >= self.params.v_peak:
            # Action potential peak registered, reset state
            spiked = True
            v = self.params.c
            u += self.params.d

        self.state.v = float(v)
        self.state.u = float(u)
        return spiked

    def simulate(
        self,
        t_span: Tuple[float, float],
        dt: float = 0.5,
        i_inj: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate single neuron trajectory over time span, returning (time, v, u, spike_times)."""
        t_start, t_end = t_span
        time_grid = np.arange(t_start, t_end + dt * 0.5, dt)
        num_steps = len(time_grid)

        v_arr = np.zeros(num_steps, dtype=np.float64)
        u_arr = np.zeros(num_steps, dtype=np.float64)
        spike_times: List[float] = []

        self.reset()
        for idx, t in enumerate(time_grid):
            # Record trajectory (pre-reset peak visibility)
            v_curr = self.state.v
            u_curr = self.state.u

            spiked = self.step(i_inj, dt)
            if spiked:
                v_arr[idx] = self.params.v_peak  # Mark spike peak on trajectory
                spike_times.append(t)
            else:
                v_arr[idx] = v_curr

            u_arr[idx] = u_curr

        return time_grid, v_arr, u_arr, np.array(spike_times, dtype=np.float64)


class IzhikevichPopulation:
    """Vectorized simulation of an N-neuron heterogeneous population."""

    def __init__(
        self,
        num_neurons: int,
        a: np.ndarray | float = 0.02,
        b: np.ndarray | float = 0.2,
        c: np.ndarray | float = -65.0,
        d: np.ndarray | float = 8.0,
        v_peak: float = 30.0,
    ) -> None:
        self.num_neurons = num_neurons
        self.v_peak = v_peak

        # Broadcast scalar parameters to arrays
        self.a = np.full(num_neurons, a) if np.isscalar(a) else np.array(a, dtype=np.float64)
        self.b = np.full(num_neurons, b) if np.isscalar(b) else np.array(b, dtype=np.float64)
        self.c = np.full(num_neurons, c) if np.isscalar(c) else np.array(c, dtype=np.float64)
        self.d = np.full(num_neurons, d) if np.isscalar(d) else np.array(d, dtype=np.float64)

        # Vector states
        self.v = np.copy(self.c)
        self.u = self.b * self.v

    def reset(self) -> None:
        """Reset all neurons in population."""
        self.v = np.copy(self.c)
        self.u = self.b * self.v

    def step(self, i_inj: np.ndarray | float, dt: float = 0.5) -> np.ndarray:
        """Vectorized state update. Returns boolean mask of spiked neuron indices."""
        if np.isscalar(i_inj):
            i_vec = np.full(self.num_neurons, i_inj, dtype=np.float64)
        else:
            i_vec = np.asarray(i_inj, dtype=np.float64)

        half_dt = 0.5 * dt
        self.v += half_dt * (0.04 * (self.v**2) + 5.0 * self.v + 140.0 - self.u + i_vec)
        self.v += half_dt * (0.04 * (self.v**2) + 5.0 * self.v + 140.0 - self.u + i_vec)
        self.u += dt * (self.a * (self.b * self.v - self.u))

        # Detect spikes
        spiked_mask = self.v >= self.v_peak

        # Apply reset rule
        self.v[spiked_mask] = self.c[spiked_mask]
        self.u[spiked_mask] += self.d[spiked_mask]

        return spiked_mask
