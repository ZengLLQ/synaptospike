"""Brunel (2000) Balanced Recurrent Cortical Microcircuit.

Simulates sparsely connected excitatory-inhibitory neural networks displaying
Asynchronous Irregular (AI), Synchronous Regular (SR), and Synchronous Irregular (SI)
collective firing states.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple, List, Dict, Optional
import numpy as np


@dataclasses.dataclass(frozen=True)
class BrunelParameters:
    """Network architecture and physiological parameters."""

    # Population size
    num_excitatory: int = 800  # N_E (80%)
    num_inhibitory: int = 200  # N_I (20%)

    # Sparse random connection probability
    epsilon: float = 0.1  # 10% connection probability

    # Membrane properties
    tau_m: float = 20.0  # Membrane time constant (ms)
    v_rest: float = 0.0  # Resting potential (mV)
    v_reset: float = 10.0  # Reset potential after spike (mV)
    v_thresh: float = 20.0  # Action potential threshold (mV)
    tau_ref: float = 2.0  # Absolute refractory period (ms)

    # Synaptic parameters
    j_excitatory: float = 0.1  # EPSP amplitude J (mV)
    g_ratio: float = 5.0  # Relative inhibitory strength g = -I/E (g > 4 for inhibition dominance)
    delay_ms: float = 1.5  # Synaptic transmission delay D (ms)

    # External background Poisson drive
    # eta = nu_ext / nu_thr: ratio to threshold rate
    eta: float = 2.0


@dataclasses.dataclass
class NetworkSpikeRecord:
    """Spike raster record across the simulation."""

    spike_times: np.ndarray  # Spike timestamps (ms)
    neuron_ids: np.ndarray  # Spiking neuron indices [0, N-1]
    is_excitatory: np.ndarray  # Boolean mask indicating excitatory spikes


@dataclasses.dataclass
class NetworkState:
    """Instantaneous dynamical state of all neurons in the microcircuit."""

    v: np.ndarray  # Membrane potentials (mV)
    refractory_time: np.ndarray  # Remaining refractory time (ms)


class BrunelNetwork:
    """Brunel recurrent cortical network engine."""

    def __init__(self, params: BrunelParameters | None = None, seed: int | None = 42) -> None:
        self.params = params or BrunelParameters()
        self.rng = np.random.default_rng(seed)

        self.n_total = self.params.num_excitatory + self.params.num_inhibitory
        self.c_e = int(self.params.epsilon * self.params.num_excitatory)
        self.c_i = int(self.params.epsilon * self.params.num_inhibitory)

        # Threshold Poisson rate nu_thr = V_thresh / (J * C_E * tau_m) in kHz (spikes/ms)
        self.nu_thr_khz = self.params.v_thresh / (self.params.j_excitatory * self.c_e * self.params.tau_m)
        self.nu_ext_khz = self.params.eta * self.nu_thr_khz

        # Synaptic weights: Excitatory J, Inhibitory -g * J
        self.w_e = self.params.j_excitatory
        self.w_i = -self.params.g_ratio * self.params.j_excitatory

        # Precompute sparse connectivity tables as 2D NumPy arrays: shape (n_total, c_e) and (n_total, c_i)
        self.e_connections = [
            self.rng.choice(self.params.num_excitatory, size=self.c_e, replace=False)
            for _ in range(self.n_total)
        ]
        self.i_connections = [
            self.params.num_excitatory + self.rng.choice(self.params.num_inhibitory, size=self.c_i, replace=False)
            for _ in range(self.n_total)
        ]
        self.e_connections_mat = np.array(self.e_connections, dtype=np.int64)
        self.i_connections_mat = np.array(self.i_connections, dtype=np.int64)

    def simulate(
        self,
        duration_ms: float,
        dt: float = 0.1,
    ) -> Tuple[NetworkSpikeRecord, np.ndarray, np.ndarray]:
        """Run network simulation for duration_ms with time step dt (ms).

        Returns:
            NetworkSpikeRecord: All recorded spikes.
            compound_current: Total instantaneous synaptic current proxy across time.
            time_grid: Simulation time array (ms).
        """
        time_grid = np.arange(0.0, duration_ms, dt)
        num_steps = len(time_grid)

        # State initialization
        v = self.rng.uniform(self.params.v_reset, self.params.v_thresh, size=self.n_total)
        refr_timer = np.zeros(self.n_total, dtype=np.float64)

        # Delay buffer: circular queue of spike boolean arrays
        delay_steps = max(1, int(round(self.params.delay_ms / dt)))
        delay_buffer: List[np.ndarray] = [np.zeros(self.n_total, dtype=bool) for _ in range(delay_steps)]

        # Spike recording containers
        all_spike_times: List[float] = []
        all_spike_ids: List[int] = []
        all_is_exc: List[bool] = []

        # Compound synaptic current for Local Field Potential (LFP) proxy
        compound_current = np.zeros(num_steps, dtype=np.float64)

        # Precompute mean external Poisson spikes per neuron per dt step
        lambda_ext = self.c_e * self.nu_ext_khz * dt

        exp_decay = np.exp(-dt / self.params.tau_m)

        for step_idx, t in enumerate(time_grid):
            # 1. Decay membrane potential towards rest for non-refractory neurons
            active_mask = refr_timer <= 0.0
            v[active_mask] = self.params.v_rest + (v[active_mask] - self.params.v_rest) * exp_decay

            # 2. External Poisson background drive
            ext_spikes = self.rng.poisson(lambda_ext, size=self.n_total)
            v[active_mask] += ext_spikes[active_mask] * self.w_e

            # 3. Delayed recurrent synaptic inputs (vectorized)
            arriving_spikes = delay_buffer[step_idx % delay_steps]
            if np.any(arriving_spikes):
                n_e_in = np.sum(arriving_spikes[self.e_connections_mat], axis=1)
                n_i_in = np.sum(arriving_spikes[self.i_connections_mat], axis=1)
                delta_v = n_e_in * self.w_e + n_i_in * self.w_i
                v[active_mask] += delta_v[active_mask]

            # 4. Action potential threshold detection
            spiked_mask = (v >= self.params.v_thresh) & active_mask
            num_spikes = int(np.sum(spiked_mask))

            if num_spikes > 0:
                spiking_indices = np.where(spiked_mask)[0]
                # Reset voltage and activate refractory timer
                v[spiking_indices] = self.params.v_reset
                refr_timer[spiking_indices] = self.params.tau_ref

                # Log spikes
                for s_id in spiking_indices:
                    all_spike_times.append(t)
                    all_spike_ids.append(int(s_id))
                    all_is_exc.append(bool(s_id < self.params.num_excitatory))

            # Store current spikes into delay buffer for future delivery
            delay_buffer[step_idx % delay_steps] = spiked_mask

            # Update refractory timers
            refr_timer[~active_mask] = np.maximum(0.0, refr_timer[~active_mask] - dt)

            # Record proxy synaptic current
            compound_current[step_idx] = float(np.sum(np.abs(v - self.params.v_rest)))

        record = NetworkSpikeRecord(
            spike_times=np.array(all_spike_times, dtype=np.float64),
            neuron_ids=np.array(all_spike_ids, dtype=np.int64),
            is_excitatory=np.array(all_is_exc, dtype=bool),
        )
        return record, compound_current, time_grid
