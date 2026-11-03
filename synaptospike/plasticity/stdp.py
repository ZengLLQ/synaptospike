"""Spike-Timing-Dependent Plasticity (STDP) Learning Engine.

Implements asymmetric Hebbian synaptic weight modifications (Bi & Poo 1998,
Song et al. 2000) with additive and multiplicative bounds, nearest-neighbor
and all-to-all spike pairing kinetics.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Tuple, List, Optional
import numpy as np


class STDPPairingMode(Enum):
    """Pairing mechanism for multi-spike trains."""

    ALL_TO_ALL = "all_to_all"
    NEAREST_NEIGHBOR = "nearest_neighbor"


@dataclasses.dataclass(frozen=True)
class STDPParameters:
    """Kinetic parameters for Hebbian STDP kernel."""

    # Maximum potentiation and depression amplitudes
    a_plus: float = 0.01
    a_minus: float = 0.0105  # Asymmetric: depression slightly dominates (Song et al. 2000)

    # Potentiation and depression time constants (ms)
    tau_plus: float = 20.0
    tau_minus: float = 20.0

    # Weight saturation boundaries
    w_min: float = 0.0
    w_max: float = 1.0

    # Boundedness exponent: mu=0.0 (additive), mu=1.0 (multiplicative soft-bound)
    mu: float = 0.0

    # Multi-spike interaction rule
    pairing_mode: STDPPairingMode = STDPPairingMode.ALL_TO_ALL


class STDPModel:
    """Synaptic plasticity manager updating weights from pre- and post-synaptic events."""

    def __init__(self, params: STDPParameters | None = None) -> None:
        self.params = params or STDPParameters()

    def kernel(self, delta_t: float) -> float:
        """Evaluate continuous STDP kernel for a single pair delta_t = t_post - t_pre (ms)."""
        if delta_t > 0:
            # Pre before post -> LTP
            return float(self.params.a_plus * np.exp(-delta_t / self.params.tau_plus))
        elif delta_t < 0:
            # Post before pre -> LTD
            return float(-self.params.a_minus * np.exp(delta_t / self.params.tau_minus))
        else:
            return 0.0

    def compute_weight_change(self, current_w: float, delta_t: float) -> float:
        """Compute delta_w considering additive or multiplicative boundaries."""
        raw_dw = self.kernel(delta_t)
        if raw_dw > 0:
            # Potentiation
            scale = (self.params.w_max - current_w) ** self.params.mu if self.params.mu > 0 else 1.0
            return float(raw_dw * scale)
        elif raw_dw < 0:
            # Depression
            scale = (current_w - self.params.w_min) ** self.params.mu if self.params.mu > 0 else 1.0
            return float(raw_dw * scale)
        return 0.0

    def apply_pairings(
        self,
        initial_w: float,
        pre_spikes: np.ndarray | List[float],
        post_spikes: np.ndarray | List[float],
    ) -> Tuple[float, List[float]]:
        """Simulate an entire spike train protocol, returning (final_w, weight_history)."""
        pre_arr = np.sort(np.asarray(pre_spikes, dtype=np.float64))
        post_arr = np.sort(np.asarray(post_spikes, dtype=np.float64))

        w = float(initial_w)
        w_history: List[float] = [w]

        if self.params.pairing_mode == STDPPairingMode.NEAREST_NEIGHBOR:
            # Nearest-neighbor matching
            # For each post-spike, find immediate preceding pre-spike
            for t_post in post_arr:
                pre_prior = pre_arr[pre_arr < t_post]
                if len(pre_prior) > 0:
                    dt = t_post - pre_prior[-1]
                    dw = self.compute_weight_change(w, dt)
                    w = np.clip(w + dw, self.params.w_min, self.params.w_max)
                    w_history.append(float(w))

            # For each pre-spike, find immediate preceding post-spike
            for t_pre in pre_arr:
                post_prior = post_arr[post_arr < t_pre]
                if len(post_prior) > 0:
                    dt = post_prior[-1] - t_pre  # negative delta_t
                    dw = self.compute_weight_change(w, dt)
                    w = np.clip(w + dw, self.params.w_min, self.params.w_max)
                    w_history.append(float(w))

        else:
            # Online trace-based all-to-all pairing
            # Merge all events into chronological queue: (time, is_pre)
            events: List[Tuple[float, bool]] = [(t, True) for t in pre_arr] + [(t, False) for t in post_arr]
            events.sort(key=lambda x: x[0])

            x_pre = 0.0
            y_post = 0.0
            last_t = 0.0

            for t, is_pre in events:
                dt_step = t - last_t
                if dt_step > 0:
                    x_pre *= np.exp(-dt_step / self.params.tau_plus)
                    y_post *= np.exp(-dt_step / self.params.tau_minus)

                if is_pre:
                    # Presynaptic spike arrived: depress based on post trace y_post
                    if y_post > 0:
                        scale = (w - self.params.w_min) ** self.params.mu if self.params.mu > 0 else 1.0
                        dw = -self.params.a_minus * y_post * scale
                        w = np.clip(w + dw, self.params.w_min, self.params.w_max)
                        w_history.append(float(w))
                    x_pre += 1.0
                else:
                    # Postsynaptic spike arrived: potentiate based on pre trace x_pre
                    if x_pre > 0:
                        scale = (self.params.w_max - w) ** self.params.mu if self.params.mu > 0 else 1.0
                        dw = self.params.a_plus * x_pre * scale
                        w = np.clip(w + dw, self.params.w_min, self.params.w_max)
                        w_history.append(float(w))
                    y_post += 1.0

                last_t = t

        return float(w), w_history
