"""Unit & Invariant Tests for Spike-Timing-Dependent Plasticity (STDP)."""

import numpy as np
import pytest
from synaptospike.plasticity.stdp import (
    STDPModel,
    STDPParameters,
    STDPPairingMode,
)


class TestSTDP:
    """Test suite for STDP Hebbian learning rules."""

    def test_causal_pairing_ltp(self) -> None:
        """Causal pairing (pre before post, delta_t > 0) must induce Long-Term Potentiation (LTP)."""
        model = STDPModel()
        delta_t = 10.0  # ms
        dw = model.compute_weight_change(current_w=0.5, delta_t=delta_t)
        assert dw > 0.0

        # Simulate 20 causal pairings
        t_pre = [10.0 + i * 100.0 for i in range(20)]
        t_post = [t + 10.0 for t in t_pre]

        w_final, _ = model.apply_pairings(0.5, t_pre, t_post)
        assert w_final > 0.5

    def test_anti_causal_pairing_ltd(self) -> None:
        """Anti-causal pairing (post before pre, delta_t < 0) must induce Long-Term Depression (LTD)."""
        model = STDPModel()
        delta_t = -10.0  # ms
        dw = model.compute_weight_change(current_w=0.5, delta_t=delta_t)
        assert dw < 0.0

        # Simulate 20 anti-causal pairings
        t_pre = [20.0 + i * 100.0 for i in range(20)]
        t_post = [t - 10.0 for t in t_pre]

        w_final, _ = model.apply_pairings(0.5, t_pre, t_post)
        assert w_final < 0.5

    def test_stdp_exponential_decay(self) -> None:
        """Weight modification magnitude must decay exponentially with increasing pairing separation."""
        params = STDPParameters(tau_plus=20.0, tau_minus=20.0)
        model = STDPModel(params)

        dw_5ms = model.compute_weight_change(0.5, delta_t=5.0)
        dw_25ms = model.compute_weight_change(0.5, delta_t=25.0)
        dw_60ms = model.compute_weight_change(0.5, delta_t=60.0)

        assert dw_5ms > dw_25ms > dw_60ms > 0.0
        # Check ratio: dw(25)/dw(5) should be exp(-(25-5)/20) = exp(-1) = 0.3679
        ratio = dw_25ms / dw_5ms
        assert pytest.approx(ratio, rel=1e-3) == np.exp(-1.0)

    def test_weight_saturation_bounds(self) -> None:
        """Synaptic weight must never exceed w_max or fall below w_min."""
        params = STDPParameters(w_min=0.0, w_max=1.0)
        model = STDPModel(params)

        # Apply 200 causal pairings that would theoretically push w >> 1.0
        t_pre = [10.0 + i * 50.0 for i in range(200)]
        t_post = [t + 5.0 for t in t_pre]
        w_pot, _ = model.apply_pairings(0.8, t_pre, t_post)
        assert w_pot <= 1.0

        # Apply 200 anti-causal pairings
        t_pre_ltd = [10.0 + i * 50.0 for i in range(200)]
        t_post_ltd = [t - 5.0 for t in t_pre_ltd]
        w_dep, _ = model.apply_pairings(0.2, t_pre_ltd, t_post_ltd)
        assert w_dep >= 0.0

    def test_nearest_neighbor_pairing(self) -> None:
        """Nearest-neighbor pairing mode must run and maintain monotonic bounds."""
        params = STDPParameters(pairing_mode=STDPPairingMode.NEAREST_NEIGHBOR)
        model = STDPModel(params)

        t_pre = [10.0, 50.0, 100.0]
        t_post = [20.0, 60.0, 110.0]

        w_final, history = model.apply_pairings(0.5, t_pre, t_post)
        assert len(history) > 1
        assert w_final > 0.5
