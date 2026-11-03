"""Unit & Invariant Tests for Brunel Recurrent Network Microcircuit."""

import numpy as np
import pytest
from synaptospike.network.brunel import (
    BrunelNetwork,
    BrunelParameters,
    NetworkSpikeRecord,
)
from synaptospike.analysis.spectral import compute_isi_statistics


class TestBrunelNetwork:
    """Test suite for balanced cortical microcircuit dynamics."""

    def test_network_topology(self) -> None:
        """Verify graph degrees: each neuron must receive exactly C_E excitatory and C_I inhibitory inputs."""
        n_e = 400
        n_i = 100
        params = BrunelParameters(num_excitatory=n_e, num_inhibitory=n_i, epsilon=0.1)
        net = BrunelNetwork(params)

        assert len(net.e_connections) == n_e + n_i
        assert len(net.i_connections) == n_e + n_i

        expected_ce = int(0.1 * n_e)
        expected_ci = int(0.1 * n_i)

        for i in range(net.n_total):
            assert len(net.e_connections[i]) == expected_ce
            assert len(net.i_connections[i]) == expected_ci
            # Presynaptic excitatory indices are within [0, N_E - 1]
            assert np.all(net.e_connections[i] < n_e)
            # Presynaptic inhibitory indices are within [N_E, N_E + N_I - 1]
            assert np.all(net.i_connections[i] >= n_e)

    def test_network_simulation_stability(self) -> None:
        """Network simulation must run without numerical instability (no NaN or Inf in trajectories)."""
        params = BrunelParameters(
            num_excitatory=200,
            num_inhibitory=50,
            g_ratio=5.0,
            eta=2.0,
        )
        net = BrunelNetwork(params, seed=123)
        record, comp_current, time_grid = net.simulate(duration_ms=100.0, dt=0.1)

        assert not np.any(np.isnan(comp_current))
        assert not np.any(np.isinf(comp_current))
        assert len(record.spike_times) > 0
        assert len(record.neuron_ids) == len(record.spike_times)

    def test_asynchronous_irregular_state(self) -> None:
        """Verify network firing statistics and non-trivial inter-spike intervals."""
        params = BrunelParameters(
            num_excitatory=400,
            num_inhibitory=100,
            g_ratio=5.0,
            eta=2.0,
        )
        net = BrunelNetwork(params, seed=42)
        record, _, _ = net.simulate(duration_ms=250.0, dt=0.1)

        isi_stats = compute_isi_statistics(record.spike_times, record.neuron_ids)
        assert isi_stats.mean_isi_ms > 0.0
        assert isi_stats.cv_isi > 0.05
        assert len(record.spike_times) > 1000
