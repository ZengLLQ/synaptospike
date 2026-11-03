"""Unit & Invariant Tests for Izhikevich Phenomenological Spiking Models."""

import numpy as np
import pytest
from synaptospike.models.izhikevich import (
    IzhikevichNeuron,
    IzhikevichPopulation,
    IzhikevichParameters,
    IzhikevichPreset,
)


class TestIzhikevich:
    """Test suite for Izhikevich spiking dynamics."""

    def test_all_presets_instantiation(self) -> None:
        """Every canonical preset must instantiate valid parameters."""
        for preset in IzhikevichPreset:
            params = IzhikevichParameters.from_preset(preset)
            assert params.a > 0.0
            assert params.b > 0.0
            assert params.c < 0.0
            assert params.d >= 0.0
            neuron = IzhikevichNeuron(params)
            assert neuron.state.v == params.c

    def test_regular_spiking_adaptation(self) -> None:
        """Regular Spiking (RS) neurons must exhibit spike frequency adaptation (longer ISIs over time)."""
        params = IzhikevichParameters.from_preset(IzhikevichPreset.REGULAR_SPIKING)
        neuron = IzhikevichNeuron(params)
        _, _, _, spikes = neuron.simulate(t_span=(0.0, 300.0), dt=0.5, i_inj=10.0)

        assert len(spikes) >= 4
        isis = np.diff(spikes)
        # First ISI should be shorter than later ISIs due to recovery variable accumulation
        assert isis[0] < isis[-1]

    def test_fast_spiking_no_adaptation(self) -> None:
        """Fast Spiking (FS) interneurons must fire high-frequency spikes with minimal adaptation."""
        params = IzhikevichParameters.from_preset(IzhikevichPreset.FAST_SPIKING)
        neuron = IzhikevichNeuron(params)
        _, _, _, spikes = neuron.simulate(t_span=(0.0, 200.0), dt=0.5, i_inj=10.0)

        assert len(spikes) >= 15
        isis = np.diff(spikes)
        # Fast spiking interneurons fire rapidly (>80 Hz, mean ISI < 15 ms) without strong adaptation
        assert np.mean(isis) < 15.0
        # Unlike RS where first ISI is ~40% shorter than steady-state, FS has consistent steady ISIs
        assert isis[-1] < 20.0

    def test_intrinsically_bursting(self) -> None:
        """Intrinsically Bursting (IB) neurons must fire bursts of rapid spikes followed by a quiet inter-burst period."""
        params = IzhikevichParameters.from_preset(IzhikevichPreset.INTRINSICALLY_BURSTING)
        neuron = IzhikevichNeuron(params)
        _, _, _, spikes = neuron.simulate(t_span=(0.0, 300.0), dt=0.5, i_inj=10.0)

        assert len(spikes) >= 5
        isis = np.diff(spikes)
        # Bimodal ISI distribution: intra-burst ISIs (< 10 ms) and inter-burst ISIs (> 30 ms)
        assert np.any(isis < 10.0)
        assert np.any(isis > 30.0)

    def test_vectorized_population_consistency(self) -> None:
        """Vectorized population step must match individual single neuron steps identically."""
        num_neurons = 5
        params = IzhikevichParameters.from_preset(IzhikevichPreset.REGULAR_SPIKING)

        pop = IzhikevichPopulation(
            num_neurons=num_neurons,
            a=params.a,
            b=params.b,
            c=params.c,
            d=params.d,
        )

        single_neurons = [IzhikevichNeuron(params) for _ in range(num_neurons)]

        dt = 0.5
        i_inj_val = 10.0

        for step in range(100):
            pop_spikes = pop.step(i_inj_val, dt)
            for idx, n in enumerate(single_neurons):
                s = n.step(i_inj_val, dt)
                assert s == pop_spikes[idx]
                assert pytest.approx(n.state.v, rel=1e-5) == pop.v[idx]
                assert pytest.approx(n.state.u, rel=1e-5) == pop.u[idx]
