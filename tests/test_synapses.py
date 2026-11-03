"""Unit & Invariant Tests for Conductance-Based Synaptic Receptors."""

import numpy as np
import pytest
from synaptospike.synapses.conductance import (
    ConductanceSynapse,
    SynapseType,
    SynapticParameters,
)


class TestConductanceSynapses:
    """Test suite for AMPA, NMDA, and GABA conductance synapses."""

    def test_ampa_peak_conductance(self) -> None:
        """AMPA receptor must reach its designated peak conductance g_max upon a single spike."""
        g_target = 2.5
        params = SynapticParameters.ampa_default(g_max=g_target)
        syn = ConductanceSynapse(params)

        # Trigger presynaptic spike
        syn.on_presynaptic_spike(weight=1.0)

        dt = 0.05
        conductance_trace = []
        for _ in range(200):  # 10 ms
            syn.step(dt)
            conductance_trace.append(syn.params.g_max * syn.s)

        peak_g = np.max(conductance_trace)
        assert pytest.approx(peak_g, rel=0.02) == g_target

    def test_ampa_fast_decay(self) -> None:
        """AMPA conductance must decay back to near zero within ~10-15 ms."""
        params = SynapticParameters.ampa_default(g_max=1.0)
        syn = ConductanceSynapse(params)
        syn.on_presynaptic_spike()

        dt = 0.1
        for _ in range(150):  # 15 ms
            syn.step(dt)

        assert syn.s < 0.05

    def test_nmda_voltage_dependent_mg_block(self) -> None:
        """NMDA magnesium block must be highly occluded at -70 mV and largely unblocked at +20 mV."""
        params = SynapticParameters.nmda_default()
        syn = ConductanceSynapse(params)

        b_hyperpolarized = syn.mg_block(v_post=-70.0)
        b_depolarized = syn.mg_block(v_post=20.0)

        # At -70 mV, B(V) should be small (< 0.15)
        assert b_hyperpolarized < 0.15
        # At +20 mV, B(V) should be open (> 0.70)
        assert b_depolarized > 0.70
        assert b_depolarized > 4.0 * b_hyperpolarized

    def test_gaba_a_hyperpolarizing_current(self) -> None:
        """GABA_A receptor (E_rev = -70 mV) produces hyperpolarizing outward current when V > -70 mV."""
        params = SynapticParameters.gaba_a_default(g_max=1.0)
        syn = ConductanceSynapse(params)
        syn.on_presynaptic_spike()

        # Step to peak
        for _ in range(20):
            syn.step(0.05)

        # For depolarized post potential (-50 mV > -70 mV), current is positive (outward/hyperpolarizing)
        i_hyper = syn.current(v_post=-50.0)
        assert i_hyper > 0.0

        # For potential below reversal (-80 mV < -70 mV), current is negative (inward/depolarizing)
        i_depol = syn.current(v_post=-80.0)
        assert i_depol < 0.0

    def test_synaptic_summation(self) -> None:
        """Multiple rapid presynaptic spikes must summate conductances higher than a single spike."""
        params = SynapticParameters.ampa_default(g_max=1.0)
        syn = ConductanceSynapse(params)

        # Single spike peak
        syn.on_presynaptic_spike()
        peaks = []
        for _ in range(30):
            syn.step(0.05)
            peaks.append(syn.s)
        single_peak = max(peaks)

        # Summation: deliver 3 spikes at 2 ms intervals
        syn_sum = ConductanceSynapse(params)
        syn_sum.on_presynaptic_spike()
        for _ in range(40):  # 2 ms
            syn_sum.step(0.05)
        syn_sum.on_presynaptic_spike()
        for _ in range(40):  # 2 ms
            syn_sum.step(0.05)
        syn_sum.on_presynaptic_spike()
        summed_peaks = []
        for _ in range(60):
            syn_sum.step(0.05)
            summed_peaks.append(syn_sum.s)

        assert max(summed_peaks) > single_peak * 1.5
