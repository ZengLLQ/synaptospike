"""Unit & Invariant Tests for Hodgkin-Huxley Dynamics."""

import numpy as np
import pytest
from synaptospike.models.hodgkin_huxley import HodgkinHuxley, HHParameters, HHState


class TestHodgkinHuxley:
    """Test suite for Hodgkin-Huxley biophysical model."""

    def test_resting_equilibrium_stability(self) -> None:
        """Membrane potential without stimulus must stay strictly at resting potential."""
        hh = HodgkinHuxley()
        eq_state = hh.equilibrium_state()
        assert pytest.approx(eq_state.v, abs=0.1) == -65.0

        # Simulate 50 ms without stimulus
        sol = hh.simulate(t_span=(0.0, 50.0), dt=0.025, stimulus=0.0, initial_state=eq_state)
        assert len(sol.spike_times) == 0
        assert np.max(np.abs(sol.v - (-65.0))) < 0.5  # Sub-millivolt fluctuation

    def test_singularity_safe_rate_functions(self) -> None:
        """Singularity in alpha_m and alpha_n at exact critical voltages must not raise ZeroDivisionError."""
        hh = HodgkinHuxley()
        # V - V_rest = 25 mV -> V = -40 mV (critical for alpha_m)
        v_crit_m = -40.0
        val_m = hh.alpha_m(v_crit_m)
        assert not np.isnan(val_m)
        assert not np.isinf(val_m)
        assert val_m > 0.0

        # V - V_rest = 10 mV -> V = -55 mV (critical for alpha_n)
        v_crit_n = -55.0
        val_n = hh.alpha_n(v_crit_n)
        assert not np.isnan(val_n)
        assert not np.isinf(val_n)
        assert val_n > 0.0

    def test_suprathreshold_action_potentials(self) -> None:
        """Sustained 10 uA/cm^2 current injection must trigger regular repetitive firing (>50 Hz)."""
        hh = HodgkinHuxley()
        sol = hh.simulate(t_span=(0.0, 100.0), dt=0.025, stimulus=10.0, method="rk4")

        assert len(sol.spike_times) >= 5
        assert sol.firing_rate_hz > 50.0  # Typically ~65-70 Hz

        # Verify action potential amplitude: peak > +10 mV, hyperpolarization trough < -65 mV
        assert np.max(sol.v) > 10.0
        assert np.min(sol.v) < -65.0

    def test_subthreshold_depolarization(self) -> None:
        """Weak stimulus current (1.5 uA/cm^2) depolarizes without generating action potentials."""
        hh = HodgkinHuxley()
        sol = hh.simulate(t_span=(0.0, 50.0), dt=0.025, stimulus=1.5)

        assert len(sol.spike_times) == 0
        # Depolarizes above rest (-65 mV) but below spike threshold (0 mV)
        assert np.max(sol.v) > -65.0
        assert np.max(sol.v) < -50.0

    def test_refractory_period(self) -> None:
        """A second stimulus pulse during absolute refractory period fails to fire a second spike."""
        hh = HodgkinHuxley()

        # Stimulus with two short pulses (2 ms duration each) separated by 3 ms
        def stimulus_short_interval(t: float) -> float:
            if 10.0 <= t < 12.0:
                return 30.0  # First pulse
            elif 14.0 <= t < 16.0:
                return 30.0  # Second pulse during refractory phase
            return 0.0

        sol = hh.simulate(t_span=(0.0, 40.0), dt=0.025, stimulus=stimulus_short_interval)
        # Only 1 spike should be triggered because neuron is in absolute refractory state
        assert len(sol.spike_times) == 1

        # Stimulus with second pulse separated by 15 ms (after relative refractory period)
        def stimulus_long_interval(t: float) -> float:
            if 10.0 <= t < 12.0:
                return 30.0  # First pulse
            elif 27.0 <= t < 29.0:
                return 30.0  # Second pulse after recovery
            return 0.0

        sol_recovered = hh.simulate(t_span=(0.0, 40.0), dt=0.025, stimulus=stimulus_long_interval)
        assert len(sol_recovered.spike_times) == 2

    def test_rk4_vs_exponential_euler_convergence(self) -> None:
        """Exponential Euler and RK4 solutions must agree closely for small dt."""
        hh = HodgkinHuxley()
        sol_rk4 = hh.simulate(t_span=(0.0, 50.0), dt=0.01, stimulus=10.0, method="rk4")
        sol_ee = hh.simulate(t_span=(0.0, 50.0), dt=0.01, stimulus=10.0, method="exponential_euler")

        # Firing rates and spike counts should match
        assert len(sol_rk4.spike_times) == len(sol_ee.spike_times)
        # Spike times within 0.5 ms
        np.testing.assert_allclose(sol_rk4.spike_times, sol_ee.spike_times, atol=0.5)
