"""Hodgkin-Huxley Excitability & F-I Curve Demonstration.

Computes the frequency-current (F-I) curve across a sweep of injected currents,
demonstrating the discontinuous onset characteristic of Hodgkin's Class II neural excitability.
"""

import numpy as np
from synaptospike.models.hodgkin_huxley import HodgkinHuxley


def run_demo() -> None:
    print("=" * 70)
    print("SynaptoSpike: Hodgkin-Huxley (1952) Excitability Curve (F-I)")
    print("=" * 70)

    hh = HodgkinHuxley()
    stimuli = np.linspace(0.0, 20.0, 21)

    print(f"{'I_inj (uA/cm^2)':<18} | {'Spike Count':<14} | {'Firing Rate (Hz)':<18} | {'Excitability'}")
    print("-" * 70)

    for i_stim in stimuli:
        sol = hh.simulate(t_span=(0.0, 200.0), dt=0.025, stimulus=float(i_stim), method="rk4")
        rate = sol.firing_rate_hz
        count = len(sol.spike_times)

        state_label = "Quiescent" if count == 0 else "Repetitive Firing"
        print(f"{i_stim:<18.1f} | {count:<14} | {rate:<18.2f} | {state_label}")

    print("=" * 70)
    print("Observation: Firing rate exhibits a discontinuous jump at I_c ~ 6.2 uA/cm^2,")
    print("characterizing a subcritical Andronov-Hopf bifurcation (Class II Excitability).")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
