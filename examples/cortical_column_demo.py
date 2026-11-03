"""Cortical Microcircuit Simulation Demo.

Simulates a 1,000-neuron balanced recurrent network (800 excitatory, 200 inhibitory)
exhibiting asynchronous irregular firing and gamma-band Local Field Potential oscillations.
"""

import time
import numpy as np
from synaptospike.network.brunel import BrunelNetwork, BrunelParameters
from synaptospike.analysis.spectral import (
    compute_lfp,
    compute_power_spectral_density,
    compute_band_powers,
    compute_isi_statistics,
    compute_fano_factor,
)


def run_demo() -> None:
    print("=" * 70)
    print("SynaptoSpike: Cortical Microcircuit Simulation (Brunel 2000)")
    print("=" * 70)

    params = BrunelParameters(
        num_excitatory=800,
        num_inhibitory=200,
        epsilon=0.1,
        g_ratio=5.0,
        eta=2.0,
        delay_ms=1.5,
    )
    print(f"Network Scale:     {params.num_excitatory} Excitatory (80%) + {params.num_inhibitory} Inhibitory (20%)")
    print(f"Connectivity:      Sparse Random (p = {params.epsilon*100:.0f}%, C_E = {int(params.epsilon*params.num_excitatory)}, C_I = {int(params.epsilon*params.num_inhibitory)})")
    print(f"Inhibitory Ratio:  g = {params.g_ratio} (inhibition dominance)")
    print(f"External Drive:    eta = {params.eta} * nu_threshold")

    net = BrunelNetwork(params, seed=2026)
    duration_ms = 400.0
    dt = 0.1

    print(f"\nSimulating {duration_ms} ms of cortical activity (dt = {dt} ms)...")
    t0 = time.perf_counter()
    record, comp_current, time_grid = net.simulate(duration_ms=duration_ms, dt=dt)
    elapsed = time.perf_counter() - t0
    print(f"Simulation completed in {elapsed:.2f} seconds ({len(time_grid)} steps).")

    total_spikes = len(record.spike_times)
    duration_s = duration_ms / 1000.0
    exc_spikes = np.sum(record.is_excitatory)
    inh_spikes = total_spikes - exc_spikes

    rate_e = exc_spikes / (params.num_excitatory * duration_s)
    rate_i = inh_spikes / (params.num_inhibitory * duration_s)

    isi_stats = compute_isi_statistics(record.spike_times, record.neuron_ids)
    fano = compute_fano_factor(record.spike_times, duration_ms, bin_size_ms=50.0)

    # Local Field Potential spectral analysis
    fs_hz = 1000.0 / dt  # 10 kHz sampling rate
    lfp = compute_lfp(comp_current, sampling_rate_hz=fs_hz, cutoff_hz=250.0)
    psd_res = compute_power_spectral_density(lfp, sampling_rate_hz=fs_hz)
    bands = compute_band_powers(psd_res)

    print("\n" + "-" * 70)
    print("MACROSCOPIC FIRING METRICS")
    print("-" * 70)
    print(f"Total Action Potentials: {total_spikes:,}")
    print(f"Mean Excitatory Rate:    {rate_e:.2f} Hz")
    print(f"Mean Inhibitory Rate:    {rate_i:.2f} Hz")
    print(f"Inter-Spike Interval:    mu = {isi_stats.mean_isi_ms:.2f} ms, sigma = {isi_stats.std_isi_ms:.2f} ms")
    print(f"Coefficient of Var CV:   {isi_stats.cv_isi:.3f} -> {'Asynchronous Irregular (AI)' if isi_stats.cv_isi > 0.6 else 'Synchronous Regular (SR)'}")
    print(f"Spike Count Fano Factor: {fano:.3f}")

    print("\n" + "-" * 70)
    print("LOCAL FIELD POTENTIAL (LFP) SPECTRAL PROFILE")
    print("-" * 70)
    print(f"Dominant Oscillation:    {psd_res.peak_frequency_hz:.1f} Hz (Peak Power = {psd_res.peak_power:.2e})")
    print(f"Delta Power (1-4 Hz):    {bands.delta:.2e} ({(bands.delta/bands.total_power)*100:.1f}%)")
    print(f"Theta Power (4-8 Hz):    {bands.theta:.2e} ({(bands.theta/bands.total_power)*100:.1f}%)")
    print(f"Alpha Power (8-12 Hz):   {bands.alpha:.2e} ({(bands.alpha/bands.total_power)*100:.1f}%)")
    print(f"Beta Power (12-30 Hz):   {bands.beta:.2e} ({(bands.beta/bands.total_power)*100:.1f}%)")
    print(f"Gamma Power (30-80 Hz):  {bands.gamma:.2e} ({(bands.gamma/bands.total_power)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
