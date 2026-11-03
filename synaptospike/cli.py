"""Command-Line Interface for SynaptoSpike Neural Dynamics Engine."""

from __future__ import annotations

import argparse
import sys
import numpy as np

from synaptospike.models.hodgkin_huxley import HodgkinHuxley, HHParameters
from synaptospike.models.morris_lecar import MorrisLecar, MLParameters
from synaptospike.models.izhikevich import IzhikevichNeuron, IzhikevichParameters, IzhikevichPreset
from synaptospike.plasticity.stdp import STDPModel, STDPParameters, STDPPairingMode
from synaptospike.network.brunel import BrunelNetwork, BrunelParameters
from synaptospike.analysis.spectral import (
    compute_lfp,
    compute_power_spectral_density,
    compute_band_powers,
    compute_isi_statistics,
    compute_fano_factor,
)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run Hodgkin-Huxley biophysical simulation."""
    print("=" * 60)
    print("SynaptoSpike: Hodgkin-Huxley (1952) Biophysical Simulation")
    print("=" * 60)
    print(f"Duration:       {args.duration} ms (dt = {args.dt} ms)")
    print(f"Stimulus:       {args.stimulus} uA/cm^2")
    print(f"Integrator:     {args.method.upper()}")

    hh = HodgkinHuxley()
    sol = hh.simulate(
        t_span=(0.0, args.duration),
        dt=args.dt,
        stimulus=args.stimulus,
        method=args.method,
    )

    print("-" * 60)
    print(f"Total Spikes:   {len(sol.spike_times)}")
    print(f"Firing Rate:    {sol.firing_rate_hz:.2f} Hz")
    if len(sol.inter_spike_intervals) > 0:
        print(f"Mean ISI:       {np.mean(sol.inter_spike_intervals):.2f} ms (std = {np.std(sol.inter_spike_intervals):.2f} ms)")
    if len(sol.spike_times) > 0:
        first_few = [f"{t:.1f}" for t in sol.spike_times[:5]]
        print(f"First Spikes:   [{', '.join(first_few)}{'...' if len(sol.spike_times) > 5 else ''}] ms")
    print("=" * 60)


def cmd_phase_plane(args: argparse.Namespace) -> None:
    """Perform Morris-Lecar phase-plane analysis."""
    print("=" * 60)
    print("SynaptoSpike: Morris-Lecar (1981) Phase-Plane Analysis")
    print("=" * 60)
    print(f"Regime:         {args.regime.upper()}")
    print(f"Current I_inj:  {args.stimulus} uA/cm^2")

    params = MLParameters.hopf_regime() if args.regime == "hopf" else MLParameters.snic_regime()
    ml = MorrisLecar(params)
    analysis = ml.analyze_phase_plane(i_inj=args.stimulus, v_range=(-80.0, 50.0))

    print("-" * 60)
    print(f"Fixed Points Found: {len(analysis.fixed_points)}")
    for idx, (v_star, w_star) in enumerate(analysis.fixed_points):
        eigs = analysis.eigenvalues[idx]
        stab = analysis.stability[idx]
        eig_str = f"({eigs[0].real:.3f} + {eigs[0].imag:.3f}j, {eigs[1].real:.3f} + {eigs[1].imag:.3f}j)"
        print(f" [{idx + 1}] V* = {v_star:7.2f} mV | w* = {w_star:6.4f} | Type = {stab:<15} | Eigs = {eig_str}")
    print("=" * 60)


def cmd_izhikevich(args: argparse.Namespace) -> None:
    """Run Izhikevich single neuron simulation."""
    print("=" * 60)
    print("SynaptoSpike: Izhikevich (2003) Phenomenological Simulation")
    print("=" * 60)
    preset_enum = IzhikevichPreset(args.preset)
    params = IzhikevichParameters.from_preset(preset_enum)
    print(f"Preset:         {args.preset}")
    print(f"Parameters:     a={params.a}, b={params.b}, c={params.c}, d={params.d}")
    print(f"Duration:       {args.duration} ms | I_inj = {args.stimulus} pA")

    neuron = IzhikevichNeuron(params)
    t, v, u, spikes = neuron.simulate(t_span=(0.0, args.duration), dt=args.dt, i_inj=args.stimulus)

    print("-" * 60)
    print(f"Spikes Emitter: {len(spikes)}")
    if len(spikes) >= 2:
        isis = np.diff(spikes)
        rate = len(spikes) / (args.duration / 1000.0)
        print(f"Firing Rate:    {rate:.2f} Hz")
        print(f"Mean ISI:       {np.mean(isis):.2f} ms (CV = {np.std(isis) / np.mean(isis):.3f})")
    print("=" * 60)


def cmd_network(args: argparse.Namespace) -> None:
    """Simulate Brunel recurrent cortical network."""
    print("=" * 60)
    print("SynaptoSpike: Brunel (2000) Recurrent Cortical Microcircuit")
    print("=" * 60)
    print(f"Neurons:        N_E = {args.excitatory}, N_I = {args.inhibitory} (Total {args.excitatory + args.inhibitory})")
    print(f"Inhibitory g:   {args.g} | External drive eta: {args.eta}")
    print(f"Duration:       {args.duration} ms (dt = {args.dt} ms)")

    params = BrunelParameters(
        num_excitatory=args.excitatory,
        num_inhibitory=args.inhibitory,
        g_ratio=args.g,
        eta=args.eta,
    )
    net = BrunelNetwork(params)
    record, comp_current, time_grid = net.simulate(duration_ms=args.duration, dt=args.dt)

    total_spikes = len(record.spike_times)
    duration_s = args.duration / 1000.0

    exc_spikes = np.sum(record.is_excitatory)
    inh_spikes = total_spikes - exc_spikes

    rate_e = exc_spikes / (args.excitatory * duration_s) if duration_s > 0 else 0.0
    rate_i = inh_spikes / (args.inhibitory * duration_s) if duration_s > 0 else 0.0

    isi_stats = compute_isi_statistics(record.spike_times, record.neuron_ids)
    fano = compute_fano_factor(record.spike_times, args.duration)

    print("-" * 60)
    print(f"Total Spikes:   {total_spikes}")
    print(f"Exc Mean Rate:  {rate_e:.2f} Hz")
    print(f"Inh Mean Rate:  {rate_i:.2f} Hz")
    print(f"Mean ISI:       {isi_stats.mean_isi_ms:.2f} ms")
    print(f"CV_ISI:         {isi_stats.cv_isi:.3f} ({'Irregular (AI)' if isi_stats.cv_isi > 0.8 else 'Regular (SR)'})")
    print(f"Fano Factor:    {fano:.3f}")
    print("=" * 60)


def cmd_stdp(args: argparse.Namespace) -> None:
    """Simulate Spike-Timing-Dependent Plasticity pairing protocol."""
    print("=" * 60)
    print("SynaptoSpike: Spike-Timing-Dependent Plasticity (STDP) Protocol")
    print("=" * 60)
    print(f"Pairings:       {args.pairings} at delta_t = {args.delta_t} ms")
    print(f"Initial Weight: {args.initial_weight}")
    print(f"Pairing Mode:   {args.mode.upper()}")

    mode_enum = STDPPairingMode.NEAREST_NEIGHBOR if args.mode == "nearest" else STDPPairingMode.ALL_TO_ALL
    params = STDPParameters(pairing_mode=mode_enum)
    model = STDPModel(params)

    # Generate synthetic pairing train at 10 Hz (interval = 100 ms)
    inter_pair_interval = 100.0
    t_pre_list = [10.0 + i * inter_pair_interval for i in range(args.pairings)]
    t_post_list = [t + args.delta_t for t in t_pre_list]

    final_w, history = model.apply_pairings(args.initial_weight, t_pre_list, t_post_list)
    delta_w = final_w - args.initial_weight
    pct_change = (delta_w / args.initial_weight) * 100.0 if args.initial_weight > 0 else 0.0

    print("-" * 60)
    print(f"Final Weight:   {final_w:.4f}")
    print(f"Delta W:        {delta_w:+.4f} ({pct_change:+.1f}%)")
    print(f"Plasticity:     {'LTP (Potentiation)' if delta_w > 0 else 'LTD (Depression)' if delta_w < 0 else 'No Change'}")
    print("=" * 60)


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="synaptospike",
        description="SynaptoSpike: High-Performance Biophysical Neural Dynamics & Network Simulation Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: simulate (Hodgkin-Huxley)
    p_sim = subparsers.add_parser("simulate", help="Run Hodgkin-Huxley biophysical simulation")
    p_sim.add_argument("--duration", type=float, default=100.0, help="Simulation duration in ms")
    p_sim.add_argument("--dt", type=float, default=0.025, help="Integration time step in ms")
    p_sim.add_argument("--stimulus", type=float, default=10.0, help="Injected current in uA/cm^2")
    p_sim.add_argument("--method", choices=["rk4", "exponential_euler"], default="rk4", help="Integration method")
    p_sim.set_defaults(func=cmd_simulate)

    # Subcommand: phase-plane (Morris-Lecar)
    p_pp = subparsers.add_parser("phase-plane", help="Morris-Lecar phase-plane analysis")
    p_pp.add_argument("--regime", choices=["hopf", "snic"], default="hopf", help="Bifurcation regime")
    p_pp.add_argument("--stimulus", type=float, default=80.0, help="Current stimulus I_inj in uA/cm^2")
    p_pp.set_defaults(func=cmd_phase_plane)

    # Subcommand: izhikevich
    p_izh = subparsers.add_parser("izhikevich", help="Izhikevich phenomenological spiking simulation")
    p_izh.add_argument(
        "--preset",
        choices=[e.value for e in IzhikevichPreset],
        default="regular_spiking",
        help="Canonical firing type",
    )
    p_izh.add_argument("--duration", type=float, default=200.0, help="Duration in ms")
    p_izh.add_argument("--dt", type=float, default=0.5, help="Time step in ms")
    p_izh.add_argument("--stimulus", type=float, default=10.0, help="Current stimulus in pA")
    p_izh.set_defaults(func=cmd_izhikevich)

    # Subcommand: network (Brunel)
    p_net = subparsers.add_parser("network", help="Simulate Brunel balanced recurrent network")
    p_net.add_argument("--excitatory", type=int, default=800, help="Number of excitatory neurons")
    p_net.add_argument("--inhibitory", type=int, default=200, help="Number of inhibitory neurons")
    p_net.add_argument("--g", type=float, default=5.0, help="Inhibitory to excitatory weight ratio")
    p_net.add_argument("--eta", type=float, default=2.0, help="External rate to threshold ratio")
    p_net.add_argument("--duration", type=float, default=300.0, help="Duration in ms")
    p_net.add_argument("--dt", type=float, default=0.1, help="Time step in ms")
    p_net.set_defaults(func=cmd_network)

    # Subcommand: stdp
    p_stdp = subparsers.add_parser("stdp", help="Simulate STDP pairing protocol")
    p_stdp.add_argument("--pairings", type=int, default=60, help="Number of spike pairings")
    p_stdp.add_argument("--delta-t", type=float, default=10.0, dest="delta_t", help="Pairing interval t_post - t_pre (ms)")
    p_stdp.add_argument("--initial-weight", type=float, default=0.5, help="Initial synaptic weight [0, 1]")
    p_stdp.add_argument("--mode", choices=["all_to_all", "nearest"], default="all_to_all", help="Spike pairing mode")
    p_stdp.set_defaults(func=cmd_stdp)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
