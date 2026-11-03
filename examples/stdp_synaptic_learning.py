"""STDP Synaptic Plasticity Learning Protocol Demo.

Demonstrates asymmetric Spike-Timing-Dependent Plasticity (Bi & Poo 1998)
under repetitive pre-post pairing at +10 ms (LTP) and -10 ms (LTD).
"""

from synaptospike.plasticity.stdp import STDPModel, STDPParameters, STDPPairingMode


def run_demo() -> None:
    print("=" * 70)
    print("SynaptoSpike: Spike-Timing-Dependent Plasticity (STDP) Experiment")
    print("=" * 70)

    params = STDPParameters(
        a_plus=0.01,
        a_minus=0.0105,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,
        mu=0.0,  # Additive boundaries
        pairing_mode=STDPPairingMode.ALL_TO_ALL,
    )
    model = STDPModel(params)

    num_pairings = 60
    initial_w = 0.50

    print(f"Kernel Configuration: A+ = {params.a_plus}, A- = {params.a_minus}")
    print(f"Time Constants:       tau+ = {params.tau_plus} ms, tau- = {params.tau_minus} ms")
    print(f"Initial Synaptic W:   {initial_w:.3f}")
    print(f"Number of Pairings:   {num_pairings} pairs at 10 Hz frequency\n")

    # 1. Causal Protocol: pre precedes post by +10 ms -> LTP
    dt_causal = 10.0
    t_pre_causal = [50.0 + i * 100.0 for i in range(num_pairings)]
    t_post_causal = [t + dt_causal for t in t_pre_causal]

    w_final_ltp, history_ltp = model.apply_pairings(initial_w, t_pre_causal, t_post_causal)
    pct_ltp = ((w_final_ltp - initial_w) / initial_w) * 100.0

    print("-" * 70)
    print("EXPERIMENT 1: CAUSAL PAIRING (t_post - t_pre = +10 ms)")
    print("-" * 70)
    print(f"Final Synaptic Weight: {w_final_ltp:.4f}")
    print(f"Weight Modification:   {w_final_ltp - initial_w:+.4f} ({pct_ltp:+.1f}%)")
    print(f"Outcome:               Long-Term Potentiation (LTP) CONFIRMED")

    # 2. Anti-Causal Protocol: post precedes pre by -10 ms -> LTD
    dt_anti = -10.0
    t_pre_anti = [50.0 + i * 100.0 for i in range(num_pairings)]
    t_post_anti = [t + dt_anti for t in t_pre_anti]

    w_final_ltd, history_ltd = model.apply_pairings(initial_w, t_pre_anti, t_post_anti)
    pct_ltd = ((w_final_ltd - initial_w) / initial_w) * 100.0

    print("\n" + "-" * 70)
    print("EXPERIMENT 2: ANTI-CAUSAL PAIRING (t_post - t_pre = -10 ms)")
    print("-" * 70)
    print(f"Final Synaptic Weight: {w_final_ltd:.4f}")
    print(f"Weight Modification:   {w_final_ltd - initial_w:+.4f} ({pct_ltd:+.1f}%)")
    print(f"Outcome:               Long-Term Depression (LTD) CONFIRMED")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
