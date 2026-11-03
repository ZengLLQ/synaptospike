"""Biophysical Conductance-Based Synaptic Receptors.

Implements AMPA, NMDA (with Year & Stevens voltage-dependent magnesium block),
GABA_A, and GABA_B kinetic dual-exponential postsynaptic conductances.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class SynapseType(Enum):
    """Physiological neurotransmitter receptor classification."""

    AMPA = "ampa"
    NMDA = "nmda"
    GABA_A = "gaba_a"
    GABA_B = "gaba_b"


@dataclasses.dataclass(frozen=True)
class SynapticParameters:
    """Biophysical kinetic parameters for a postsynaptic receptor."""

    syn_type: SynapseType
    g_max: float  # Maximal peak conductance (uS or nS)
    e_rev: float  # Reversal potential (mV)
    tau_rise: float  # Rise time constant (ms)
    tau_decay: float  # Decay time constant (ms)
    mg_concentration: float = 1.0  # Extracellular Mg2+ concentration (mM, for NMDA)

    @classmethod
    def ampa_default(cls, g_max: float = 1.0) -> SynapticParameters:
        return cls(
            syn_type=SynapseType.AMPA,
            g_max=g_max,
            e_rev=0.0,
            tau_rise=0.5,
            tau_decay=2.4,
        )

    @classmethod
    def nmda_default(cls, g_max: float = 0.5, mg_conc: float = 1.0) -> SynapticParameters:
        return cls(
            syn_type=SynapseType.NMDA,
            g_max=g_max,
            e_rev=0.0,
            tau_rise=4.0,
            tau_decay=80.0,
            mg_concentration=mg_conc,
        )

    @classmethod
    def gaba_a_default(cls, g_max: float = 1.5) -> SynapticParameters:
        return cls(
            syn_type=SynapseType.GABA_A,
            g_max=g_max,
            e_rev=-70.0,
            tau_rise=0.5,
            tau_decay=7.0,
        )

    @classmethod
    def gaba_b_default(cls, g_max: float = 0.3) -> SynapticParameters:
        return cls(
            syn_type=SynapseType.GABA_B,
            g_max=g_max,
            e_rev=-90.0,
            tau_rise=30.0,
            tau_decay=150.0,
        )


@dataclasses.dataclass
class ReceptorCurrents:
    """Breakdown of compound postsynaptic currents (uA or nA)."""

    i_total: float
    i_ampa: float = 0.0
    i_nmda: float = 0.0
    i_gaba_a: float = 0.0
    i_gaba_b: float = 0.0


class ConductanceSynapse:
    """Postsynaptic receptor complex with dual-exponential kinetic states."""

    def __init__(self, params: SynapticParameters) -> None:
        self.params = params
        self.s: float = 0.0  # Conductance activation state
        self.h: float = 0.0  # Intermediate rise state

        # Compute peak normalization factor so peak s reaches 1.0
        tr, td = self.params.tau_rise, self.params.tau_decay
        if abs(td - tr) > 1e-5:
            t_peak = (tr * td / (td - tr)) * np.log(td / tr)
            self.norm_factor = (td - tr) / (tr * td * (np.exp(-t_peak / td) - np.exp(-t_peak / tr)))
        else:
            self.norm_factor = np.e / tr

    def on_presynaptic_spike(self, weight: float = 1.0) -> None:
        """Triggered when an upstream presynaptic action potential arrives."""
        self.h += weight * self.norm_factor

    def step(self, dt: float) -> None:
        """Advance internal gating variables by time step dt (ms) using exact exponential integration."""
        exp_decay = np.exp(-dt / self.params.tau_decay)
        exp_rise = np.exp(-dt / self.params.tau_rise)

        tr, td = self.params.tau_rise, self.params.tau_decay
        if abs(td - tr) > 1e-5:
            # Analytical convolution step: (exp_decay - exp_rise) is positive
            delta_s = self.h * (tr * td / (td - tr)) * (exp_decay - exp_rise)
            self.s = self.s * exp_decay + delta_s
        else:
            self.s = (self.s + dt * self.h) * exp_decay

        self.h = self.h * exp_rise

    def mg_block(self, v_post: float) -> float:
        """Jahr & Stevens (1990) voltage-dependent Mg2+ block for NMDA receptors."""
        if self.params.syn_type != SynapseType.NMDA:
            return 1.0
        mg = self.params.mg_concentration
        return 1.0 / (1.0 + (mg / 3.57) * np.exp(-0.062 * v_post))

    def current(self, v_post: float) -> float:
        """Evaluate postsynaptic current I_syn = g_max * s * B(V) * (V - E_rev)."""
        g = self.params.g_max * self.s * self.mg_block(v_post)
        return float(g * (v_post - self.params.e_rev))
