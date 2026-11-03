"""SynaptoSpike: High-Performance Biophysical Neural Dynamics & Network Simulation Engine.

Provides biophysical Hodgkin-Huxley conductance systems, Morris-Lecar phase-plane
analysis, Izhikevich phenomenological spiking models, multi-receptor conductance
synapses (AMPA, NMDA, GABA_A, GABA_B), Spike-Timing-Dependent Plasticity (STDP),
and recurrent Brunel cortical microcircuit simulation.
"""

from synaptospike.models.hodgkin_huxley import (
    HodgkinHuxley,
    HHParameters,
    HHState,
    HHSolution,
)
from synaptospike.models.morris_lecar import (
    MorrisLecar,
    MLParameters,
    MLState,
    MLPhasePlaneAnalysis,
)
from synaptospike.models.izhikevich import (
    IzhikevichNeuron,
    IzhikevichPopulation,
    IzhikevichPreset,
    IzhikevichState,
)
from synaptospike.synapses.conductance import (
    ConductanceSynapse,
    SynapseType,
    SynapticParameters,
    ReceptorCurrents,
)
from synaptospike.plasticity.stdp import (
    STDPModel,
    STDPParameters,
    STDPPairingMode,
)
from synaptospike.network.brunel import (
    BrunelNetwork,
    BrunelParameters,
    NetworkState,
    NetworkSpikeRecord,
)
from synaptospike.analysis.spectral import (
    compute_lfp,
    compute_power_spectral_density,
    compute_band_powers,
    compute_isi_statistics,
    compute_fano_factor,
)

__version__ = "0.1.0"
__all__ = [
    "HodgkinHuxley",
    "HHParameters",
    "HHState",
    "HHSolution",
    "MorrisLecar",
    "MLParameters",
    "MLState",
    "MLPhasePlaneAnalysis",
    "IzhikevichNeuron",
    "IzhikevichPopulation",
    "IzhikevichPreset",
    "IzhikevichState",
    "ConductanceSynapse",
    "SynapseType",
    "SynapticParameters",
    "ReceptorCurrents",
    "STDPModel",
    "STDPParameters",
    "STDPPairingMode",
    "BrunelNetwork",
    "BrunelParameters",
    "NetworkState",
    "NetworkSpikeRecord",
    "compute_lfp",
    "compute_power_spectral_density",
    "compute_band_powers",
    "compute_isi_statistics",
    "compute_fano_factor",
]
