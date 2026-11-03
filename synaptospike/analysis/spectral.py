"""Electrophysiological Spectral Analysis & Firing Statistics.

Computes Local Field Potential (LFP) spectral dynamics via Welch's method,
canonical neuro-rhythm band powers (theta, alpha, beta, gamma), Inter-Spike
Interval (ISI) distributions, and Fano factors.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple, List
import numpy as np
from scipy import signal


@dataclasses.dataclass(frozen=True)
class SpectralDensityResult:
    """Welch power spectral density profile."""

    frequencies: np.ndarray  # Frequency axis (Hz)
    psd: np.ndarray  # Spectral power density (V^2 / Hz)
    peak_frequency_hz: float  # Dominant rhythm frequency (Hz)
    peak_power: float  # Maximum power at peak


@dataclasses.dataclass(frozen=True)
class NeuroBandPowers:
    """Integrated powers across physiological EEG/LFP frequency bands."""

    delta: float  # 1 - 4 Hz
    theta: float  # 4 - 8 Hz
    alpha: float  # 8 - 12 Hz
    beta: float  # 12 - 30 Hz
    gamma: float  # 30 - 80 Hz
    high_gamma: float  # 80 - 150 Hz
    total_power: float


@dataclasses.dataclass(frozen=True)
class ISIStatistics:
    """Statistical properties of inter-spike intervals."""

    mean_isi_ms: float
    std_isi_ms: float
    cv_isi: float  # Coefficient of variation (sigma / mu)
    all_isis: np.ndarray


def compute_lfp(
    compound_current: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float = 300.0,
) -> np.ndarray:
    """Low-pass Butterworth filter compound synaptic currents to produce a clean LFP proxy."""
    nyquist = 0.5 * sampling_rate_hz
    normalized_cutoff = min(0.99, cutoff_hz / nyquist)
    b, a = signal.butter(4, normalized_cutoff, btype="low", analog=False)
    filtered_lfp = signal.filtfilt(b, a, compound_current)
    # Mean-center to remove baseline DC offset
    return np.asarray(filtered_lfp - np.mean(filtered_lfp), dtype=np.float64)


def compute_power_spectral_density(
    signal_trace: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int | None = None,
) -> SpectralDensityResult:
    """Estimate Power Spectral Density (PSD) using Welch's averaged periodogram method."""
    trace_len = len(signal_trace)
    if nperseg is None:
        nperseg = min(trace_len, max(64, int(sampling_rate_hz)))

    freqs, psd = signal.welch(signal_trace, fs=sampling_rate_hz, nperseg=nperseg)

    # Ignore 0 Hz DC term for peak identification
    non_dc_mask = freqs > 1.0
    if np.any(non_dc_mask):
        peak_idx = np.argmax(psd[non_dc_mask])
        peak_freq = float(freqs[non_dc_mask][peak_idx])
        peak_pwr = float(psd[non_dc_mask][peak_idx])
    else:
        peak_freq = 0.0
        peak_pwr = 0.0

    return SpectralDensityResult(
        frequencies=freqs,
        psd=psd,
        peak_frequency_hz=peak_freq,
        peak_power=peak_pwr,
    )


def compute_band_powers(
    psd_result: SpectralDensityResult,
) -> NeuroBandPowers:
    """Integrate power in standard neurological frequency bands via trapezoidal quadrature."""
    freqs = psd_result.frequencies
    psd = psd_result.psd

    def _trapz_quad(y: np.ndarray, x: np.ndarray) -> float:
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        elif hasattr(np, "trapz"):
            return float(np.trapz(y, x))
        else:
            return float(scipy.integrate.trapezoid(y, x))

    def integrate_band(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        if np.sum(mask) < 2:
            return 0.0
        return _trapz_quad(psd[mask], freqs[mask])

    delta = integrate_band(1.0, 4.0)
    theta = integrate_band(4.0, 8.0)
    alpha = integrate_band(8.0, 12.0)
    beta = integrate_band(12.0, 30.0)
    gamma = integrate_band(30.0, 80.0)
    high_gamma = integrate_band(80.0, 150.0)
    total = _trapz_quad(psd[freqs >= 1.0], freqs[freqs >= 1.0]) if len(freqs) > 1 else 0.0

    return NeuroBandPowers(
        delta=delta,
        theta=theta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        high_gamma=high_gamma,
        total_power=total,
    )


def compute_isi_statistics(
    spike_times: np.ndarray,
    neuron_ids: np.ndarray | None = None,
) -> ISIStatistics:
    """Compute mean, standard deviation, and coefficient of variation (CV_ISI) of inter-spike intervals."""
    if len(spike_times) < 2:
        return ISIStatistics(mean_isi_ms=0.0, std_isi_ms=0.0, cv_isi=0.0, all_isis=np.array([]))

    if neuron_ids is None:
        # Single spike train
        sorted_times = np.sort(spike_times)
        isis = np.diff(sorted_times)
    else:
        # Multi-unit recording: compute per-neuron ISIs and aggregate
        unique_ids = np.unique(neuron_ids)
        all_isis_list: List[float] = []
        for uid in unique_ids:
            u_times = np.sort(spike_times[neuron_ids == uid])
            if len(u_times) >= 2:
                all_isis_list.extend(np.diff(u_times))
        isis = np.array(all_isis_list, dtype=np.float64)

    if len(isis) == 0:
        return ISIStatistics(mean_isi_ms=0.0, std_isi_ms=0.0, cv_isi=0.0, all_isis=np.array([]))

    mean_isi = float(np.mean(isis))
    std_isi = float(np.std(isis))
    cv_isi = float(std_isi / mean_isi) if mean_isi > 1e-9 else 0.0

    return ISIStatistics(
        mean_isi_ms=mean_isi,
        std_isi_ms=std_isi,
        cv_isi=cv_isi,
        all_isis=isis,
    )


def compute_fano_factor(
    spike_times: np.ndarray,
    duration_ms: float,
    bin_size_ms: float = 50.0,
) -> float:
    """Calculate the Fano factor (variance / mean) of spike counts across non-overlapping time bins."""
    if len(spike_times) == 0 or duration_ms <= 0:
        return 0.0

    num_bins = max(1, int(duration_ms / bin_size_ms))
    bin_edges = np.linspace(0.0, duration_ms, num_bins + 1)
    counts, _ = np.histogram(spike_times, bins=bin_edges)

    mean_count = float(np.mean(counts))
    var_count = float(np.var(counts))

    if mean_count > 1e-9:
        return float(var_count / mean_count)
    return 0.0
