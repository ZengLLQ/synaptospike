"""Unit & Invariant Tests for Spectral Analysis and Fano Factors."""

import numpy as np
import pytest
from synaptospike.analysis.spectral import (
    compute_lfp,
    compute_power_spectral_density,
    compute_band_powers,
    compute_fano_factor,
    compute_isi_statistics,
)


class TestSpectralAnalysis:
    """Test suite for neural electrophysiological signal processing."""

    def test_psd_frequency_peak_recovery(self) -> None:
        """Welch PSD estimation must accurately detect the frequency of a synthetic sine wave."""
        fs = 1000.0  # 1 kHz sampling rate
        duration = 2.0  # 2 seconds
        t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)

        # 40 Hz gamma rhythm sine wave + noise
        target_freq = 40.0
        signal_val = np.sin(2 * np.pi * target_freq * t) + 0.1 * np.random.default_rng(42).normal(size=len(t))

        psd_res = compute_power_spectral_density(signal_val, sampling_rate_hz=fs, nperseg=512)
        assert pytest.approx(psd_res.peak_frequency_hz, abs=2.0) == target_freq

    def test_band_powers_integration(self) -> None:
        """Synthetic gamma oscillation power must concentrate primarily in the gamma band (30-80 Hz)."""
        fs = 1000.0
        t = np.linspace(0.0, 2.0, 2000, endpoint=False)
        gamma_signal = np.sin(2 * np.pi * 50.0 * t)  # 50 Hz gamma

        psd_res = compute_power_spectral_density(gamma_signal, sampling_rate_hz=fs)
        bands = compute_band_powers(psd_res)

        # Gamma power should dominate theta and delta by orders of magnitude
        assert bands.gamma > bands.theta * 10.0
        assert bands.gamma > bands.delta * 10.0
        assert bands.gamma > 0.5 * bands.total_power

    def test_fano_factor_poisson_property(self) -> None:
        """For a synthetic Poisson spike train, Fano factor should be approximately 1.0."""
        rng = np.random.default_rng(999)
        rate_hz = 20.0
        duration_s = 50.0
        # Exponential inter-arrival times
        isi_samples = rng.exponential(scale=1.0 / rate_hz, size=int(rate_hz * duration_s * 1.5))
        spike_times_s = np.cumsum(isi_samples)
        spike_times_s = spike_times_s[spike_times_s < duration_s]
        spike_times_ms = spike_times_s * 1000.0

        fano = compute_fano_factor(spike_times_ms, duration_ms=duration_s * 1000.0, bin_size_ms=100.0)
        # Poisson Fano factor is unity (variance / mean = 1.0)
        assert pytest.approx(fano, abs=0.25) == 1.0
