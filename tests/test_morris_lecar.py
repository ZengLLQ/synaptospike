"""Unit & Invariant Tests for Morris-Lecar Phase-Plane Analysis."""

import numpy as np
import pytest
from synaptospike.models.morris_lecar import MorrisLecar, MLParameters, MLState


class TestMorrisLecar:
    """Test suite for Morris-Lecar 2D nonlinear model."""

    def test_nullclines_evaluation(self) -> None:
        """V-nullcline and w-nullcline must evaluate smoothly without NaN in physiological range."""
        ml = MorrisLecar(MLParameters.hopf_regime())
        v_grid = np.linspace(-70.0, 30.0, 200)

        w_null = ml.compute_w_nullcline(v_grid)
        assert np.all(w_null >= 0.0)
        assert np.all(w_null <= 1.0)
        assert np.all(np.diff(w_null) > 0.0)  # w_inf is monotonically increasing

        v_null = ml.compute_v_nullcline(v_grid, i_inj=80.0)
        assert not np.any(np.isnan(v_null))

    def test_equilibrium_finding(self) -> None:
        """Find fixed points where dV/dt = 0 and dw/dt = 0 simultaneously."""
        ml = MorrisLecar(MLParameters.hopf_regime())
        equilibria = ml.find_equilibria(i_inj=0.0)

        assert len(equilibria) >= 1
        v_star, w_star = equilibria[0]
        # In resting regime without current, membrane is polarized
        assert v_star < -40.0
        assert 0.0 <= w_star <= 1.0

        # Evaluate derivatives at fixed point
        derivs = ml.derivatives(MLState(v=v_star, w=w_star), i_inj=0.0)
        assert abs(derivs[0]) < 1e-3
        assert abs(derivs[1]) < 1e-3

    def test_analytical_jacobian(self) -> None:
        """Analytical Jacobian matrix must match numerical finite-difference approximation."""
        ml = MorrisLecar(MLParameters.hopf_regime())
        v0, w0, i_inj = -30.0, 0.25, 50.0

        j_analytical = ml.jacobian(v0, w0, i_inj)

        # Finite difference
        eps = 1e-6
        d_v = ml.derivatives(MLState(v0 + eps, w0), i_inj) - ml.derivatives(MLState(v0 - eps, w0), i_inj)
        d_w = ml.derivatives(MLState(v0, w0 + eps), i_inj) - ml.derivatives(MLState(v0, w0 - eps), i_inj)

        j_numerical = np.column_stack([d_v / (2 * eps), d_w / (2 * eps)])

        np.testing.assert_allclose(j_analytical, j_numerical, rtol=1e-3, atol=1e-4)

    def test_hopf_bifurcation_oscillation(self) -> None:
        """Current stimulus above critical threshold must trigger sustained limit cycle oscillations."""
        ml = MorrisLecar(MLParameters.hopf_regime())

        # Subthreshold: I_inj = 0 mV -> stays at fixed point
        t, v_rest, _ = ml.simulate(t_span=(0.0, 100.0), dt=0.1, i_inj=0.0)
        assert np.max(v_rest) - np.min(v_rest) < 2.0  # Flat

        # Suprathreshold: I_inj = 100.0 uA/cm^2 with small initial perturbation -> limit cycle oscillations (>40 mV p2p)
        init_perturbed = MLState(v=-20.0, w=0.2)
        t, v_osc, _ = ml.simulate(t_span=(0.0, 250.0), dt=0.1, i_inj=100.0, initial_state=init_perturbed)
        peak_to_peak = np.max(v_osc[len(v_osc) // 2 :]) - np.min(v_osc[len(v_osc) // 2 :])
        assert peak_to_peak > 40.0
