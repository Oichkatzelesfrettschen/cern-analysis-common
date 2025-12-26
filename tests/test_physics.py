"""Tests for physics module."""

import numpy as np
import pytest

from cern_analysis_common.constants import PION_MASS, PROTON_MASS, Z_MASS
from cern_analysis_common.physics import (
    FourVector,
    invariant_mass,
    transverse_momentum,
    pseudorapidity,
    delta_r,
    delta_phi,
    apply_pt_cut,
    apply_eta_cut,
    apply_mass_window,
    combine_cuts,
    efficiency_ratio,
    clopper_pearson_interval,
)


class TestFourVector:
    """Tests for FourVector class."""

    def test_mass_at_rest(self):
        """Test mass of particle at rest."""
        p = FourVector(E=PROTON_MASS, px=0, py=0, pz=0)
        assert abs(p.mass - PROTON_MASS) < 1e-6

    def test_mass_moving(self):
        """Test invariant mass is Lorentz invariant."""
        # Proton with momentum
        E = np.sqrt(PROTON_MASS**2 + 1.0**2)
        p = FourVector(E=E, px=1.0, py=0, pz=0)
        assert abs(p.mass - PROTON_MASS) < 1e-6

    def test_transverse_momentum(self):
        """Test pt calculation."""
        p = FourVector(E=10, px=3, py=4, pz=5)
        assert abs(p.pt - 5.0) < 1e-6

    def test_addition(self):
        """Test four-vector addition."""
        p1 = FourVector(E=5, px=1, py=2, pz=3)
        p2 = FourVector(E=7, px=2, py=1, pz=4)
        total = p1 + p2
        assert total.E == 12
        assert total.px == 3
        assert total.py == 3
        assert total.pz == 7

    def test_from_pt_eta_phi_m(self):
        """Test construction from pt, eta, phi, m."""
        pt = 10.0
        eta = 0.5
        phi = 0.3
        m = PION_MASS

        p = FourVector.from_pt_eta_phi_m(pt, eta, phi, m)

        assert abs(p.pt - pt) < 1e-6
        assert abs(p.eta - eta) < 1e-6
        assert abs(p.phi - phi) < 1e-6
        assert abs(p.mass - m) < 1e-4

    def test_invariant_mass_two_particles(self):
        """Test invariant mass of two-particle system."""
        # Two pions -> should give rho meson mass region
        p1 = FourVector.from_pt_eta_phi_m(2.0, 0.3, 0.0, PION_MASS)
        p2 = FourVector.from_pt_eta_phi_m(2.0, -0.3, np.pi, PION_MASS)

        m_inv = invariant_mass(p1, p2)
        assert m_inv > 2 * PION_MASS  # At least twice pion mass


class TestKinematics:
    """Tests for kinematic functions."""

    def test_delta_phi_wrap(self):
        """Test delta phi wrapping."""
        # Should wrap to [-pi, pi]
        dphi = delta_phi(np.array([3.0]), np.array([-3.0]))
        assert abs(dphi[0]) < np.pi

    def test_delta_r_same_direction(self):
        """Test delta R for same direction is zero."""
        dr = delta_r(
            np.array([0.5]), np.array([0.3]),
            np.array([0.5]), np.array([0.3])
        )
        assert abs(dr[0]) < 1e-10

    def test_delta_r_opposite(self):
        """Test delta R for opposite hemispheres."""
        dr = delta_r(
            np.array([0.5]), np.array([0.0]),
            np.array([-0.5]), np.array([np.pi])
        )
        # deta = 1, dphi = pi
        expected = np.sqrt(1.0 + np.pi**2)
        assert abs(dr[0] - expected) < 1e-6

    def test_pseudorapidity_midrapidity(self):
        """Test eta = 0 at midrapidity."""
        pt = np.array([10.0])
        pz = np.array([0.0])
        eta = pseudorapidity(pt, pz)
        assert abs(eta[0]) < 1e-10


class TestCuts:
    """Tests for selection cuts."""

    def test_pt_cut_lower(self):
        """Test pt minimum cut."""
        pt = np.array([0.5, 1.0, 1.5, 2.0])
        mask = apply_pt_cut(pt, pt_min=1.0)
        assert list(mask) == [False, True, True, True]

    def test_eta_cut_symmetric(self):
        """Test symmetric eta cut."""
        eta = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        mask = apply_eta_cut(eta, eta_max=0.9)
        assert list(mask) == [False, True, True, True, False]

    def test_mass_window(self):
        """Test mass window cut."""
        mass = np.array([85, 90, 91.2, 95, 100])
        mask = apply_mass_window(mass, center=Z_MASS, width=5)
        # Z_MASS = 91.1876, so 86.2 to 96.2
        assert mask[2]  # 91.2 in window
        assert not mask[0]  # 85 outside

    def test_combine_cuts_and(self):
        """Test combining cuts with AND."""
        mask1 = np.array([True, True, False, False])
        mask2 = np.array([True, False, True, False])
        combined = combine_cuts(mask1, mask2)
        assert list(combined) == [True, False, False, False]


class TestEfficiency:
    """Tests for efficiency calculations."""

    def test_efficiency_ratio_simple(self):
        """Test basic efficiency calculation."""
        eff, err = efficiency_ratio(50, 100)
        assert abs(eff - 0.5) < 1e-6
        assert err > 0

    def test_efficiency_ratio_zero(self):
        """Test efficiency with zero denominator."""
        eff, err = efficiency_ratio(0, 0)
        assert eff == 0.0
        assert err == 0.0

    def test_clopper_pearson_coverage(self):
        """Test Clopper-Pearson interval contains true value."""
        # Perfect efficiency
        eff, lo, hi = clopper_pearson_interval(100, 100)
        assert eff == 1.0
        assert hi == 1.0
        assert lo < 1.0

        # Zero efficiency
        eff, lo, hi = clopper_pearson_interval(0, 100)
        assert eff == 0.0
        assert lo == 0.0
        assert hi > 0.0

    def test_clopper_pearson_50_percent(self):
        """Test interval for 50% efficiency."""
        eff, lo, hi = clopper_pearson_interval(50, 100, confidence=0.68)
        assert abs(eff - 0.5) < 1e-6
        assert lo < 0.5
        assert hi > 0.5
