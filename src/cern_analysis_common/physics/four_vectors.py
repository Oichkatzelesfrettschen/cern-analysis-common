"""Four-vector operations for relativistic kinematics.

Implements Lorentz four-vectors with (+,-,-,-) metric convention.
All calculations assume natural units (c=1).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class FourVector:
    """Lorentz four-vector (E, px, py, pz).

    Uses (+,-,-,-) metric convention common in particle physics.

    Parameters
    ----------
    E : float or array
        Energy component
    px : float or array
        x-momentum component
    py : float or array
        y-momentum component
    pz : float or array
        z-momentum component

    Examples
    --------
    >>> p = FourVector(E=10.0, px=3.0, py=4.0, pz=0.0)
    >>> p.mass
    8.660254...
    >>> p.pt
    5.0
    """

    E: Union[float, np.ndarray]
    px: Union[float, np.ndarray]
    py: Union[float, np.ndarray]
    pz: Union[float, np.ndarray]

    @property
    def mass(self) -> Union[float, np.ndarray]:
        """Invariant mass: sqrt(E^2 - p^2)."""
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(np.maximum(m2, 0))

    @property
    def mass_squared(self) -> Union[float, np.ndarray]:
        """Invariant mass squared (can be negative for spacelike)."""
        return self.E**2 - self.px**2 - self.py**2 - self.pz**2

    @property
    def pt(self) -> Union[float, np.ndarray]:
        """Transverse momentum."""
        return np.sqrt(self.px**2 + self.py**2)

    @property
    def p(self) -> Union[float, np.ndarray]:
        """Total 3-momentum magnitude."""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def eta(self) -> Union[float, np.ndarray]:
        """Pseudorapidity."""
        return pseudorapidity(self.pt, self.pz)

    @property
    def phi(self) -> Union[float, np.ndarray]:
        """Azimuthal angle."""
        return np.arctan2(self.py, self.px)

    @property
    def rapidity(self) -> Union[float, np.ndarray]:
        """Rapidity: 0.5 * ln((E+pz)/(E-pz))."""
        return rapidity(self.E, self.pz)

    @property
    def mt(self) -> Union[float, np.ndarray]:
        """Transverse mass: sqrt(E^2 - pz^2)."""
        return np.sqrt(np.maximum(self.E**2 - self.pz**2, 0))

    def __add__(self, other: "FourVector") -> "FourVector":
        """Add two four-vectors."""
        return FourVector(
            E=self.E + other.E,
            px=self.px + other.px,
            py=self.py + other.py,
            pz=self.pz + other.pz,
        )

    def __sub__(self, other: "FourVector") -> "FourVector":
        """Subtract two four-vectors."""
        return FourVector(
            E=self.E - other.E,
            px=self.px - other.px,
            py=self.py - other.py,
            pz=self.pz - other.pz,
        )

    def __mul__(self, scalar: float) -> "FourVector":
        """Multiply by scalar."""
        return FourVector(
            E=self.E * scalar,
            px=self.px * scalar,
            py=self.py * scalar,
            pz=self.pz * scalar,
        )

    def __rmul__(self, scalar: float) -> "FourVector":
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    def dot(self, other: "FourVector") -> Union[float, np.ndarray]:
        """Minkowski dot product with (+,-,-,-) metric."""
        return (
            self.E * other.E
            - self.px * other.px
            - self.py * other.py
            - self.pz * other.pz
        )

    @classmethod
    def from_pt_eta_phi_m(
        cls,
        pt: Union[float, np.ndarray],
        eta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray],
        m: Union[float, np.ndarray],
    ) -> "FourVector":
        """Create from (pt, eta, phi, mass) coordinates.

        Parameters
        ----------
        pt : float or array
            Transverse momentum
        eta : float or array
            Pseudorapidity
        phi : float or array
            Azimuthal angle
        m : float or array
            Invariant mass

        Returns
        -------
        FourVector
        """
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(pt**2 * np.cosh(eta) ** 2 + m**2)
        return cls(E=E, px=px, py=py, pz=pz)

    @classmethod
    def from_pt_eta_phi_E(
        cls,
        pt: Union[float, np.ndarray],
        eta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray],
        E: Union[float, np.ndarray],
    ) -> "FourVector":
        """Create from (pt, eta, phi, E) coordinates."""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        return cls(E=E, px=px, py=py, pz=pz)


def invariant_mass(p1: FourVector, p2: FourVector) -> Union[float, np.ndarray]:
    """Compute invariant mass of two-particle system.

    Parameters
    ----------
    p1 : FourVector
        First particle
    p2 : FourVector
        Second particle

    Returns
    -------
    float or array
        Invariant mass sqrt((p1+p2)^2)
    """
    total = p1 + p2
    return total.mass


def invariant_mass_from_arrays(
    pt1: np.ndarray,
    eta1: np.ndarray,
    phi1: np.ndarray,
    m1: np.ndarray,
    pt2: np.ndarray,
    eta2: np.ndarray,
    phi2: np.ndarray,
    m2: np.ndarray,
) -> np.ndarray:
    """Compute invariant mass from kinematic arrays.

    Efficient vectorized computation without creating FourVector objects.

    Parameters
    ----------
    pt1, eta1, phi1, m1 : array
        Kinematic variables for particle 1
    pt2, eta2, phi2, m2 : array
        Kinematic variables for particle 2

    Returns
    -------
    array
        Invariant masses
    """
    # Energy
    E1 = np.sqrt(pt1**2 * np.cosh(eta1) ** 2 + m1**2)
    E2 = np.sqrt(pt2**2 * np.cosh(eta2) ** 2 + m2**2)

    # Momentum components
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    # Invariant mass
    E_tot = E1 + E2
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2

    m2_inv = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    return np.sqrt(np.maximum(m2_inv, 0))


def transverse_momentum(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Compute transverse momentum."""
    return np.sqrt(px**2 + py**2)


def pseudorapidity(pt: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Compute pseudorapidity eta = -ln(tan(theta/2)).

    Parameters
    ----------
    pt : array
        Transverse momentum
    pz : array
        z-momentum

    Returns
    -------
    array
        Pseudorapidity values
    """
    p = np.sqrt(pt**2 + pz**2)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = 0.5 * np.log((p + pz) / (p - pz))
    return eta


def rapidity(E: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Compute rapidity y = 0.5 * ln((E+pz)/(E-pz)).

    Parameters
    ----------
    E : array
        Energy
    pz : array
        z-momentum

    Returns
    -------
    array
        Rapidity values
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log((E + pz) / (E - pz))
    return y


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Compute azimuthal angle difference in [-pi, pi].

    Parameters
    ----------
    phi1 : array
        First azimuthal angle
    phi2 : array
        Second azimuthal angle

    Returns
    -------
    array
        Delta phi values wrapped to [-pi, pi]
    """
    dphi = phi1 - phi2
    # Wrap to [-pi, pi]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return dphi


def delta_r(
    eta1: np.ndarray,
    phi1: np.ndarray,
    eta2: np.ndarray,
    phi2: np.ndarray,
) -> np.ndarray:
    """Compute angular separation Delta R = sqrt(deta^2 + dphi^2).

    Parameters
    ----------
    eta1 : array
        First pseudorapidity
    phi1 : array
        First azimuthal angle
    eta2 : array
        Second pseudorapidity
    phi2 : array
        Second azimuthal angle

    Returns
    -------
    array
        Delta R values
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def boost_to_cm(p1: FourVector, p2: FourVector) -> Tuple[FourVector, FourVector]:
    """Boost two particles to their center-of-mass frame.

    Parameters
    ----------
    p1 : FourVector
        First particle in lab frame
    p2 : FourVector
        Second particle in lab frame

    Returns
    -------
    tuple of FourVector
        (p1_cm, p2_cm) in center-of-mass frame
    """
    total = p1 + p2
    E_cm = total.mass

    # Boost velocity (beta)
    bx = total.px / total.E
    by = total.py / total.E
    bz = total.pz / total.E
    b2 = bx**2 + by**2 + bz**2
    gamma = 1.0 / np.sqrt(1.0 - b2)

    def boost(p: FourVector) -> FourVector:
        bp = bx * p.px + by * p.py + bz * p.pz
        gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0

        E_new = gamma * (p.E - bp)
        px_new = p.px + bx * (gamma2 * bp - gamma * p.E)
        py_new = p.py + by * (gamma2 * bp - gamma * p.E)
        pz_new = p.pz + bz * (gamma2 * bp - gamma * p.E)

        return FourVector(E=E_new, px=px_new, py=py_new, pz=pz_new)

    return boost(p1), boost(p2)
