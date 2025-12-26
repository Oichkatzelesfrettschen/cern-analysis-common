import numpy as np

class FourVector:
    """
    Simple FourVector class.
    """
    def __init__(self, E, px, py, pz):
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz
        
    @property
    def pt(self):
        return np.sqrt(self.px**2 + self.py**2)
        
    @property
    def eta(self):
        return -0.5 * np.log((np.sqrt(self.px**2 + self.py**2 + self.pz**2) - self.pz) / 
                             (np.sqrt(self.px**2 + self.py**2 + self.pz**2) + self.pz))
                             
    @property
    def phi(self):
        return np.arctan2(self.py, self.px)
        
    @property
    def mass(self):
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(np.abs(m2)) * np.sign(m2)

def invariant_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
    """Calculate invariant mass of two particles."""
    E = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    m2 = E**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.abs(m2))

def delta_r(eta1, phi1, eta2, phi2):
    """Calculate Delta R between two particles."""
    deta = eta1 - eta2
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)
