"""Physical constants for particle physics.

All masses in GeV/c^2 (natural units where c=1).
PDG 2024 values where available.
"""

# Particle masses [GeV/c^2]
PROTON_MASS = 0.938272088
NEUTRON_MASS = 0.939565420
PION_CHARGED_MASS = 0.13957039
PION_NEUTRAL_MASS = 0.1349768
PION_MASS = PION_CHARGED_MASS  # Default to charged pion
KAON_CHARGED_MASS = 0.493677
KAON_NEUTRAL_MASS = 0.497611
KAON_MASS = KAON_CHARGED_MASS  # Default to charged kaon
ELECTRON_MASS = 0.00051099895
MUON_MASS = 0.1056583755
TAU_MASS = 1.77686
W_MASS = 80.377
Z_MASS = 91.1876
HIGGS_MASS = 125.25
TOP_MASS = 172.69
BOTTOM_MASS = 4.18  # MS-bar mass

# Fundamental constants
C_LIGHT = 299792458.0  # Speed of light [m/s]
HBAR_C = 0.1973269804  # hbar*c [GeV*fm]
ALPHA_EM = 1.0 / 137.035999084  # Fine structure constant
ALPHA_S_MZ = 0.1179  # Strong coupling at M_Z

# Unit conversions
GEV_TO_MEV = 1000.0
MEV_TO_GEV = 0.001
GEV_TO_TEV = 0.001
TEV_TO_GEV = 1000.0

# ALICE specific
ALICE_MAGNETIC_FIELD = 0.5  # Tesla (nominal)
ALICE_RAPIDITY_ACCEPTANCE = 0.9  # |y| < 0.9 for central barrel

# LHC beam energies [TeV per nucleon]
LHC_PP_13TEV = 13.0
LHC_PP_13P6TEV = 13.6  # Run 3
LHC_PBPB_5P02TEV = 5.02
LHC_PBPB_5P36TEV = 5.36  # Run 3
