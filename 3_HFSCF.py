import psi4 as ps
import numpy as np
import matplotlib.pyplot as plt
import time
import unittest
from HFSCF import Diag, OrthoS, OrthoF, fock, ElEnergy
ps.core.set_output_file('output.dat', True)
ps.set_memory(int(5e8))
np_memory = 2

# Initialization Molecular Geometry
ps.set_options({'basis': 'sto-3g'})
mol = ps.geometry("""
O  0.000000000000 -0.143225816552 0.000000000000
H  1.638036840407 1.136548822547 -0.000000000000
H -1.638036840407 1.136548822547 -0.000000000000
units bohr
""")

mol.update_geometry()

# Nuclear Repulsion Energy calculation
nre = mol.nuclear_repulsion_energy()

# Initialization Atom Orbital basis set
wave = ps.core.Wavefunction.build(mol, ps.core.get_global_option('basis'))
mints = ps.core.MintsHelper(wave.basisset())

# One-electron integrals
S = np.asarray(mints.ao_overlap())
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())

# Initialization of the core Hamiltonian
Hcore = T + V

# Two-Electron Integrals
ERI = np.asarray(mints.ao_eri())

# Building the Orthogonalization Matrix
Sval, Svec = Diag(S)
Smin = OrthoS(Sval, Svec)

# Initial Guess Density Matrix (AO)
Fz = OrthoF(Hcore, Smin)
Fz_val, Fz_vec = Diag(Fz)

# Eigenvector transformation from orthonormal to non-orthogonal (original) AO basis
Cz = Smin @ Fz_vec
occ = wave.nalpha()
Dens_z = np.einsum('ik,jk->ij',Cz[:, :occ], Cz[:, :occ])

# Compute the Initial SCF Energy
Ez_el = np.einsum('ij,ij->', Dens_z, 2*Hcore)
Ez_tot = Ez_el + nre

class TestHF(unittest.TestCase):
    def test_Hcore(self):
        Hcore_h2o = [[-32.5773954, -7.5788328, -0.0144738, 0.0,  0.0, -1.2401023, -1.2401023],
                     [-7.5788328, -9.2009433, -0.1768902, 0.0,  0.0, -2.9067098, -2.9067098],
                     [-0.0144738, -0.1768902, -7.4153118, 0.0,  0.0, -1.3568683, -1.3568683],
                     [0.0, 0.0, 0.0, -7.4588193, 0.0, -1.6751501, 1.6751501],
                     [0.0, 0.0, 0.0, 0.0, -7.3471449, 0.0, 0.0],
                     [-1.2401023, -2.9067098, -1.3568683, -1.6751501, 0.0, -4.5401711, -1.0711459],
                     [-1.2401023, -2.9067098, -1.3568683, 1.6751501, 0.0, -1.0711459, -4.5401711]]
        self.assertIsNone(np.testing.assert_array_almost_equal(Hcore, Hcore_h2o))
    def test_Smin(self):
        Smin_h2o = [[1.0236346, -0.1368547, -0.0074873, -0.0,  -0.0, 0.0190279, 0.0190279],
                    [-0.1368547, 1.1578632, 0.0721601, 0.0,  0.0, -0.2223326, -0.2223326],
                    [-0.0074873, 0.0721601, 1.038305, 0.0,  0.0, -0.1184626, -0.1184626],
                    [-0.0, 0.0, 0.0, 1.0733148, -0.0, -0.1757583, 0.1757583],
                    [-0.0, 0.0, -0.0, 0.0, 1.0, -0.0, -0.0],
                    [0.0190279, -0.2223326, -0.1184626, -0.1757583, -0.0, 1.1297234, -0.0625975],
                    [0.0190279, -0.2223326, -0.1184626, 0.1757583, -0.0, -0.0625975, 1.1297234]]
        self.assertIsNone(np.testing.assert_array_almost_equal(Smin, Smin_h2o))
    def test_InitialElEn(self):
        self.assertAlmostEqual(Ez_el, -125.842077437699)
    def test_InitialFock(self):
        InitialFock = [[-32.2545866, -2.7914909, 0.008611, -0.0,  0.0, -0.1812967, -0.1812967],
                       [-2.7914909, -8.2368891, -0.2282926, -0.0,  0.0, -0.3857987, -0.3857987],
                       [0.008611, -0.2282926, -7.4570295, -0.0, -0.0, -0.1102196, -0.1102196],
                       [-0.0, -0.0, -0.0, -7.542889, 0.0, -0.1132121, 0.1132121],
                       [0.0, 0.0, 0.0, -0.0, -7.3471449, 0.0, 0.0],
                       [-0.1812967, -0.3857987, -0.1102196, -0.1132121, 0.0, -4.0329547, -0.0446466],
                       [-0.1812967, -0.3857987, -0.1102196, 0.1132121, 0.0, -0.0446466, -4.0329547]]
        self.assertIsNone(np.testing.assert_array_almost_equal(InitialFock, Fz))
# LOOP
Dens_p, Ep_tot, Ep_el = Dens_z, Ez_tot, Ez_el
DE = -Ez_tot
lamda = 1e-12
iteration = 0
energies = []

total_time = time.time()
while abs(DE) > lamda:
    DE = 0
    iteration += 1
    # Compute New Fock Matrix
    Fock = fock(Dens_p, Hcore, ERI, Cz.shape[0])
    # Compute the New SCF Energy
    E_el = ElEnergy(Dens_p, Hcore, Fock)
    E_tot = E_el + nre
    # Build the New Density Matrix
    Facc = OrthoF(Fock, Smin)
    Facc_val, Facc_vec = Diag(Facc)
    C = Smin @ Facc_vec
    Dens_old = Dens_p
    Dens_p = np.einsum('ik,jk->ij', C[:, :occ], C[:, :occ])
    # Test for Convergence
    print('Energy: {}\tIteration: {}'.format(E_tot, iteration))
    energies.append(E_tot)
    DE = E_el - Ep_el
    Ep_tot, Ep_el = E_tot, E_el
print('Loop time: {}s'.format(time.time() - total_time))

if __name__ == '__main__':
    unittest.main()
