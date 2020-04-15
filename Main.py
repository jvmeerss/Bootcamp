import numpy as np
import psi4
from HFSCF import Diag, OrthoS, OrthoF, Fock, ElEnergy, Unitary, Hermitian, EHF
import time
import matplotlib.pyplot as plt

psi4.core.set_output_file('output.dat', True)
psi4.set_memory(int(5e8))
np_memory = 2

psi4.set_options({'basis': 'sto-3g'})

mol = psi4.geometry("""
H   0.0 0.0 0.0
H   0.0 0.0 {R}
units angstrom
""".format(R=1))
mol.update_geometry()

nre = mol.nuclear_repulsion_energy()
wave = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wave.basisset())
occ_a = wave.nalpha()
occ_b = wave.nbeta()

S = np.asarray(mints.ao_overlap())
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
ERI = np.asarray(mints.ao_eri())
Hcore = T + V
Sval, Svec = Diag(S)
Smin = OrthoS(Sval, Svec)
Fa = Fb = OrthoF(Hcore, Smin)
Fa_val, Fa_vec = Fb_val, Fb_vec = Diag(Fa)
Ca = Cb = Smin @ Fa_vec

Da = np.einsum('ik,jk->ij',Ca[:, :occ_a], Ca[:, :occ_a])
Db = np.einsum('ik,jk->ij',Cb[:, :occ_b], Cb[:, :occ_b])

Eza_el = np.einsum('ij,ij->', Da, 2*Hcore)
Eza_tot = Eza_el + nre
Ezb_el = np.einsum('ij,ij->', Db, 2*Hcore)
Ezb_tot = Ezb_el + nre

ehf = 0
DE = -Eza_tot
val = 1e-12
iteration = 0
energiesa = []
energiesb = []
energieshf = []
iterations  =[]
total_time = time.time()

while DE > val:
    DE = 0
    iteration += 1
    iterations.append(iteration)
    Fa = Fock(Da, Hcore, ERI, Ca.shape[0])
    Fb = Fock(Db, Hcore, ERI, Cb.shape[0])
    Ea_el = ElEnergy(Da, Hcore, Fa)
    Eb_el = ElEnergy(Db, Hcore, Fb)
    Ea_tot = Ea_el + nre
    Eb_tot = Eb_el + nre

    Fanew = OrthoF(Fa, Smin)
    Fbnew = OrthoF(Fb, Smin)
    Fanew_val, Fanew_vec = Diag(Fanew)
    Fbnew_val, Fbnew_vec = Diag(Fbnew)
    Ca = Smin @ Fanew_vec
    Cb = Smin @ Fbnew_vec
    ehf_new = EHF(Hcore, Da, Fa, Db, Fb)
    Densa_old = Da
    Densb_old = Db
    Da = np.einsum('ik,jk->ij', Ca[:, :occ_a], Ca[:, :occ_a])
    Db = np.einsum('ik,jk->ij', Cb[:, :occ_b], Cb[:, :occ_b])
    energiesa.append(Ea_tot)
    energiesb.append(Eb_tot)
    DEa = abs(Ea_el - Eza_el)
    DEb = abs(Eb_el - Ezb_el)
    energieshf.append(ehf_new)
    DE = abs(ehf - ehf_new)
    print('HF_Energy: {}\tIteration: {}\tDE: {}'.format(ehf_new, iteration, DE))
    Eza_el = Ea_el
    Ezb_el = Eb_el
    ehf = ehf_new

print('Loop time: {}s'.format(time.time() - total_time))
print(energieshf[-1]+nre)
print(iterations)

