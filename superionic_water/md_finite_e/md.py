import sys
sys.path.append('../cace/')
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

import cace
from cace.representations.cace_representation import Cace
from cace.calculators import CACECalculator

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

init_conf = read('./init-2gcc.xyz', '0')
cace_nnp = torch.load('best_model.pth') #,map_location=torch.device('cpu'))

cace_nnp.remove_module(-1)
cace_nnp.remove_module(-1)
cace_nnp.remove_module(-1)

polarization = cace.modules.Polarization(pbc=True, phase_key='phase', output_index=2, normalization_factor = -3.1**0.5/9.48933)
grad = cace.modules.Grad(
    y_key = 'polarization',
    x_key = 'positions',
    output_key = 'bec_complex',
)
dephase = cace.modules.Dephase(
    input_key = 'bec_complex',
    phase_key = 'phase',
    output_key = 'bec'
)

cace_nnp.add_module(polarization)
cace_nnp.add_module(grad)
cace_nnp.add_module(dephase)

e_field = float(sys.argv[1]) # in eV/A

calculator = CACECalculator(model_path=cace_nnp, 
                            device='cuda', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            bec_key='bec',
                            compute_stress=False,
                           atomic_energies={1: -2.886623868115813, 8: -1.4433119340579157},
                            external_field=e_field)

init_conf.set_calculator(calculator)

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

temperature = 2000 #float(sys.argv[2])# in K

# Set initial velocities using Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(init_conf, temperature * units.kB)


def print_energy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)  '
          'Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_frame():
        dyn.atoms.write('nvt-2cc-T-'+str(temperature)+'-E-'+str(e_field)+'.xyz', append=True)

# Define the NPT ensemble
NPTdamping_timescale = 100 * units.fs  # Time constant for NPT dynamics
NVTdamping_timescale = 10 * units.fs  # Time constant for NVT dynamics (NPT includes both)
dyn = NPT(init_conf, timestep=0.3 * units.fs, temperature_K=temperature,
          ttime=NVTdamping_timescale, pfactor=None, #0.1*NPTdamping_timescale**2,
          externalstress=0.0)

# equilibrate
n_steps = 100
for step in range(n_steps):
    print_energy(a=init_conf)
    dyn.run(100)

dyn.attach(write_frame, interval=100)

dyn.attach(MDLogger(dyn, init_conf, 'nvt-2cc-T-'+str(temperature)+'-E-'+str(e_field)+'.log', header=True, stress=False,
           peratom=False, mode="w"), interval=10)

# Run the MD simulation
n_steps = 100
for step in range(n_steps):
    print_energy(a=init_conf)
    dyn.run(1000)
