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

calculator = CACECalculator(model_path=cace_nnp, 
                            device='cuda', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            compute_stress=False,
                           atomic_energies={1: -2.886623868115813, 8: -1.4433119340579157})

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
        dyn.atoms.write('nvt-2cc-T-'+str(temperature)+'.xyz', append=True)

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

dyn.attach(write_frame, interval=1)

dyn.attach(MDLogger(dyn, init_conf, 'nvt-2cc-T-'+str(temperature)+'.log', header=True, stress=False,
           peratom=False, mode="w"), interval=10)

# Run the MD simulation
n_steps = 400
for step in range(n_steps):
    print_energy(a=init_conf)
    dyn.run(1000)
