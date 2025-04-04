import sys
sys.path.append('../cace/')
import cace
import torch
from cace.calculators import CACECalculator
from ase.io import read

###### variables ######
model_path = './best_model.pth'
DEVICE = 'cuda'
temperature = 300
timestep = 0.25 # to ensure to capture the fast O-H vibrations
nsteps = 200000
trajectory_file = f'md_h2o.traj'
logfile = f'md_h2o.log'
average_E0 = {1: -5.8530643373406335, 8: -2.92653216867032}

###### load model ######
cace_nnp = torch.load(model_path, map_location=DEVICE)
calculator = CACECalculator(
    model_path=cace_nnp,
    device=DEVICE,
    compute_stress=False,
    energy_key='CACE_energy',
    forces_key='CACE_forces',
    atomic_energies= average_E0
)


###### load init_config ######
init_config = read('./liquid-64.xyz', index=0)
atoms = init_config.copy()
atoms.calc = calculator

from ase.optimize import BFGS
optimizer = BFGS(atoms)
optimizer.run(fmax=0.03)

###### set NVT velocity ######
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
MaxwellBoltzmannDistribution(atoms, temperature_K= temperature )

###### set NVT ######
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.logger import MDLogger
from ase import Atoms, units
from ase.io.trajectory import Trajectory
equilibration_steps = nsteps
equil_dyn = NPT(atoms, 
                    timestep * units.fs, 
                    temperature_K= temperature,
                    ttime = 10 * units.fs,
                    pfactor=None,
                    externalstress=0.0
                        )

equil_dyn.run(300)

md_logger = MDLogger(equil_dyn, atoms, logfile=logfile, 
                     header=True, stress=False, mode='w')
equil_dyn.attach(md_logger, interval=100)
traj = Trajectory(trajectory_file, 'w', atoms)
equil_dyn.attach(traj.write, interval=1)

##optional
#from ase.io import write
#def save_xyz(atoms):
#    write('./md_out.xyz', atoms, format='extxyz', append=True)
#equil_dyn.attach(save_xyz, interval=100, atoms=atoms)
#import numpy as np
#def log_extra_info(atoms):
#    total_force = np.sum(atoms.get_forces(), axis=0)
#    velocities = atoms.get_velocities()
#    masses = atoms.get_masses().reshape(-1, 1) 
#    com_velocity = np.sum(velocities * masses, axis=0) / np.sum(masses)
#    with open("md_extra.log", "a") as f:
#        f.write(f"{equil_dyn.nsteps} {total_force[0]} {total_force[1]} {total_force[2]} {com_velocity[0]} {com_velocity[1]} {com_velocity[2]}\n")
#equil_dyn.attach(log_extra_info, interval=100, atoms=atoms)


print("Starting equilibration (NVT) ...")
equil_dyn.run(equilibration_steps)
print("complete.")
