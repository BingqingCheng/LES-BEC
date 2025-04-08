cuda_device = "cuda"

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

import cace
from cace.calculators import CACECalculator
from cace.models.atomistic import NeuralNetworkPotential

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen, NVTBerendsen
from ase.md import MDLogger
from ase.io import read, write, Trajectory

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from types import SimpleNamespace

############# set up the atoms object #############
traj = Trajectory('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/md_simulations/with_EF/cubic_300K_equi/md_sample_300K.traj', 'r')
ref_atom = traj[500]

timestep = 2
temperature = 300
steps_heat = 2000
steps_sample = 1000 * 100

args_dict = {"timestep": timestep,
             "temperature": temperature,
             "pressure": (1.01325)* 1e-4 + 2.8,
             "taut": 20 * timestep,
             "taup": 200 * timestep,
             "bulk_modulus": 10,
             "external_field": [2000, 0, 0], # in kV/cm 
             "electric_field_unit": 1e-5, # convert kV/cm to eV/A 
             "keep_neutral": True,
             "trajectory_sample": f'md_sample_{temperature}K.traj',
             "logfile_sample": f'md_sample_{temperature}K.log',
             "trajectory_heat": f'md_heat_{temperature}K.traj',
             "logfile_heat": f'md_heat_{temperature}K.log',
             "loginterval": 50,}

args = SimpleNamespace(**args_dict)

#################### Load BEC model ###############################
cace_nnp_noBEC = torch.load('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/train_CACE_LR_EF/CACE_NNP_phase_4.pth', 
                             weights_only = False, map_location=torch.device(cuda_device))
# cace_nnp_noBEC = cace_nnp.to(torch.device(cuda_device))

with open('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/train_CACE_LR_EF/avge0.pkl', 'rb') as f:
    avge0 = pickle.load(f)

forces = cace.modules.Forces(energy_key='CACE_energy',
                             forces_key='CACE_forces',
                             stress_key='CACE_stress',)

polarization = cace.modules.Polarization(pbc=True, phase_key='phase') #, output_index=2)

grad = cace.modules.Grad(
    y_key = 'polarization',
    x_key = 'positions',
    output_key = 'bec_complex',
    #output_key = 'bec'
)
dephase = cace.modules.Dephase(
    input_key = 'bec_complex',
    phase_key = 'phase',
    output_key = 'CACE_bec'
)

output_modules =  cace_nnp_noBEC.output_modules[0:4] + [forces, polarization, grad, dephase]

cace_nnp = NeuralNetworkPotential(
    representation=cace_nnp_noBEC.representation,
    output_modules= output_modules,
    keep_graph=True
)

# Copy weights and biases from cace_nnp_noBEC to cace_nnp_withBEC
for param_noBEC, param_withBEC in zip(cace_nnp_noBEC.parameters(), cace_nnp.parameters()):
    if param_noBEC.shape == param_withBEC.shape:
        param_withBEC.data = param_noBEC.data.clone()
    else:
        print(f"Skipping parameter with shape {param_noBEC.shape} as it does not match {param_withBEC.shape}")

cace_nnp = cace_nnp.to(torch.device(cuda_device))

#################### Load BEC model ###############################




calculator = CACECalculator(model_path=cace_nnp, 
                            device= cuda_device, 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            stress_key='CACE_stress',
                            bec_key = 'CACE_bec',
                            external_field= args.external_field, # [0,0,e_field]
                            electric_field_unit = args.electric_field_unit,
                            keep_neutral = args.keep_neutral,
                            compute_stress=True,
                            atomic_energies= avge0,
                            )
ref_atom.set_calculator(calculator)



ptime = args.taup * units.fs
bulk_modulus_au = args.bulk_modulus / 160.2176  # GPa to eV/A^3
compressibility_au = 1 / bulk_modulus_au


####### Sample MD #######

dyn = NPT(
        atoms= ref_atom,
        timestep= args.timestep * units.fs,
        temperature_K= args.temperature,
        externalstress= args.pressure * units.GPa,  # ase NPT does not like externalstress=None
        ttime= args.taut * units.fs,
        pfactor= args.bulk_modulus * units.GPa * ptime * ptime , # None for NVT simulations
        trajectory= args.trajectory_sample,
        logfile= args.logfile_sample,
        loginterval= args.loginterval,
        append_trajectory=False,
        mask = (1, 1, 1),
    )

dyn.run(steps_sample)

