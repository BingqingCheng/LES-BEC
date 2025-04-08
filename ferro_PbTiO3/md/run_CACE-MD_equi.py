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
from ase.io import read, write

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from types import SimpleNamespace

############# set up the atoms object #############
cuda_device = "cuda"

ref_atom = read('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/md_simulations/cubic_prim.xyz')
ref_atom = ref_atom.repeat((9,9,9))

timestep = 2
temperature = 300
steps_heat = 1000//timestep * 5
steps_sample = 1000//timestep * 200

args_dict = {"timestep": timestep,
             "temperature": temperature,
             "pressure": (1.01325)* 1e-4 + 2.8,
             "taut": 20 * timestep,
             "taup": 200 * timestep,
             "bulk_modulus": 10,
             "trajectory_sample": f'md_sample_{temperature}K.traj',
             "logfile_sample": f'md_sample_{temperature}K.log',
             "trajectory_heat": f'md_heat_{temperature}K.traj',
             "logfile_heat": f'md_heat_{temperature}K.log',
             "loginterval": 100,}

###################################################

cace_nnp_noStress = torch.load('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/train_CACE_LR_EF/CACE_NNP_phase_4.pth', weights_only = False,
                            map_location=torch.device('cuda'))
with open('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/train_CACE_LR_EF/avge0.pkl', 'rb') as f:
    avge0 = pickle.load(f)

forces = cace.modules.Forces(energy_key='CACE_energy',
                             forces_key='CACE_forces',
                             stress_key='CACE_stress',)


output_modules =  cace_nnp_noStress.output_modules[0:4] + [forces]

cace_nnp = NeuralNetworkPotential(
    representation=cace_nnp_noStress.representation,
    output_modules= output_modules,
    keep_graph=True
)

for param_noS, param_withS in zip(cace_nnp_noStress.parameters(), cace_nnp.parameters()):
    if param_noS.shape == param_withS.shape:
        param_withS.data = param_noS.data.clone()
    else:
        print(f"Skipping parameter with shape {param_noS.shape} as it does not match {param_withS.shape}")

cace_nnp = cace_nnp.to(torch.device(cuda_device))


############################################



calculator = CACECalculator(model_path=cace_nnp, 
                            device= cuda_device, 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            stress_key='CACE_stress',
                            compute_stress=True,
                            atomic_energies= avge0,
                            )
ref_atom.set_calculator(calculator)


args = SimpleNamespace(**args_dict)


ptime = args.taup * units.fs
bulk_modulus_au = args.bulk_modulus / 160.2176  # GPa to eV/A^3
compressibility_au = 1 / bulk_modulus_au


dyn_heat = NPTBerendsen(
        atoms= ref_atom,
        timestep= args.timestep * units.fs,
        temperature_K= args.temperature,
        pressure_au = args.pressure * units.GPa,
        taut= args.taut * units.fs,
        taup= args.taup * units.fs,
        compressibility_au = compressibility_au, 
        trajectory= args.trajectory_heat,
        logfile= args.logfile_heat,
        loginterval= args.loginterval,
        append_trajectory=False,
    )
dyn_heat.run(steps_heat)


####### Sample MD #######
atoms_heat = read(args.trajectory_heat, index=-1)

calculator = CACECalculator(model_path=cace_nnp, 
                            device= cuda_device, 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            stress_key='CACE_stress',
                            compute_stress=True,
                            atomic_energies= avge0,
                            )
atoms_heat.set_calculator(calculator)

dyn = NPT(
        atoms= atoms_heat,
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

