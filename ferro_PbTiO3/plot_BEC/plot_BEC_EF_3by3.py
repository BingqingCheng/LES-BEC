# %%
import sys
import cace
from cace.models.atomistic import NeuralNetworkPotential
import ase
import torch
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import os


cuda_device = "cuda"

def get_element_indices(atoms):
    Ti_indices = []
    Pb_indices = []
    O_indices = []
    for atom in atoms:
        if 'Pb' in atom.symbol:
            Pb_indices.append(atom.index)
        elif 'Ti' in atom.symbol:
            Ti_indices.append(atom.index)
        elif 'O' in atom.symbol:
            O_indices.append(atom.index)

    Ti_indices = np.array(Ti_indices)
    Pb_indices = np.array(Pb_indices)
    O_indices = np.array(O_indices)

    return Pb_indices, Ti_indices, O_indices


cace_nnp_noBEC = torch.load('/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/train_CACE_LR_EF/CACE_NNP_phase_4.pth', weights_only = False, map_location=torch.device('cuda'))

# %%
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

output_modules =  cace_nnp_noBEC.output_modules + [polarization, grad, dephase]

cace_nnp_withBEC = NeuralNetworkPotential(
    representation=cace_nnp_noBEC.representation,
    output_modules= output_modules,
    keep_graph=True
)

# %%


# %%
# Copy weights and biases from cace_nnp_noBEC to cace_nnp_withBEC
for param_noBEC, param_withBEC in zip(cace_nnp_noBEC.parameters(), cace_nnp_withBEC.parameters()):
    if param_noBEC.shape == param_withBEC.shape:
        param_withBEC.data = param_noBEC.data.clone()
    else:
        print(f"Skipping parameter with shape {param_noBEC.shape} as it does not match {param_withBEC.shape}")

# %%
cace_nnp_withBEC = cace_nnp_withBEC.to(cuda_device)

# %%


# %%
cutoff = 6.0

xyz_path = "/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/dataset/pbe_BEC/test_CACE_desc.xyz"
test_xyz = ase.io.read(xyz_path, ':')
test_xyz = [atoms.repeat((3,3,3)) for atoms in test_xyz]


# %%
dataset = [cace.data.AtomicData.from_atoms(atoms, cutoff=cutoff) for atoms in test_xyz]
data_loader = cace.tools.torch_geometric.dataloader.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False
)



# %%
bec_input_all = np.array([]).reshape(0, 9)
bec_input_DFT_all = np.array([]).reshape(0, 9)

bec_output_all = np.array([]).reshape(0, 9)
bec_output_scale_all = np.array([]).reshape(0, 9)

Pb_bec_input_DFT_all = np.array([]).reshape(0, 9)
Pb_bec_output_scale_all = np.array([]).reshape(0, 9)

Ti_bec_input_DFT_all = np.array([]).reshape(0, 9)
Ti_bec_output_scale_all = np.array([]).reshape(0, 9)

O_bec_input_DFT_all = np.array([]).reshape(0, 9)
O_bec_output_scale_all = np.array([]).reshape(0, 9)


atoms_to_save = []

for ii, batch in enumerate(data_loader):
    print(f"Processing batch {ii+1}/{len(data_loader)}")

    batch = batch.to(torch.device(cuda_device))
    output = cace_nnp_withBEC(batch)

    atoms_input = test_xyz[ii]
    bec_micro = atoms_input.get_array('BEC_unscreen_micro').reshape(-1, 9)
    bec_macro = atoms_input.get_array('BEC_unscreen_macro').reshape(-1, 9)
    # bec_DFT = atoms_input.get_array('BEC').reshape(-1, 9)
    epsilon_r_micro = atoms_input.info['epsilon_r_micro']
    epsilon_r_macro = atoms_input.info['epsilon_r_macro']

    # bec_input = output.get('bec_complex').cpu().detach().numpy() #[ batch, 3, 3]

    bec_output = output.get('CACE_bec').cpu().detach().numpy() #[ batch, 3, 3]
    bec_output= bec_output.reshape(-1, 9) # scale to macroscopic BEC

    # Clear CUDA memory of output
    del output
    torch.cuda.empty_cache()
    
    directions = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    # for ii, direction in enumerate(directions):
    #     bec_component = bec_output[:,ii]
    #     atoms_input.set_array('BEC_' + direction, bec_component)

    atoms_to_save.append(atoms_input)

    bec_output_all = np.vstack((bec_output_all, bec_output))
    bec_input_all = np.vstack((bec_input_all, bec_micro))  # scale to macroscopic BEC
    
    bec_input_DFT_all = np.vstack((bec_input_DFT_all, bec_micro * np.sqrt(epsilon_r_micro)))  
    bec_output_scale_all = np.vstack((bec_output_scale_all, bec_output * np.sqrt(epsilon_r_micro)))

    Pb_indices, Ti_indices, O_indices = get_element_indices(atoms_input)
    
    Pb_bec_input_DFT_all = np.vstack((Pb_bec_input_DFT_all, bec_micro[Pb_indices] * np.sqrt(epsilon_r_micro)))
    Pb_bec_output_scale_all = np.vstack((Pb_bec_output_scale_all, bec_output[Pb_indices] * np.sqrt(epsilon_r_micro)))

    Ti_bec_input_DFT_all = np.vstack((Ti_bec_input_DFT_all, bec_micro[Ti_indices] * np.sqrt(epsilon_r_micro)))
    Ti_bec_output_scale_all = np.vstack((Ti_bec_output_scale_all, bec_output[Ti_indices] * np.sqrt(epsilon_r_micro)))

    O_bec_input_DFT_all = np.vstack((O_bec_input_DFT_all, bec_micro[O_indices] * np.sqrt(epsilon_r_micro)))
    O_bec_output_scale_all = np.vstack((O_bec_output_scale_all, bec_output[O_indices] * np.sqrt(epsilon_r_micro)))

# %%
# atoms_input.info['epsilon_r_micro']

# %%
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

parity = -1 

directions = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']

for index in range(9):
    row = index // 3
    col = index % 3
    axes[row, col].plot(Pb_bec_input_DFT_all[:, index], parity * Pb_bec_output_scale_all[:, index], 
                        'o', alpha = 0.35, color = 'dimgray')

    axes[row, col].plot(Ti_bec_input_DFT_all[:, index], parity * Ti_bec_output_scale_all[:, index],
                        'o', alpha = 0.35, color = 'deepskyblue')
    
    axes[row, col].plot(O_bec_input_DFT_all[:, index], parity * O_bec_output_scale_all[:, index],
                        'o', alpha = 0.35, color = 'red')

    # axes[row, col].plot(finetune_bec_input_all[:, index], finetune_bec_output_all[:, index] * np.sqrt(EPSILON), 'x', alpha=0.1)

    axes[row, col].plot([-10, 10], [-10, 10], '--', color='gray', alpha=0.5)
    axes[row, col].set_title(f'{directions[index]}', fontsize= 20)
    axes[row, col].set_xlim(-10, 10)
    axes[row, col].set_ylim(-10, 10)

    # ticks = [-1.5, -1, -0.5, 0, +0.5, +1, 1.5]
    # axes[row, col].set_xticks(ticks, ticks, fontsize = 15)
    # axes[row, col].set_yticks(ticks, ticks, fontsize = 15)



fig.text(0.5, -0.03, 'DFT BEC [e]', ha='center', fontsize=20)
fig.text(-0.03, 0.5, 'CACE BEC [e]', va='center', rotation='vertical', fontsize=20)

plt.tight_layout()

plt.savefig('BEC_EFS_CACE_vs_DFT.png')