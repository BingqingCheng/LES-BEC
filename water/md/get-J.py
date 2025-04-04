import sys
import torch
sys.path.append('../cace/')
import cace
from ase.io import read, write
import numpy as np
import pickle

root = '../best_model.pth' #use the same model that you used for MD
to_read = './md_h2o.traj'

DEVICE='cuda'
import torch
cace_nnp = torch.load(root, map_location=DEVICE)
print([module.__class__.__name__ for module in cace_nnp.output_modules])

cace_representation = cace_nnp.representation
q = cace_nnp.output_modules[1]
polarization = cace.modules.Polarization(pbc=True, normalization_factor = 1./9.48933 * 1.333)

grad  = cace.modules.Grad(
    y_key = 'polarization',
    x_key = 'positions',
    output_key = 'bec_complex'
)
dephase = cace.modules.Dephase(
    input_key = 'bec_complex',
    phase_key = 'phase',
    output_key = 'CACE_bec'
)
cace_bec = cace.models.NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, polarization, grad, dephase],
)

## if model already has all modules in it, this would be enough
#cace_nnp.output_modules[5].normalization_factor = 1./9.48933 *1.333
#print(cace_nnp.output_modules[5].normalization_factor)
#cace_representation = cace_nnp.representation
#cace_bec=cace_nnp

from ase.io import read
from ase.io import Trajectory
traj_iter = Trajectory(to_read, 'r')
print(len(traj_iter))
print('now collect polarization data')

from cace.tools import torch_geometric
from cace.data import AtomicData
from tqdm import tqdm 
import gc

DEVICE = 'cuda'

total_dP_list = []
for i, atoms in tqdm(enumerate(traj_iter), total=len(traj_iter)):
    atomic_data = AtomicData.from_atoms(atoms, cutoff=cace_representation.cutoff).to(DEVICE)
    data_loader = torch_geometric.dataloader.DataLoader(
                    dataset=[atomic_data],
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )
    batch = next(iter(data_loader)).to(DEVICE)
    batch = batch.to_dict()
    output = cace_bec(batch)
    BEC = output['CACE_bec'].squeeze(0)
    velocity = torch.tensor(atoms.get_velocities(), dtype=BEC.dtype, device=DEVICE)
    dP = torch.bmm(BEC, velocity.unsqueeze(-1)).squeeze(-1)
    total_dP = torch.sum(dP, dim=0)

    total_dP_list.append(total_dP.detach().cpu())

    del atomic_data, data_loader, batch, output, BEC, velocity, dP, total_dP
    torch.cuda.empty_cache()
    gc.collect()

    if (i+1) % 100000 == 0:
        print(f'{i+1} frames are done.')
        with open(f'bec_{i+1}.pkl', 'wb') as f:
            pickle.dump({'total_dp': torch.stack(total_dP_list).numpy()}, f)

total_dP_stack = np.array(torch.stack(total_dP_list))
print('save dict')

dict = {
    'total_dp': total_dP_stack
}

with open('bec_dict.pkl', 'wb') as f:
    pickle.dump(dict, f)
