import sys
import torch
sys.path.append('../cace/')
import cace
from ase.io import read, write
import numpy as np
import pickle

to_read = './md_h2o.traj'
from ase.io import read
from ase.io import Trajectory
traj_iter = Trajectory(to_read, 'r')
print(len(traj_iter))
print('now collect polarization data')

from tqdm import tqdm

#fix the charge for O and H (-2 and 1)
total_dP_list = []
for i, atoms in tqdm(enumerate(traj_iter), total=len(traj_iter)):
    atomic_numbers = atoms.get_atomic_numbers()
    q = [1 if num==1 else (-2 if num == 8 else 0) for num in atomic_numbers]
    velocities = atoms.get_velocities()
    dP = np.array([q[i]*velocities[i] for i in range(len(q))])
    total_dP = np.sum(dP, axis=0)
    total_dP_list.append(total_dP)

    if (i+1) % 50000 == 0:
        print(f'{i+1} frames are done.')
        with open(f'bec_{i+1}.pkl', 'wb') as f:
            pickle.dump({'total_dp': np.stack(total_dP_list)}, f)

total_dP_stack = np.stack(total_dP_list)
print('save dict')

dict = {
    'total_dp': total_dP_stack
}

with open('bec_dict_fixed_q.pkl', 'wb') as f:
    pickle.dump(dict, f)
