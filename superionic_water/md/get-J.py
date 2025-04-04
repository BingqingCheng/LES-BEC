import numpy as np
import sys
from ase import Atoms
from ase.io import read

relative_dielectric_2cc = 3.1

xyzs = read(sys.argv[1],':')
timestep = 0.3 #fs

nframe = len(xyzs)
print('nframe =',nframe)

topo_charge = np.array([1 if a.symbol == 'H' else -2 for a in xyzs[0]])

Jt = np.zeros((nframe,4))
Jt[:,0] = np.arange(nframe) * timestep
J_topo_t = np.zeros((nframe,4))
J_topo_t[:,0] = np.arange(nframe) * timestep

for i, xyz in enumerate(xyzs):
    v = xyz.get_velocities() * 0.098
    bec = xyz.arrays['bec'].reshape(162, 3, 3) * relative_dielectric_2cc**0.5
    J = np.einsum('ijk,ij->k', bec, v)
    Jt[i,1:4] = J

    J_topo = np.einsum('i,ij->j', topo_charge, v)
    J_topo_t[i,1:4] = J_topo

np.savetxt(sys.argv[1]+'_Jt.dat', Jt, fmt='%12.6f')
np.savetxt(sys.argv[1]+'_J_topo_t.dat', J_topo_t, fmt='%12.6f')
