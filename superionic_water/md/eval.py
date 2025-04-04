import sys
sys.path.append('../cace/')
import cace
import ase
import torch

xyz_path = sys.argv[1]
output_prefix = sys.argv[2]
slice_index_1 = int(sys.argv[3])
slice_index_2 = int(sys.argv[4])
batch_size = 1

cace_lr = torch.load('best_model.pth', map_location=torch.device('cuda'))
cace_representation = cace_lr.representation
q = cace_lr.output_modules[1]
polarization = cace.modules.Polarization(pbc=True, phase_key='phase') #, output_index=2)
polarization_nopbc = cace.modules.Polarization(pbc=False, output_key='polarization_nopbc') #, output_index=2)
grad = cace.modules.Grad(
    y_key = 'polarization',
    x_key = 'positions',
    output_key = 'bec_complex',
    #output_key = 'bec'
)
dephase = cace.modules.Dephase(
    input_key = 'bec_complex',
    phase_key = 'phase',
    output_key = 'bec'
)

cace_bec = cace.models.NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, polarization, grad, dephase, polarization_nopbc],
)

evaluator = cace.tasks.EvaluateTask(model_path=cace_bec, device='cuda',
                                    other_keys=['q', 'bec', 'polarization', 'polarization_nopbc'],
                                    )


test_xyz = ase.io.read(xyz_path, index=slice(slice_index_1, slice_index_2))
test_result = evaluator(test_xyz[:], batch_size=batch_size, xyz_output=output_prefix+'-'+str(slice_index_1)+'-'+str(slice_index_2))
