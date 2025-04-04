import sys
sys.path.append('../cace/')
import cace
import ase
import torch

xyz_path = sys.argv[1]
output_prefix = sys.argv[2]
batch_size = 1

cace_lr = torch.load('best_model.pth', map_location=torch.device('cpu'))
cace_representation = cace_lr.representation
q = cace_lr.output_modules[1]
polarization = cace.modules.Polarization(pbc=True, phase_key='phase') #, output_index=2)
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

cace_bec = cace.models.NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, polarization, grad, dephase],
    #output_modules=[q, polarization, grad],
)

evaluator = cace.tasks.EvaluateTask(model_path=cace_bec, device='cpu',
                                    other_keys=['q', 'bec', 'polarization'],
                                    )


test_xyz = ase.io.read(xyz_path, ':')
test_result = evaluator(test_xyz[:], batch_size=batch_size, xyz_output=output_prefix)


