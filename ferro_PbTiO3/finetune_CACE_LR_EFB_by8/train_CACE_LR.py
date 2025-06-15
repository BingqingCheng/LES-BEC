import torch
import logging
import ase.io
import cace
import pickle
import os
from cace.representations import Cace
from cace.modules import PolynomialCutoff, BesselRBF, Atomwise, Forces
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask, GetLoss
from cace.tools import Metrics, init_device, compute_average_E0s, setup_logger, get_unique_atomic_number
# from cace.tools import parse_arguments

from types import SimpleNamespace

args_dict = {'zs': None, 
        "train_path": "/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/dataset/pbe_BEC/supercell_222/train_CACE_desc.xyz",
        "val_path": "/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/dataset/pbe_BEC/supercell_222/val_CACE_desc.xyz",
        # "random_path": "/home/zhongpc/Project/cace_BEC/dataset/test_CACE_desc.xyz",
        "energy_key": "energy",
        "forces_key": "forces",
        "stress_key": "stress",
        "bec": "BEC",
        "parity": -1,
        "cutoff": 6,
        "batch_size": 2,
        "valid_batch_size": 1,
        "valid_fraction": 0.1,
        "n_rbf": 6,
        "trainable_rbf": True,
        "cutoff_fn_p": 5,
        "n_atom_basis": 3, # default 3
        "n_radial_basis": 12, # 31 for chgnet 0.3.0; 8 for chgnet 0.2.0
        "max_l": 3,
        "max_nu": 3,
        "num_message_passing": 1,
        "embed_receiver_nodes": True,
        "atomwise_layers": 3,
        "atomwise_hidden": [32, 16],
        "atomwise_residual": False,
        "atomwise_batchnorm": False,
        "atomwise_linear_nn": True,
        "lr": 2e-4, # 1e-3 for uni
        "lr_second": 0.002,
        "scheduler_factor": 0.8,
        "scheduler_patience": 3,
        "max_grad_norm": 1,
        "ema": True, # True,
        "ema_start": 5,
        "warmup_steps": 1,
        "first_phase_epochs": 20, # default 200
        "energy_loss_weight": 1000.0, 
        "force_loss_weight": 1000.0, # default 1000.0
        "bec_loss_weight": 1000.0 * 5,
        "stress_loss_weight": 0, # 1000 * 160.21, # To keep the same scale as CHGNet
        "num_restart": 5, # default 5
        "prefix": "CACE_NNP",
        "use_device": "cuda",
        "type_message_passing": ['Bchi'],
        "args_message_passing": {'Bchi': {'shared_channels': False, 'shared_l': False}},
        "lr_loss_weight": 1,
        "lr_dl": 2,
        "lr_sigma": 1.0,
        "lr_n_hidden": [24, 12],
        "lr_n_layers": 3,
        "lr_n_out": 1, # why 4 ?
        }


args = SimpleNamespace(**args_dict)

setup_logger(level='INFO', tag=args.prefix, directory='./')
device = init_device(args.use_device)

cace_nnp_noBEC = torch.load('../train_CACE_LR_EF/CACE_NNP_phase_4.pth', 
                            weights_only = False,
                            map_location= device)


if args.zs is None:
    xyz = ase.io.read(args.train_path, ':')
    args.zs = get_unique_atomic_number(xyz)

# load the avge0 dict from a file if possible
if os.path.exists('avge0.pkl'):
    with open('avge0.pkl', 'rb') as f:
        avge0 = pickle.load(f)
else:
    # Load Dataset
    avge0 = compute_average_E0s(xyz)
    with open('avge0.pkl', 'wb') as f:
        pickle.dump(avge0, f)

print(avge0)




####  Prepare Data Loaders  ####
collection_train = cace.tasks.get_dataset_from_xyz(
    train_path=args.train_path,
    valid_fraction= 0.0, # args.valid_fraction,
    data_key={'energy': args.energy_key, 'forces': args.forces_key, 'stress': args.stress_key, 'bec': 'BEC_unscreen_micro'}, # 'stress': args.stress_key},
    atomic_energies=avge0,
    cutoff=args.cutoff)

collection_val = cace.tasks.get_dataset_from_xyz(
    train_path=args.val_path,
    valid_fraction= 1.0, # args.valid_fraction,
    data_key={'energy': args.energy_key, 'forces': args.forces_key, 'stress': args.stress_key, 'bec': 'BEC_unscreen_micro'}, # 'stress': args.stress_key},
    atomic_energies=avge0,
    cutoff=args.cutoff)


train_loader = cace.tasks.load_data_loader(
    collection=collection_train,
    data_type='train',
    batch_size=args.batch_size)

val_loader = cace.tasks.load_data_loader(
    collection=collection_val,
    data_type='valid',
    batch_size=args.valid_batch_size)



#### Define new model ####
polarization = cace.modules.Polarization(pbc=True, phase_key='phase', 
                                         normalization_factor = 1./9.48933 * args.parity, )
# * np.sqrt(args.epsilon_r) 


grad = cace.modules.Grad(
    y_key = 'polarization',
    x_key = 'positions',
    output_key = 'bec_complex',
    #output_key = 'bec'
)
dephase = cace.modules.Dephase(
    input_key = 'bec_complex',
    phase_key = 'phase',
    output_key = 'CACE_bec',
    # scale_key = 'epsilon_r'
)

output_modules =  cace_nnp_noBEC.output_modules + [polarization, grad, dephase]

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

cace_nnp.to(device)


# Phase 1 Training Configuration
optimizer_args = {'lr': args.lr}
scheduler_args = {'mode': 'min', 'factor': args.scheduler_factor, 'patience': args.scheduler_patience}

energy_loss = GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn= torch.nn.MSELoss(), # torch.nn.HuberLoss(delta = 0.1),
    loss_weight=args.energy_loss_weight)

force_loss = GetLoss(
    target_name='forces', 
    predict_name='CACE_forces', 
    loss_fn= torch.nn.MSELoss(), # torch.nn.HuberLoss(delta = 0.1), 
    loss_weight=args.force_loss_weight)

# stress_loss = GetLoss(
#     target_name='stress',
#     predict_name='CACE_stress',
#     loss_fn=torch.nn.MSELoss(), # HuberLoss(delta = 0.1), 
#     loss_weight=args.stress_loss_weight)

bec_loss = cace.tasks.GetLoss(
    target_name='bec',
    predict_name='CACE_bec',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=args.bec_loss_weight
)



e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# s_metric = Metrics(
#     target_name='stress',
#     predict_name='CACE_stress',
#     name='s'
# )


bec_metric = Metrics(
    target_name='bec',
    predict_name='CACE_bec',
    name='bec'
)


print("Before training, number of model parameters: ", sum(p.numel() for p in cace_nnp.parameters()))

task = TrainingTask(
    model=cace_nnp, losses=[energy_loss, force_loss, bec_loss], 
    metrics=[e_metric, f_metric, bec_metric],
    device=device, 
    optimizer_args=optimizer_args, 
    scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_args=scheduler_args, max_grad_norm=args.max_grad_norm, ema=args.ema,
    ema_start=args.ema_start, warmup_steps=args.warmup_steps)

task.fit(train_loader, val_loader, epochs= args.first_phase_epochs, print_stride=0, 
         # verbose = 1
         )
task.save_model(args.prefix+'_finetune.pth')

print("After training, number of model parameters: ", sum(p.numel() for p in cace_nnp.parameters()))
