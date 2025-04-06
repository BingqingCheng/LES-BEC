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
        "train_path": "/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/dataset/train_CACE_desc.xyz",
        "val_path": "/home/zhongpc/Project/cace_BEC/ferro_PbTiO3/dataset/val_CACE_desc.xyz",
        "energy_key": "energy",
        "forces_key": "forces",
        "stress_key": "stress",
        "bec": "BEC",
        "cutoff": 6,
        "batch_size": 4,
        "valid_batch_size": 16,
        "valid_fraction": 0.1,
        "n_rbf": 6,
        "trainable_rbf": True,
        "cutoff_fn_p": 5,
        "n_atom_basis": 3, 
        "n_radial_basis": 12,
        "max_l": 3,
        "max_nu": 3,
        "num_message_passing": 1,
        "embed_receiver_nodes": True,
        "atomwise_layers": 3,
        "atomwise_hidden": [32, 16],
        "atomwise_residual": False,
        "atomwise_batchnorm": False,
        "atomwise_linear_nn": True,
        "lr": 0.01, 
        "lr_second": 0.001,
        "scheduler_factor": 0.8,
        "scheduler_patience": 3,
        "max_grad_norm": 1,
        "ema": True, # True,
        "ema_start": 5,
        "warmup_steps": 1,
        "first_phase_epochs": 100, # default 200
        "second_phase_epochs": 30, # default 100
        "energy_loss_weight": 0.1,
        "force_loss_weight": 1000.0, # default 1000.0
        "second_phase_energy_loss_weight": 1000.0,
        "second_phase_force_loss_weight": 1000.0,
        "second_phase_bec_loss_weight": 100,
        "stress_loss_weight": 0, 
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
        "lr_n_out": 1, 
        }


args = SimpleNamespace(**args_dict)

setup_logger(level='INFO', tag=args.prefix, directory='./')
device = init_device(args.use_device)

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


# Prepare Data Loaders
collection_train = cace.tasks.get_dataset_from_xyz(
    train_path=args.train_path,
    valid_fraction= 0.0, # args.valid_fraction,
    data_key={'energy': args.energy_key, 'forces': args.forces_key},  # 'bec': 'BEC'}, # 'stress': args.stress_key},
    atomic_energies=avge0,
    cutoff=args.cutoff)

collection_val = cace.tasks.get_dataset_from_xyz(
    train_path=args.val_path,
    valid_fraction= 1.0, # args.valid_fraction,
    data_key={'energy': args.energy_key, 'forces': args.forces_key},  # 'bec': 'BEC'}, # 'stress': args.stress_key},
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




# Configure CACE Representation

cutoff_fn = PolynomialCutoff(cutoff=args.cutoff, p=args.cutoff_fn_p)
radial_basis = BesselRBF(cutoff=args.cutoff, n_rbf=args.n_rbf, trainable=args.trainable_rbf)
cace_representation = Cace(
    zs=args.zs, n_atom_basis=args.n_atom_basis, embed_receiver_nodes=args.embed_receiver_nodes,
    cutoff=args.cutoff, cutoff_fn=cutoff_fn, radial_basis=radial_basis,
    n_radial_basis=args.n_radial_basis, max_l=args.max_l, max_nu=args.max_nu,
    device=device, num_message_passing=args.num_message_passing, 
    # node_feature_dim= args.node_feature_dim, type_output_feature = args.type_output_feature
    )

logging.info(f"Representation: {cace_representation}")

# Configure Atomwise Module
sr_energy = cace.modules.Atomwise(
            n_layers=args.atomwise_layers, n_hidden=args.atomwise_hidden, residual=args.atomwise_residual,
            use_batchnorm=args.atomwise_batchnorm, add_linear_nn=args.atomwise_linear_nn,
            output_key='SR_energy')


q = cace.modules.Atomwise(
    n_layers = args.lr_n_layers,
    n_hidden = args.lr_n_hidden,
    n_out = args.lr_n_out , # why default is 4?
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.EwaldPotential(dl = args.lr_dl,
                    sigma= args.lr_sigma,
                    feature_key='q',
                    output_key='ewald_potential',
                    remove_self_interaction=False,
                    aggregation_mode='sum')

e_add = cace.modules.FeatureAdd(feature_keys=['SR_energy', 'ewald_potential'],
                 output_key='CACE_energy')

forces = cace.modules.Forces(energy_key='CACE_energy',
                             forces_key='CACE_forces')


cace_nnp = NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[sr_energy, q, ep, e_add, forces], # , polarization, grad, dephase],
    keep_graph=True
)

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

# bec_loss = cace.tasks.GetLoss(
#     target_name='bec',
#     predict_name='CACE_bec',
#     loss_fn=torch.nn.MSELoss(),
#     loss_weight=10
# )


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

# bec_metric = Metrics(
#     target_name='bec',
#     predict_name='CACE_bec',
#     name='bec'
# )


# s_metric = Metrics(
#     target_name='stress',
#     predict_name='CACE_stress',
#     name='f'
# )

print("Before training, number of model parameters: ", sum(p.numel() for p in cace_nnp.parameters()))

# Phase 1 Training
for _ in range(args.num_restart):
   # Initialize and Fit Training Task for Phase 1
    task = TrainingTask(
        model=cace_nnp, losses=[energy_loss, force_loss], # , bec_loss], # stress_loss], 
        metrics=[e_metric, f_metric], #, bec_metric],
        device=device, 
        optimizer_args=optimizer_args, 
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=scheduler_args, max_grad_norm=args.max_grad_norm, ema=args.ema,
        ema_start=args.ema_start, warmup_steps=args.warmup_steps)

    task.fit(train_loader, val_loader, epochs=int(args.first_phase_epochs/args.num_restart), print_stride=0, 
             # verbose = 1
             )
task.save_model(args.prefix+'_phase_1.pth')

print("After training, number of model parameters: ", sum(p.numel() for p in cace_nnp.parameters()))


logging.info(f"********** Start phase 2 training **********")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 * args.energy_loss_weight,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, val_loader, epochs= args.second_phase_epochs)
task.save_model(args.prefix+'_phase_2.pth')


logging.info(f"********** Start phase 3 training **********")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=100 * args.energy_loss_weight,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, val_loader, epochs= args.second_phase_epochs)
task.save_model(args.prefix+'_phase_3.pth')


logging.info(f"********** Start phase 4 training **********")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10000 * args.energy_loss_weight,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, val_loader, epochs= args.second_phase_epochs)
task.save_model(args.prefix+'_phase_4.pth')

