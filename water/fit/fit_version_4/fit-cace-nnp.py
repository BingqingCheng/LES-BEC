#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../cace/')

import numpy as np
import torch
import torch.nn as nn
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 4.5

logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='../data/H2O_BEC.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces', 'bec': 'BEC_ref'}, 
                                 atomic_energies={1: -5.868579157459375, 8: -2.9342895787296874}
                                 )
batch_size = 2

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=4,
                              )

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1,8],
    n_atom_basis=2,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    #avg_num_neighbors=1,
    device=device,
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

sr_energy = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='SR_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)



q = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=1,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.EwaldPotential(dl=2,
                    sigma=1.0,
                    feature_key='q',
                    output_key='ewald_potential',
                    remove_self_interaction=False,
                   aggregation_mode='sum')

e_add = cace.modules.FeatureAdd(feature_keys=['SR_energy', 'ewald_potential'],
                 output_key='CACE_energy')

forces = cace.modules.Forces(energy_key='CACE_energy',
                             forces_key='CACE_forces')

polarization = cace.modules.Polarization(pbc=True, phase_key='phase',
                                         normalization_factor = 1.333/9.48933
                                         ) #, output_index=2)
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

cace_nnp = NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[sr_energy, q, ep, e_add, forces, polarization, grad, dephase],
    keep_graph=True
)

cace_nnp.to(device)


logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

bec_loss = cace.tasks.GetLoss(
    target_name='bec',
    predict_name='CACE_bec',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10
)

from cace.tools import Metrics

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

bec_metric = Metrics(
    target_name='bec',
    predict_name='CACE_bec',
    name='bec'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss, bec_loss],
        metrics=[e_metric, f_metric, bec_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False, #True,
        ema_start=10,
        warmup_steps=5,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False, print_stride=0)

task.save_model('water-model.pth')
cace_nnp.to(device)

logging.info(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

bec_loss = cace.tasks.GetLoss(
    target_name='bec',
    predict_name='CACE_bec',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=100
)

task.update_loss([energy_loss, force_loss, bec_loss])
logging.info("training")
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, print_stride=0)


task.save_model('water-model-2.pth')
cace_nnp.to(device)

logging.info(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 
)

bec_loss = cace.tasks.GetLoss(
    target_name='bec',
    predict_name='CACE_bec',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss, bec_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, print_stride=0)

task.save_model('water-model-3.pth')

logging.info(f"Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss, bec_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, print_stride=0)

task.save_model('water-model-4.pth')

logging.info(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")



