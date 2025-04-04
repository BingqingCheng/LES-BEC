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
cutoff = 3.5

logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='S-2cc-T2000K-BEC.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces', 'bec':'BEC'}, 
                                 atomic_energies={1: -2.886623868115813, 8: -1.4433119340579157} # avg
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

cace_nnp = torch.load('old_model.pth')

polarization = cace.modules.Polarization(pbc=True, phase_key='phase',normalization_factor = -3.1**0.5/9.48933) #, output_index=2)
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

cace_nnp.add_module(polarization)
cace_nnp.add_module(grad)
cace_nnp.add_module(dephase)

cace_nnp.keep_graph = True
cace_nnp.to(device)



logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=100
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
    loss_weight=3000
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

optimizer_args = {'lr': 5e-3, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

for i in range(1):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss, bec_loss],
        metrics=[e_metric, f_metric, bec_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False,
        ema_start=10,
        warmup_steps=5,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=200, screen_nan=False)

exit()

task.save_model('water-model.pth')
cace_nnp.to(device)

logging.info(f"Second train loop:")
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
    loss_weight=100000
)

task.update_loss([energy_loss, force_loss])
logging.info("training")
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)


task.save_model('water-model-2.pth')
cace_nnp.to(device)

logging.info(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000 
)
bec_loss = cace.tasks.GetLoss(
    target_name='bec',
    predict_name='CACE_bec',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000000
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

task.save_model('water-model-3.pth')
logging.info(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")



