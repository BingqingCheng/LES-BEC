dataset: The training set for PbTiO3. 
dataset/pbe_BEC: The training and test dataset for fine-tuning and BEC evaluations. 

train_CACE_LR_EF: training script for CACE-LR model
finetune_CACE_LR_EFB_by8: fine-tuning script for CACE-LR model with pbe_BEC dataset. To start the training, please make a 2*2*2 supercell of each configuration in dataset/pbe_BEC to reduce the finite size effect. 

plot_BEC: script for plot the BEC from CACE-LR vs DFPT results.

md: molecular dynamics script for zero-filed (equi) and finite field (field) simulations

 






