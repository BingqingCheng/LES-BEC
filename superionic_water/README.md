training_set: The original training set for superionic water is from https://github.com/BingqingCheng/superionic-water/. We randomly selected 5,000 configurations.

fit: Contains the script to train CACE-LR potential, the trained potential, and a checkpoint file.

bec_data: xyz files (from 2g/cc 2000K, 3g/cc 3000K, and 4g/cc 1000K) with BEC information. We also provide the VASP input files for computing the BECs. Note that because of the low planewave cutoff used here, the energy/force information in these xyz files are not converged, and only the BEC data are converged.

finetune_2cc: inputs and the trained model by finetuning the original model (from ./fit/) using the 2g/cc 2000K data.

md: Example MD input files for running NVT MD without external field. Scripts for computing J(t) are also provided.

md_finite_e: MD input for runing NVT under external field.

eval_bec: Example script for computing BECs for a xyz file.
 





