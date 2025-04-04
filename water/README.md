training_set: The original training set for water is originally from https://zenodo.org/records/12622341 (https://doi.org/10.48550/arXiv.2404.19674). We converted it to xyz file. H2O_BEC.xyz contains 100 configurations with energy, forces, and Born effective charges. H2O_RPBE-D3.xyz has 654 configutaions with energy and forces.

fit: Contains the script to train CACE-LR potential, the trained potential, and a checkpoint file. 4 versions. 

md: Example MD input files for running NVT MD without external field. Scripts for computing J(t) are also provided.

md_finite_e: MD input for runing NVT under external field.
