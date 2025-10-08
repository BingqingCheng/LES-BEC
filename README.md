# LES-BEC

This repository contains the data and scripts used in the study [*Machine learning interatomic potential can infer electrical response*](https://arxiv.org/abs/2504.05169).

Specifically, it provides the training datasets, training scripts, BEC inference scripts, 
molecular dynamics (MD) scripts with and without external electric fields, 
and trained `CACE` potentials.

## Implementation

In this work, the LES method was implemented in the [`CACE`](https://github.com/BingqingCheng/cace) code 
by adding new modules: `ewald`, `polarization`, `grad`, and `dephase`. Please check the [`cace/modules/`](https://github.com/BingqingCheng/cace/tree/main/cace/modules) for detailed implementations on `CACE`.

For a detailed explanation of the LES method, please refer to [Latent Ewald summation for machine learning of long-range interactions](https://www.nature.com/articles/s41524-025-01577-7), 
and [A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials](https://arxiv.org/abs/2507.14302), which describes the [LES library](https://github.com/ChengUCB/les).
This library is already integrated with a broad range of MLIPs, including **MACE**, **NequIP**, **Allegro**, **CACE**, and **CHGNet**.


## Example scripts
Each example folder - [`water`](water), [`superionic_water`](superionic_water), and [`PbTiO3`](ferro_PbTiO3) - 
includes its own README that explains the scripts available in that directory.

- **Training scripts** can be found in each data repository (e.g., [`water/fit`](water/fit)).  
- **BEC inference scripts** are available at [`superionic_water/eval_bec`](superionic_water/eval_bec).  
- **Example MD scripts**, with or without an external electric field, can be found in each dataset folder (e.g., [`ferro_PbTiO3/md`](ferro_PbTiO3/md)).  
- Scripts for calculating the polarization current *J(t)* are provided in [`superionic_water/md`](superionic_water/md) and [`water/md`](water/md).

## License
This project is licensed under the CC BY-NC 4.0 License.

## Citation

If you use this data in your academic work, please cite:

```text
@article{zhong2025machine,
  title={Machine learning interatomic potential can infer electrical response},
  author={Zhong, Peichen and Kim, Dongjin and King, Daniel S and Cheng, Bingqing},
  journal={arXiv preprint arXiv:2504.05169},
  year={2025}
}
```

And also consider citing:
 1. [Latent Ewald summation for machine learning of long-range interactions](https://www.nature.com/articles/s41524-025-01577-7)
    
 2. [Machine Learning of Charges and Long-Range Interactions from Energies and Forces](https://www.nature.com/articles/s41467-025-63852-x)
    
 3. [A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials](https://arxiv.org/abs/2507.14302)
