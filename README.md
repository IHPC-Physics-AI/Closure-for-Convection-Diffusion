# Closure-for-Convection-Diffusion

## Repository Structure

`dataset` contains information generated from CFD software (Fluent), such as the dataset and generated turbulent diffusivity values. Information regarding dataset is contained inside `expt1_baseline.ipynb`.

`numerical_methods` contains helper functions used to compute advection and diffusion terms using the finite volume approach.

`expt1_baseline.py`, `expt2_datadriven.py`, `expt3_markovian.py` are training scripts for the different experiments.

To generate videos, first generate pickle files using `generate_video_pickle.ipynb`, then generate videos from the pickle files using `generate_video.ipynb`.

## Abstract

Computational Fluid Dynamics (CFD) simulations are often used to solve and understand complicated fluid flow problems. However, while the simulation needs to be carried out in a 3D domain, only information on a 2D plane-slice may be necessary. Therefore, to reduce computational cost, a physics-based surrogate model is developed to simulate passive scalar transport on a coarse 2D plane-slice subset of the 3D domain. The physics is accounted for by solving the convection-diffusion equation numerically. However, this is incomplete as unresolved processes such as flow normal to the plane-slice are not accounted for. Therefore, a deep learning approach involving a memory unit is used to compute the non-Markovian closure term. The results show that the training process helps generalize to unseen test data, and qualitatively, the simulation is consistent with our understanding of the physical system. 

<img width="590" height="256" alt="convection_diffusion drawio" src="https://github.com/user-attachments/assets/7a6ee47b-37a8-477f-a0ad-6dcbb2ad07be" />

## Experiments

- Experiment 1: Baseline approach
- Experiment 2: Data-driven approach by not computing the physical terms (diffusion, advection, source)
- Experiment 3: Markovian approach by removing the memory unit
