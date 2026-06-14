# MD17-SchNet-Neural-Potential

Neural Network Potential (NNP) for Molecular Dynamics using SchNet and the MD17 dataset.

## Overview

This project demonstrates how to train a SchNet graph neural network to learn molecular potential energy surfaces from ab initio molecular dynamics data.

The model predicts molecular energies and atomic forces directly from atomic numbers and 3D coordinates, enabling molecular dynamics simulations at a fraction of the computational cost of Density Functional Theory (DFT).

This notebook serves as an introduction to Neural Network Potentials (NNPs), Graph Neural Networks (GNNs), and AI-driven molecular simulation.

---

## Scientific Background

Traditional molecular dynamics simulations based on DFT provide accurate energies and forces but are computationally expensive.

Neural Network Potentials learn the mapping:

Atomic Structure → Energy → Forces

allowing molecular simulations to run thousands of times faster while maintaining near-DFT accuracy.

SchNet is one of the foundational architectures for molecular machine learning and neural force fields.

---

## Dataset

MD17 Dataset

Molecules included:

* Aspirin
* Benzene
* Ethanol
* Malonaldehyde
* Naphthalene
* Salicylic Acid
* Toluene
* Uracil

Each sample contains:

* Atomic numbers
* Cartesian coordinates
* Total energy
* Atomic forces

The data were generated using ab initio molecular dynamics simulations.

---

## Model Architecture

SchNet consists of:

1. Atomic Embedding Layer
2. Continuous Filter Convolution
3. Interaction Blocks
4. Energy Prediction Head
5. Force Computation via Automatic Differentiation

The force prediction is obtained from the energy gradient:

F = −∇E

which guarantees energy conservation and physical consistency.

---

## Project Workflow

```text
MD17 Dataset
      ↓
Load Molecular Structures
      ↓
Build Neighbor Graph
      ↓
SchNet
      ↓
Predict Energy
      ↓
Autograd
      ↓
Predict Forces
      ↓
Energy + Force Loss
      ↓
Training
```

---

## Features

* MD17 Dataset Loading
* SchNet Implementation
* Energy Prediction
* Force Prediction
* Force Matching Training
* GPU Support
* Training Loss Visualization
* Model Checkpoint Saving

---

## Installation

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

---

## Running the Notebook

Open:

```text
MD17_SchNet.ipynb
```

and execute all cells.

The notebook will:

1. Download MD17
2. Create train/validation splits
3. Build a SchNet model
4. Train on molecular energies and forces
5. Plot training loss
6. Save the trained model

---

## Example Results

Typical outputs include:

* Training Loss Curve
* Validation Loss Curve
* Energy Prediction Error
* Force Prediction Error

Example training curve:

```text
Loss
│\
│ \
│  \
│   \__
│      \___
│
└──────────── Epoch
```

---

## Future Work

Planned extensions:

* PaiNN
* TorchMD-Net
* GemNet
* MACE
* Equiformer
* Neural Potentials for Reaction Pathways
* Transition State Prediction
* Molecular Dynamics with Learned Potentials

---

## Repository Structure

```text
MD17-SchNet-Neural-Potential/
│
├── MD17_SchNet.ipynb
├── README.md
├── data/
├── models/
├── results/
├── figures/
│   └── loss_curve.png
└── checkpoints/
    └── schnet_md17.pt
```

---

## Skills Demonstrated

* Machine Learning
* Deep Learning
* Graph Neural Networks
* Molecular Dynamics
* Computational Chemistry
* PyTorch
* PyTorch Geometric
* Scientific Computing
* Neural Network Potentials

---

## Author

Zhechen Wang

Background:

* Computational Chemistry
* Physical Chemistry
* Molecular Modeling
* Mass Spectrometry
* Scientific Machine Learning

Interested in AI for Molecular Science, Materials Discovery, and Neural Network Potentials.
