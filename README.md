

# Continual Learning and Catastrophic Forgetting in Hypergraph Neural Networks

## Overview

This repository contains the implementation and evaluation framework for analyzing catastrophic forgetting in Hypergraph Neural Networks (HGNNs). The project explores how sequential task training impacts the stability of hyperedge representations and evaluates multiple Continual Learning (CL) strategies to mitigate forgetting across domain shifts.


## Task Setup

The network is trained on a sequential node classification pipeline using Reddit community graphs. The hyperedges capture user-comment interactions and structural discourse patterns.

* **Task 1:** `r/Python`
* **Task 2:** `r/JavaScript`

## Implemented Methods

The repository includes the following baseline and continual learning implementations:

* **Finetune (Baseline):** Standard sequential training with no forgetting mitigation.
* **Learning without Forgetting (LwF):** An output distillation method originally developed for Convolutional Neural Networks (CNNs). In our evaluation, LwF is highly robust and serves as the top-performing method for mitigating catastrophic forgetting in a strict two-task hypergraph sequence.
* **Experience Replay Graph Neural Network (ERGNN):** A replay-based strategy specifically designed for graph structures, buffering and replaying nodes along with their topological neighborhood to preserve structural knowledge.
* **Elastic Weight Consolidation (EWC):** Parameter regularization using Fisher Information matrix estimation.
* **HypergraphEWC:** A custom variant of EWC adapted for hyperedge parameter importance.
* **Random Replay:** Experience replay buffer storing a subset of prior task distributions.
* **Task Attention Regularization (TAR):** Attention-based penalty mechanism.

## System Requirements

* Python 3.8+
* PyTorch
* PyTorch Geometric (PyG)

## Installation

Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/yourusername/hgnn-continual-learning.git
cd hgnn-continual-learning
pip install -r requirements.txt

```

## Usage

**1. Data Preprocessing**
Generate the hypergraph structures for the two tasks:

```bash
python preprocess.py --tasks python javascript

```

**2. Training**
Run the sequential training pipeline. You can specify the continual learning method using the `--method` flag.

```bash
python train.py --method lwf --epochs 50 --learning_rate 0.001

```

**3. Evaluation**
Evaluate the model to generate the task matrix (Accuracy, Forgetting, Forward Transfer, Backward Transfer):

```bash
python evaluate.py --checkpoint models/lwf_model.pt

```

## Metrics Tracked

The evaluation script automatically calculates and logs:

* **Average Accuracy:** Overall performance across all seen tasks.
* **Catastrophic Forgetting:** The performance drop on Task 1 after training on Task 2.
* **Backward Transfer (BWT):** The influence of learning Task 2 on the retained knowledge of Task 1.
* **Forward Transfer (FWT):** The zero-shot performance improvement on Task 2 derived from Task 1 structures.

---
