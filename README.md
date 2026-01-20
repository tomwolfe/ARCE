# ARCE: Automated Renormalization & Coarse-Graining Engine

ARCE is a high-caliber intersection of **Information Theory**, **Statistical Mechanics**, and **Geometric Deep Learning**. It automates the discovery of "Macro-states" from "Micro-states" by optimizing for **Causal Emergence**—a concept where a coarse-grained description of a system is more predictive than the system itself.

## Core Innovations

- **Differentiable Renormalization Group (RG):** Uses `IterativeDecimator` as a learnable $R_s$ operator with Gumbel-Softmax for differentiable graph partitioning.
- **Information-Theoretic Loss:** Employs soft-histograms and Gaussian-kernel-based EI (Effective Information) estimation to push the system toward higher predictive power.
- **Embedded Symbolic Regression:** Integrates ISTA (Iterative Soft-Thresholding Algorithm) in JAX for sparse identification of dynamics (SINDy-style) during representation learning.

## Project Structure

- `main.py`: Entry point for training and demonstration.
- `arce/`: Core library.
    - `models.py`: GNN and Hierarchical Pooling (MSVIB) architectures.
    - `engine.py`: Training logic, multi-task loss weighting, and symbolic discovery.
    - `metrics.py`: Implementation of EI, Causal Emergence, and IB losses.
    - `utils.py`: Data generation (Ising model), graph utilities, and Basis Library.
- `tests/`: Unit tests for various components.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running ARCE

```bash
python main.py
```

## Physical Grounding

The engine is currently demonstrated on 2D Ising lattices. It learns to coarse-grain spin configurations into macro-variables that follow simple, discoverable physical laws.

## References

- Hoel, E. P. (2017). When the Map Is Better Than the Territory. *Entropy*.
- Iten, R., et al. (2020). Discovering Physical Concepts with Neural Networks. *Physical Review Letters*.
