# Optimization of Physics-Informed Neural Networks (PINNs) for the Relativistic Klein-Gordon PDE

This repository implements an optimization pipeline for solving the homogeneous Relativistic Klein-Gordon Partial Differential Equation (PDE) using Physics-Informed Neural Networks (PINNs). By combining custom domain normalization, Sinusoidal Representation Networks (SIREN), dynamic gradient annealing, and a two-stage hybrid optimization scheme, this project resolves stiffness and gradient flow pathologies common in physics-constrained deep learning. A high-order pseudospectral solver utilizing matrix exponential propagators serves as the exact numerical ground truth for benchmarking.

---

## Key Features & Methodological Highlights

* **SIREN Neural Architecture**: Utilizes a 6-layer, 128-neuron deep MLP with sinusoidal activation functions ($\omega_0 = 30$) and variance-preserving initialization. This architecture overcomes spectral bias and models high-frequency wave dispersion.
* **High-Precision Numerical Baseline**: Solves the 1D/2D Klein-Gordon wave dynamics using a pseudospectral method combined with exact matrix exponential propagators in Fourier space under CFL stability constraints.
* **Domain & Operator Normalization**: Implements min-max variable scaling to $[-1, 1]$ alongside exact chain-rule derivative operator scaling ($\alpha_x, \alpha_t$) to prevent vanishing gradients across operator scales.
* **Memory-Efficient Dynamic Resampling**: Dynamically resamples 128 collocation points per loss term per epoch (evaluating over $6.4 \times 10^6$ spacetime points across training) to prevent overfitting without saturating RAM.

---

## Optimization Breakdown

### The Failure Mode of Vanilla PINNs
Standard PINNs optimized with standard ADAM or unweighted SGD frequently converge to invalid trivial solutions or stall at apparent Pareto fronts. Analysis via Karush-Kuhn-Tucker (KKT) conditions and Sobolev inequalities reveals that the Hessian spectral radius of the physical residual $\nabla_\theta^2 \mathcal{L}_f$ dominates the loss landscape. This severe eigenvalue dispersion causes gradient updates from initial and boundary terms ($\mathcal{L}_{ic}, \mathcal{L}_{bc}$) to be suppressed or prematurely minimized while the core differential equation constraint remains unfulfilled.

### Hybrid Two-Phase Strategy
To eliminate gradient stiffness and enforce physics constraints, optimization is executed in two distinct stages:

1. **Phase 1: ADAM + Learning Rate Annealing (LRA)**
   Dynamic weight multipliers $\lambda_i$ balance conflicting loss gradients using maximum-to-mean gradient ratios updated via exponential moving averages ($\alpha = 0.95$):
   $$\lambda_i = (1-\alpha)\lambda_i + \alpha \frac{\max_\theta |\nabla_\theta \mathcal{L}_f(\theta_n)|}{\overline{|\nabla_\theta \mathcal{L}_i(\theta_n)|}}$$
   Trained for 100,000 epochs with learning rate decay ($\eta_0 = 10^{-4}$ to $3.27 \times 10^{-6}$), driving total loss to $\approx 10^{-5}$ and positioning parameter weights within the global attraction basin.

2. **Phase 2: L-BFGS Fine-Grained Refinement**
   Switches to a second-order quasi-Newton L-BFGS optimizer using the parameter state inherited from Phase 1. Leveraging line searches and approximate Hessian curvature updates, L-BFGS achieves rapid final convergence to machine precision thresholds.

---

## Results & Benchmark Metrics

| Metric / Parameter | Pseudospectral Baseline | PINN (Phase 1: ADAM + LRA) | PINN (Phase 2: L-BFGS) |
| :--- | :--- | :--- | :--- |
| **Grid / Collocation Sampling** | $N = 1000$ spatial grid points | 128 dynamic points / domain | 128 dynamic points / domain |
| **Total Loss ($\mathcal{L}_{total}$)** | Exact (Spectral Accuracy) | $1.0 \times 10^{-5}$ | **$1.0 \times 10^{-8}$** |
| **PDE Loss ($\mathcal{L}_{f}$)** | N/A | $\approx 10^{-5}$ | **$\approx 10^{-9}$** |
| **Initial Loss ($\mathcal{L}_{ic}$)** | N/A | $\approx 10^{-6}$ | **$\approx 10^{-9}$** |
| **Boundary Loss ($\mathcal{L}_{bc}$)** | Periodic | $\approx 10^{-12}$ | **$\approx 10^{-8}$** |
| **Relative $L_2$ Error** | 0.00% (Ground Truth) | — | **9.53% ($9.53 \times 10^{-2}$)** |

---

## Repository Structure & Quickstart

```bash
├── config/
│   └── params.json          # Architecture, domain bounds, and hyperparameter setups
├── src/
│   ├── models/
│   │   └── siren_pinn.py    # SIREN MLP architecture with operator scaling
│   ├── solvers/
│   │   └── spectral_solver.py # Matrix exponential pseudospectral reference code
│   └── optimization/
│       └── lra_adam.py      # Learning Rate Annealing gradient balance engine
├── train.py                 # Hybrid Phase 1 (ADAM+LRA) & Phase 2 (L-BFGS) execution
└── evaluate.py              # $L_2$ relative error evaluation vs. pseudospectral truth
