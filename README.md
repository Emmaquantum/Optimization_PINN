# Optimization of Physics-Informed Neural Networks (PINNs) for the Relativistic Klein-Gordon PDE

This repository implements an optimization pipeline for solving the homogeneous Relativistic Klein-Gordon Partial Differential Equation (PDE) using Physics-Informed Neural Networks (PINNs)[cite: 2]. By combining custom domain normalization, Sinusoidal Representation Networks (SIREN), dynamic gradient annealing, and a two-stage hybrid optimization scheme, this project resolves stiffness and gradient flow pathologies common in physics-constrained deep learning[cite: 2]. A high-order pseudospectral solver utilizing matrix exponential propagators serves as the exact numerical ground truth for benchmarking[cite: 2].

---

## Key Features & Methodological Highlights

* **SIREN Neural Architecture**: Utilizes a 6-layer, 128-neuron deep MLP with sinusoidal activation functions ($\omega_0 = 30$) and variance-preserving initialization[cite: 2]. This architecture overcomes spectral bias and models high-frequency wave dispersion[cite: 2].
* **High-Precision Numerical Baseline**: Solves the 1D/2D Klein-Gordon wave dynamics using a pseudospectral method combined with exact matrix exponential propagators in Fourier space under CFL stability constraints[cite: 2].
* **Domain & Operator Normalization**: Implements min-max variable scaling to $[-1, 1]$ alongside exact chain-rule derivative operator scaling ($\alpha_x, \alpha_t$) to prevent vanishing gradients across operator scales[cite: 2].
* **Memory-Efficient Dynamic Resampling**: Dynamically resamples 128 collocation points per loss term per epoch (evaluating over $6.4 \times 10^6$ spacetime points across training) to prevent overfitting without saturating RAM[cite: 2].

---

## Optimization Breakdown

### The Failure Mode of Vanilla PINNs
Standard PINNs optimized with standard ADAM or unweighted SGD frequently converge to invalid trivial solutions or stall at apparent Pareto fronts[cite: 2]. Analysis via Karush-Kuhn-Tucker (KKT) conditions and Sobolev inequalities reveals that the Hessian spectral radius of the physical residual $\nabla_\theta^2 \mathcal{L}_f$ dominates the loss landscape[cite: 2]. This severe eigenvalue dispersion causes gradient updates from initial and boundary terms ($\mathcal{L}_{ic}, \mathcal{L}_{bc}$) to be suppressed or prematurely minimized while the core differential equation constraint remains unfulfilled[cite: 2].
