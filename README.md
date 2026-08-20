# Physics-Informed Neural Networks (PINNs) Optimization

Welcome to this repository! This project optimizes Physics-Informed Neural Networks (PINNs) to solve the homogeneous Klein‑Gordon equation in 2D, comparing numerical and neural approaches.

## The Challenge
Training PINNs is computationally heavy and suffers from gradient stiffness, making convergence difficult—especially for high‑frequency wave solutions like those in relativistic quantum mechanics.

## Our Approach
I implemented **ADAM + Learning Rate Annealing (LRA)** and **L‑BFGS** optimizers with a **SIREN** architecture (sinusoidal activations, 6×128 neurons, min‑max normalization, and random collocation sampling) to stabilize training and capture high‑frequency behavior.

## Real‑World Impact
With L‑BFGS, I reduced the total loss to **10⁻⁸** and achieved a **9.53% relative L₂ error** against a pseudo‑spectral solution, validating the PINN's capability to model relativistic bosonic particle dispersion efficiently.

## Deep Dive
For a comprehensive look at the architecture details, testing methodology, and performance benchmarks, please refer to the main project document: 
📄 [Optimización_PINN.pdf](./Optimización_PINN___Optimización_numérica.pdf)
