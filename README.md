# PINN for Klein-Gordon: Solving wave equations without a mesh.

How do we solve physics equations without a mesh?

## The Problem
We solve the 2D Klein-Gordon equation. This models a relativistic quantum particle confined in a box and then released.

## Why this approach?
PINNs learn the solution directly from the PDE. This avoids costly mesh generation. We get a differentiable solution that can be queried at any point in space-time.

## The Optimization Strategy
We minimize a multi-objective loss function with a two-stage optimizer. First, we use ADAM with Learning Rate Annealing (LRA). LRA dynamically balances the gradients from the physics loss against the initial and boundary condition losses. This prevents the PDE residual from dominating the training. We use an exponential learning rate schedule, starting at 1e-4. After 100,000 epochs, we switch to L-BFGS. L-BFGS refines the solution quickly, reaching a stable Pareto optimum.

## Key Results
- Final relative L2 error of 9.53% against the pseudo-spectral solver.
- ADAM + LRA achieves a stable loss of 1e-5.
- L-BFGS refinement reaches a total loss of 1e-8 in under 500 iterations.
- The PINN correctly reproduces the relativistic light cone, matching the group velocity of the wave packet.

## Quick Start
```bash
git clone https://github.com/Emmaquantum/Optimization_PINN.git
cd Optimization_PINN
pip install -r requirements.txt
python run.py
