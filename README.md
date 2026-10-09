# Physics-Informed Neural Networks for Forward and Inverse ODE Problems

A from-scratch implementation of physics-informed neural networks (PINNs) in **Keras 3 / TensorFlow**, built to study where the method works, where it fails, and why. The project uses a damped harmonic oscillator as a controlled testbed to cover three distinct regimes: solving a known equation (forward), recovering unknown physics from data (inverse), and a research investigation into spectral bias.

No high-level PDE library (e.g. DeepXDE) is used — the residual, the nested-autodiff derivatives, and the training loops are implemented directly, to demonstrate the underlying mechanics.

---

## Motivation

A PINN learns a function that is constrained to obey a differential equation, by penalising the network whenever its own autodiff-computed derivatives violate the governing physics. The same machinery that solves a *forward* problem also solves the *inverse* problem — recovering a hidden physical parameter from sparse, noisy observations — with almost no change in code. This repository is organised to make that progression explicit, and to be honest about the method's limitations rather than showing only success cases.

The governing equation throughout is the damped harmonic oscillator:

$$ m\ddot{u} + \mu\dot{u} + k u = 0, \qquad u(0) = u_0,\ \dot{u}(0) = v_0 $$

which has a closed-form underdamped solution used for validation and for generating synthetic measurements.

---

## Repository structure

```
.
├── module1_forward.py      # Forward problem: solve a known ODE, validate vs exact solution
├── module2_inverse.py      # Inverse problem: recover hidden damping from noisy data
├── module3_spectral.py     # Research: spectral bias + Fourier-feature fix (in progress)
├── figures/                # Generated plots
├── docs/
│   └── pinn_project.pdf     # Full theory write-up and project brief
├── requirements.txt
└── README.md
```

---

## Results

### Module 1 — Forward problem
A PINN solves the oscillator from physics and initial conditions alone (no solution data), validated against the analytical solution. Autodiff derivatives were cross-checked against finite differences before training, to confirm the residual is built on correct gradients.

- Relative L² error: **2.7×10⁻³** (0.27%) after 20k Adam steps — the PINN reconstructs a solution it never saw.
- **Width/depth ablation:** error does *not* decrease monotonically with capacity — the smallest network did best here, and the largest beat the mid-size one, i.e. extra parameters bring diminishing (and noisy) returns on this simple problem.

| Width × depth | Params | Relative L² |
|---|---|---|
| 16 × 2 | 321 | 3.5×10⁻³ |
| 32 × 3 | 2,209 | 1.5×10⁻² |
| 64 × 4 | 12,673 | 5.2×10⁻³ |

*(Single seed per configuration — a rigorous version would average over several seeds, since initialisation noise is comparable to the differences above.)*

![Forward solution](figs/forward_solution.png)

### Module 2 — Inverse problem
The damping coefficient μ is hidden and recovered as a trainable variable, jointly with the solution field, from ~20 sparse noisy measurements. An Adam → L-BFGS two-stage optimisation is used for the headline run.

- Recovered **μ = 0.513** (true value 0.5) from 20 points at 3% noise — a 2.6% error.
- **Basin of attraction:** starting the estimate at μ₀ = 0, 1, and 5 all converge to ≈0.51, so recovery is robust to initialisation across the tested range.
- **Noise robustness:** error degrades smoothly and roughly proportionally with the noise level — no cliff.
- **Data-count identifiability threshold:** recovery is accurate at 20–40 points but fails *catastrophically* below ~10, where the problem becomes non-identifiable — a property of the inverse problem itself, not an optimiser failure.

**Noise sweep (20 points):**

| Noise σ | Recovered μ | Error |
|---|---|---|
| 0.00 | 0.501 | 0.2% |
| 0.03 | 0.513 | 2.6% |
| 0.10 | 0.546 | 9.2% |
| 0.20 | 0.612 | 22% |

**Data-count sweep (noise 0.05):**

| Data points | Recovered μ |
|---|---|
| 5  | 8.95 (fails) |
| 10 | 5.89 (fails) |
| 20 | 0.522 |
| 40 | 0.507 |

![Parameter recovery](figs/parameter_recovery.png)

### Module 3 — Spectral bias
A high-frequency version of the problem (ω = 10, ~16 cycles over the domain) is used to expose spectral bias, then fixed with a Fourier-feature input embedding. The *only* architectural change between the two runs below is a single non-trainable sin/cos embedding layer inserted after input rescaling.

- Plain MLP at ω = 10: relative L² = **0.88** — the network cannot represent the fast oscillation and collapses toward a near-flat curve.
- With Fourier features (σ = 5): relative L² = **6.9×10⁻³**, a **127× improvement** from that one layer.

| | plain MLP | + Fourier (σ = 5) |
|---|---|---|
| rel. L² at ω = 10 | 0.88 (fails) | 0.0069 (works) |

<p align="center">
  <img src="figs/spectral_failure.png" width="48%" alt="Plain MLP fails at high frequency"/>
  <img src="figs/spectral_fixed.png" width="48%" alt="Fourier features recover the solution"/>
</p>

*Left: a plain PINN exhibits spectral bias — it fits the low-frequency envelope but not the oscillation. Right: an identical network with a Fourier-feature embedding tracks the solution.*

---

## Installation

```bash
git clone https://github.com/<your-username>/pinn-oscillator.git
cd pinn-oscillator
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, Keras 3, TensorFlow, NumPy, SciPy, Matplotlib. The TensorFlow backend must be selected before importing Keras (handled at the top of each script via `KERAS_BACKEND=tensorflow`).

---

## Usage

Each module is a self-contained script:

```bash
python module1_forward.py     # forward solve + validation + residual map + width/depth ablation
python module2_inverse.py     # inverse recovery + robustness sweeps + basin of attraction
python module3_spectral.py    # spectral-bias study + Fourier fix + classical baseline
```

Module 2 runs several experiments (basin of attraction, headline recovery with L-BFGS polish, noise and data-count sweeps) and takes a few minutes at the default 15k epochs per run; lower the `epochs` argument for faster trends.

---

## Key implementation details

- **Derivatives** are computed with nested `tf.GradientTape` contexts to obtain the second derivative required by the residual.
- **Activation** is `tanh` throughout (smooth and infinitely differentiable), since the loss depends on second derivatives — `relu` would zero out the curvature term.
- **Inverse problem:** the unknown parameter is a `tf.Variable` added to the optimiser's variable list, so it is learned by the same gradient descent that trains the weights.
- **Fair experiments:** each sweep run builds a fresh model and a fresh parameter guess, so comparisons are not contaminated by prior training.

A full derivation of the theory and the project brief is in [`docs/pinn_project.pdf`](docs/pinn_project.pdf).

---

## References

1. Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378:686–707.
2. Tancik et al. (2020). *Fourier features let networks learn high frequency functions in low dimensional domains.* NeurIPS.
3. Wang, Teng & Perdikaris (2021). *Understanding and mitigating gradient flow pathologies in PINNs.* SIAM J. Sci. Comput.

---

## License

Released under the MIT License — see `LICENSE`.
