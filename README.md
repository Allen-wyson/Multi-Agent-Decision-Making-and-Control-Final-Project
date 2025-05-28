# EPFL ME-429 Fall 2024 Final Project: Zero‑Sum Racing Game


## 📋 Project Overview

This repository contains the final project for **ME-429: Multiagent Decision‑Making and Control** at EPFL, instructed by Dr. Maryam Kamgarpour. We model and simulate a head‑to‑head racing duel between two autonomous cars using a zero‑sum, dynamic feedback game. At each time step, each car solves a quadratic program (QP) to anticipate and counter its opponent’s move—much like two chess players thinking one move ahead, but on a racetrack.

## 🚀 Key Contributions

* **Zero‑Sum Game Formulation**: Casts the racing interaction as a multi‑stage, feedback game where one player’s gain is the other’s loss.
* **Iterative Best‑Response (IBR)**: Alternating quadratic programs yield saddle‑point controls at each discretized time step.
* **Bicycle‑Model Dynamics**: Realistic, non‑holonomic vehicle kinematics in both Cartesian and Frenet coordinates.
* **Collision & Track Constraints**: Enforces boundary limits and elliptical safety zones via slack‑augmented QPs.
* **Python Implementation**: Leverages `cvxpy` with OSQP solver for real‑time quadratic programming.

> **Everyday example**: Imagine two friends racing toy cars on a tabletop track—each plans its next move while keeping an eye on the other’s strategy to avoid crashes and win the duel.



## ▶️ Running the Simulation

All components are integrated in `simulation.py`. To simulate a race on a straight track:

```bash
python code/simulation.py
```

This will:

1. Generate a straight U‑shaped track.
2. Compute optimal trajectories for each car.
3. Run an iterative best‑response loop for T steps.
4. Produce plots


## 📈 Results Highlights

* **Straight‑Line Validation**: Both cars accelerate optimally toward the finish line, confirming saddle‑point behavior.
* **Convergence Analysis**: Relative cost gap |J₁ − J₂| ≈ 10% on average, suggesting effective IBR convergence.

*(See `docs/figures/` for detailed plots.)*

## 👥 Team & Acknowledgments

* **Clément Suttor** ([clement.suttor@epfl.ch](mailto:clement.suttor@epfl.ch))
* **Yo‑Shiun Cheng** ([yo‑shiun.cheng@epfl.ch](mailto:shiun.cheng@epfl.ch))

**Instructor:** Dr. Maryam Kamgarpour, EPFL
Thanks to the ME‑429 teaching team and the CVXPY/OSQP developers for support.

