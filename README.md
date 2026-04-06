# Compositional Diffusion-Based Continuous Constraint Solvers (Diffusion-CCSP)

**Project Page:** [Diffusion-CCSP](https://diffusion-ccsp.github.io/)

**Paper:** [Compositional Diffusion-Based Continuous Constraint Solvers (CoRL 2023)](https://arxiv.org/abs/2309.00966)

---

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Research Directions & Methods](#research-directions--methods)
  - [1. Diffusion-CCSP (Original Published Approach)](#1-diffusion-ccsp-original-published-approach)
  - [2. Flow Matching as Diffusion Replacement](#2-flow-matching-as-diffusion-replacement)
  - [3. Projected Flow-CCSP (CBF-QP Correction)](#3-projected-flow-ccsp-cbf-qp-correction)
  - [4. Message-Passing Flow Architecture](#4-message-passing-flow-architecture)
  - [5. Iterative Restart Flow](#5-iterative-restart-flow)
  - [6. Stochastic Flow (SDE) and Energy-Guided Flow](#6-stochastic-flow-sde-and-energy-guided-flow)
  - [7. Systematic Constraint Composition Experiments](#7-systematic-constraint-composition-experiments)
  - [8. CNN Geometry Encoders](#8-cnn-geometry-encoders)
  - [9. 3D Robot Tasks (Packing & Stability)](#9-3d-robot-tasks-packing--stability)
- [Constraint Types](#constraint-types)
- [Evaluation Framework](#evaluation-framework)
- [Key Findings & Lessons Learned](#key-findings--lessons-learned)
- [Setup & Usage](#setup--usage)
- [Citation](#citation)

---

## Overview

This repository contains a comprehensive research program exploring **generative and optimization-based methods** for solving Compositional Constraint Satisfaction Problems (CCSPs) in robot Task and Motion Planning (TAMP). The central challenge is: given objects with geometric properties and a graph of constraints between them, find valid poses for all objects that simultaneously satisfy all constraints.

The key insight driving this research is **compositionality**: by training on individual constraint types, models should generalize to solve novel combinations at test time. This repository systematically explores diffusion models, flow matching, energy-based methods, projection-based optimization, and hybrid approaches.

---

## Problem Formulation

**Input:**
- A set of N objects, each with geometry (shape, size)
- A constraint graph where edges encode pairwise constraints (collision-free, spatial relations, containment, etc.)
- Optional: unary constraints (object must be in tray region)

**Output:**
- Continuous poses (x, y, orientation) for all objects satisfying all constraints

**Challenge:**
- Constraints can conflict; the solution space may be disconnected
- Must generalize to unseen constraint graph structures at test time
- Solutions must be found efficiently for real-time robot planning

---

## Research Directions & Methods

### 1. Diffusion-CCSP (Original Published Approach)

**Files:** `train_ddpm.py`, `networks/ddpm.py`, `networks/denoise_fn.py`, `solve_csp.py`, `solve_csp_rejection.py`

**Line of Reasoning:**
The foundational approach uses Denoising Diffusion Probabilistic Models (DDPM) to learn to denoise object poses from Gaussian noise, conditioned on the constraint graph. The core hypothesis is that the compositional structure of the denoising network mirrors the compositional structure of the constraint graph, enabling generalization.

**Method:**
- **Per-constraint MLPs:** Each constraint type has a dedicated denoising network that takes geometries, poses, and time embedding for the two involved objects
- **Scatter-add aggregation:** Predictions from all constraints acting on an object are summed with sqrt-normalization
- **Time conditioning:** Sinusoidal positional encoding enables the model to learn the reverse diffusion process across timesteps
- **Architecture variants:** 
  - ConstraintDiffuser (per-constraint MLPs)
  - StructDiffusion (Transformer-based with attention masks)
- **Sampling schedules:** Cosine, sigmoid, and linear beta schedules explored

**Energy-Based Model (EBM) Composition:**
Beyond standard diffusion, the model computes an "energy" as the sum of squared residuals between predicted and current poses. This energy landscape enables additional correction methods:
- **ULA (Unadjusted Langevin Algorithm):** Gradient-based correction using ∇E
- **ULA+:** Enhanced Langevin with adjusted step sizes
- **HMC (Hamiltonian Monte Carlo):** Momentum-based exploration
- **MALA (Metropolis-Adjusted Langevin Algorithm):** MCMC with accept/reject steps

**Results:**
- 2 objects: ~96% success rate
- 3 objects: ~92% success rate  
- 4 objects: ~70% success rate
- 5 objects: ~44% success rate
- Performance degrades gracefully with problem complexity

**Baselines:**
- Rejection sampling: samples from model predictions, checks constraints (lower bound)
- Pure reverse diffusion (no ULA): tests base generative quality

---

### 2. Flow Matching as Diffusion Replacement

**Files:** `flow_matching/train_flow.py`, `train_flow_v2.py`, `train_flow_v3.py`, `train_flow_v4.py`, `flow_matching/flow_message_passing.py`

**Line of Reasoning:**
Flow Matching offers an alternative to diffusion by learning a velocity field that transports noise to data via ODE integration, rather than learning denoising steps. The hypothesis is that flow matching may be more computationally efficient (single ODE solve vs. many diffusion steps) and may provide better compositionality through the continuous-time velocity field.

**Method:**
- **Optimal Transport CFM:** Uses the interpolant `x_t = (1-t)*x_0 + t*x_1` with velocity target `u_t = x_1 - x_0`
- **Training objective:** MSE loss between predicted and conditional velocity on free (non-clamped) nodes only
- **Architecture mirroring:** ConstraintDiffuser-like architecture (geom_encoder, pose_encoder, time_mlp, constraint_mlps, pose_decoder) with flat parameter naming for checkpoint compatibility
- **Aggregation strategies explored:**
  - `sum`: scatter-add with sqrt-normalization (original)
  - `attention`: softmax-weighted aggregation
  - `maxpool`: winner-take-all aggregation

**Findings:**
- Flow matching alone underperforms diffusion without correction mechanisms
- Trained models are faster at inference (fewer integration steps needed)
- The learned velocity field provides useful directional information for correction methods

---

### 3. Projected Flow-CCSP (CBF-QP Correction)

**Files:** `flow_matching/solve_flow_ccsp.py`, `solve_flow_ccsp.py.v0_baseline`, `flow_matching/fix_and_eval.py`, `fix_and_eval.py.v0_baseline`

**Line of Reasoning:**
Flow models alone struggle with hard constraint satisfaction. This approach combines the speed of flow-based proposal generation with rigorous constraint enforcement via Control Barrier Function Quadratic Programming (CBF-QP). The key insight is to use the flow model to propose a solution in a feasible neighborhood, then use analytical barriers to enforce exact constraint satisfaction.

**Method (Two-Phase Algorithm):**

**Phase 1 — Prediction (unconstrained):**
- Euler integration of flow model for T_p steps from noise
- Produces a quick proposal in a feasible neighborhood

**Phase 2 — Correction (CBF-QP projected):**
- For T_c correction steps:
  1. Compute vanishing velocity: `ṽ = α(1-t) · v_θ(x_t, t)`
  2. For each constraint: evaluate barrier `h_c(x_t)` and gradient `∇h_c`
  3. Solve QP: `u* = argmin ||u - ṽ||²` subject to CBF constraints
  4. Update: `x_{t+dt} = x_t + u* · dt`

**Barrier Functions (Analytic CBFs):**
- `in`: minimum margin from tray walls
- `cfree`: AABB separation with overlap-based gradient amplification
- `close-to`, `away-from`: distance-based barriers with violation-proportional scaling
- `left-of`, `top-of`, `right-of`, `bottom-of`: directional separation
- `h-aligned`, `v-aligned`: alignment barriers
- `center-in`, `left-in`, `right-in`, `top-in`, `bottom-in`: region containment

**Key Fixes Identified & Applied:**
1. **Hard tray clamping:** Flow models generate out-of-tray poses; fixed by clamping x,y to `[w+margin, 2-w-margin]` after each step
2. **Stronger QP epsilon:** Initial corrections insufficient; epsilon=1.0, rho=0.8, 40 correction steps required
3. **Multi-step prediction:** 5 prediction steps (not 1) needed for better initial proposal
4. **Cfree gradient amplification:** Added overlap-based scaling for deeply violating configurations
5. **Away-from gradient amplification:** Violation-proportional scaling for faster convergence

---

### 4. Message-Passing Flow Architecture

**Files:** `flow_matching/flow_message_passing.py`, `flow_matching/train_flow_message_passing.py`, `colab_option_ab.ipynb`

**Line of Reasoning:**
The original scatter-add aggregation processes all constraints independently in a single pass. This limits the model's ability to capture interactions between constraints. Message passing allows constraints to "communicate" within each evaluation step.

**Method:**
- **Intra-step message passing:** Run K rounds of constraint message aggregation
- **Residual updates:** `node_h = node_h + LayerNorm(node_update([node_h, agg_msg, node_t]))`
- **Parameters:** `n_rounds` (default 3), `residual` (default True)
- **Hypothesis:** Iterative message passing enables the model to reason about constraint interactions and conflicts

---

### 5. Iterative Restart Flow

**Files:** `flow_matching/eval_flow_iterative.py`

**Line of Reasoning:**
Pure flow sampling, like pure reverse diffusion, is limited in its ability to escape local minima. By treating the flow sampler as a reusable correction primitive and restarting with noise, we can achieve diffusion-like global refinement without retraining.

**Method:**
```
x_prev = None
for restart in range(n_restarts):
    if x_prev is None:
        x_t = randn()
    else:
        x_t = x_prev + sigma * randn()  # sigma decays
    x_t = flow_integrate(x_t)  # n_steps of Euler
    x_prev = x_t
return x_prev
```

**Key insight:** Restarting with decaying noise provides exploration similar to DDPM's stochastic reverse process, but works with deterministic flow models.

---

### 6. Stochastic Flow (SDE) and Energy-Guided Flow

**Files:** `flow_matching/experiment_flow_fixes.py`

**Line of Reasoning:**
Two hypotheses for why flow underperforms diffusion:
- **Hypothesis A:** Lack of stochastic exploration → Add noise to ODE (SDE)
- **Hypothesis B:** Lack of energy-guided composition → Add Langevin correction using per-constraint energies

**Method A — Stochastic Flow (SDE):**
- Convert deterministic ODE to SDE: `x_{t+dt} = x_t + v*dt + sigma(t)*sqrt(dt)*noise`
- Sigma decays over time (linear or cosine schedule)
- Early steps explore; late steps converge

**Method B — Energy-Guided Flow:**
- At each flow step, compute composed energy `E = sum ||v_c(x_t, t)||²` from per-constraint MLP outputs
- Apply Langevin correction: `x = x - lr * ∇E + noise`
- Mimics Diffusion-CCSP's ULA composition

---

### 7. Systematic Constraint Composition Experiments

**Files:** `experiments/constraint_composition/runner.py`, `methods.py`, `core.py`, `metrics.py`

**Line of Reasoning:**
This is the most extensive research direction, systematically exploring **15+ method families** for compositional constraint satisfaction. The goal is to understand which architectural choices, training strategies, and solving algorithms enable the best generalization.

#### Method Families Tested:

**1. Langevin Methods:**
- Energy descent (normalized/unnormalized gradients)
- Annealed Langevin with various noise schedules
- Tests whether gradient-based optimization on constraint violation energy suffices

**2. Projection Methods:**
- Projected energy descent (POCS-style: project onto each constraint set sequentially)
- Sequential projection with descending/fixed/random ordering
- Tests whether alternating projections can satisfy all constraints

**3. Prototype Methods:**
- Learn prototype configurations for each constraint type
- Use prototypes as energy via numerical gradients
- Hypothesis: storing canonical solutions enables composition

**4. Learned Energy Methods:**
- Train per-constraint energy models on extracted invariant features
- Compose energies by summing per-constraint predictions
- Tests whether learned energies outperform hand-crafted violation metrics

**5. Global Energy Methods:**
- Train a single energy model on global features (all objects simultaneously)
- Tests whether a monolithic model can learn composition

**6. Vector Field Methods:**
- Train direct vector field model (no time conditioning)
- Finite-difference gradients of total violation as targets
- Tests whether direct policy learning works

**7. Vector Field with Time:**
- Time-conditioned vector field (like diffusion but single-step prediction)
- Unrolled training trajectories for better targets

**8. Graph Neural Network Methods:**
- **Graph noise:** GNN vector fields with node features + edge structure
- **Graph flow matching:** Graph-based flow matching
- **Graph DAgger:** Iterative aggregation of expert trajectories
- Tests whether explicit graph structure helps

**9. Score-Based Methods:**
- **Graph score:** Score estimation at noise levels
- **Graph score +:** Score model + normalized gradient direction + residual learning
- **Graph score projected:** Score model + linear projection
- **Graph score + priority:** Priority-based selective projection
- **Graph score two-phase:** Coarse-to-fine model with learned switching

**10. Rectified Flow Methods:**
- Composed per-constraint velocity via individual projections
- Variants: raw, fixed trajectory, Langevin variants

**Aggregation Strategies:**
All methods tested with multiple aggregation strategies:
- Sum (scatter-add + sqrt-norm)
- Attention (softmax-weighted)
- Max-pool (winner-take-all)

---

### 8. CNN Geometry Encoders

**Files:** `train_encoders.py`

**Line of Reasoning:**
The original MLP geometry encoder may be insufficient for complex shapes. A CNN autoencoder trained on shape images could provide richer geometry representations.

**Method:**
- GeomAutoEncoder with GeomEncoderImage (3 conv layers + pooling)
- Trains on 64×64 images of triangle shapes
- Encoder weights frozen when training DDPM
- Tests whether better geometry encoding improves constraint satisfaction

---

### 9. 3D Robot Tasks (Packing & Stability)

**Files:** `simulation/3-panda-box-data.py`, `simulation/5-panda-stability-data.py`, `simulation/demo_utils.py`

**Line of Reasoning:**
Validate that methods developed on 2D qualitative domains generalize to realistic 3D robot planning scenarios.

**Tasks:**
1. **3D Object Packing (robot_box):** Franka Panda robot packing objects into a box/tray
   - Constraints: `gin` (grasp-in), `gfree` (grasp-free), qualitative constraints
   
2. **3D Stability (stability_flat):** Stacking shapes with stability constraints
   - Constraints: `within` (in tray), `supportedby` (on top of another object), `cfree`

**Simulation environment:** PyBullet with Franka Panda and UR5 robots, IK compilation

---

## Constraint Types

The system handles multiple constraint taxonomies across different domains:

| Domain | Constraints | Description |
|--------|-------------|-------------|
| **Puzzle (2D geometric)** | `in`, `cfree` | Object in tray, collision-free |
| **Stability (3D)** | `within`, `supportedby`, `cfree` | Containment, support, no collision |
| **Qualitative (2D relational)** | `in`, `center-in`, `left-in`, `right-in`, `top-in`, `bottom-in`, `cfree`, `left-of`, `top-of`, `close-to`, `away-from`, `h-aligned`, `v-aligned` | Spatial relationships |
| **Robot** | `gin`, `gfree` + qualitative | Grasp constraints + spatial relations |

---

## Evaluation Framework

**Metrics:**

1. **Per-trial success rate:** Fraction of individual samples satisfying all constraints (barrier ≥ -0.02 tolerance)

2. **Scene-level top-1:** Fraction of test problems solved on first attempt

3. **Scene-level top-k:** Fraction of test problems solved in at least 1 of k attempts

4. **Per-constraint satisfaction rate:** Breakdown by constraint type (e.g., `cfree`, `away-from`, `left-of`)

5. **Trajectory metrics:**
   - Monotonic fraction (does energy decrease monotonically?)
   - Plateau fraction (does energy stall?)
   - Mean step cosine (alignment with negative gradient)
   - Mean/max step size
   - Late-stage convergence metrics

6. **Timing:** Per-graph sampling time in milliseconds

**Test protocol:**
- Evaluate on held-out test problems with varying object counts (2-5 objects)
- Multiple samples per problem (typically 10)
- Report both first-success rate and overall constraint satisfaction

---

## Key Findings & Lessons Learned

1. **Diffusion vs. Flow:** Pure diffusion with ULA correction outperforms pure flow matching. Flow matching requires additional correction mechanisms (CBF-QP, restarts, or energy guidance) to match diffusion performance.

2. **Constraint composition is hard:** Performance degrades significantly with problem size (4-5 objects). This suggests that simple scatter-add aggregation cannot fully capture complex constraint interactions.

3. **Correction mechanisms are essential:** All successful methods use some form of iterative correction (Langevin, projection, QP, or restarts). Single-pass generation is insufficient for hard constraints.

4. **Analytic barriers work:** Hand-crafted barrier functions with gradient amplification for violations outperform learned energy models for geometric constraints.

5. **Aggregation matters:** The choice of aggregation (sum vs. attention vs. maxpool) significantly impacts performance, but no single strategy dominates across all problem types.

6. **Stochasticity helps:** Adding noise (whether through Langevin dynamics, SDE conversion, or iterative restarts) consistently improves success rates by enabling escape from local minima.

7. **Tray containment is a failure mode:** Flow models in particular struggle to keep objects within tray boundaries, requiring hard clamping as a post-processing step.

8. **Gradient amplification is critical:** For deeply violating configurations (especially collisions), standard gradients are too weak. Scaling gradients based on violation magnitude dramatically improves convergence.

9. **Two-phase approaches show promise:** Coarse-to-fine methods (e.g., graph score two-phase) that switch between models at different stages can outperform single-model approaches.

10. **Graph structure provides limited benefit:** Explicit GNN architectures do not consistently outperform simpler per-constraint MLPs with scatter-add, suggesting the constraint graph structure is already well-exploited by the compositional architecture.

---

## Setup & Usage

### Prerequisites

1. Clone this repo with submodules:
   ```shell
   git clone https://github.com/zt-yang/diffusion-ccsp.git --recurse-submodules
   ```

2. Set up Jacinle framework:
   ```shell
   cd ..
   git clone https://github.com/vacancy/Jacinle --recursive
   ```

3. Set up dependencies:
   ```shell
   cd diffusion-ccsp
   conda create --name diffusion-ccsp python=3.9
   conda activate diffusion-ccsp
   pip install -r requirements.txt
   ```

4. Source environment variables:
   ```shell
   source setup.sh
   ```

### Download Data & Checkpoints

```shell
python download_data_checkpoints.py
```

### Training

**Diffusion-CCSP:**
```shell
python train_ddpm.py -timesteps 1000 -EBM 'ULA' -input_mode qualitative
```

**Flow Matching:**
```shell
python flow_matching/train_flow.py -input_mode qualitative
```

**CNN Encoder:**
```shell
python train_encoders.py
```

### Evaluation

**Diffusion-CCSP:**
```shell
python solve_csp.py
```

**Projected Flow-CCSP:**
```shell
# Flow-only (fast baseline)
python solve_flow_ccsp.py -input_mode qualitative -checkpoint best

# Full CBF-QP correction
python solve_flow_ccsp.py -input_mode qualitative -checkpoint best -n_corr_steps 40
```

**Constraint Composition Experiments:**
```shell
python experiments/constraint_composition/runner.py --suite langevin
```

### Data Generation

**2D Qualitative tasks:**
```shell
python simulation/envs/data_collectors.py -world_name 'RandomSplitQualitativeWorld' -data_type 'train' -num_worlds 100
python simulation/envs/data_collectors.py -world_name 'RandomSplitQualitativeWorld' -data_type 'test' -num_worlds 10 -pngs -jsons
```

**3D Robot tasks:**
```shell
# 3D object packing
python simulation/3-panda-box-data.py

# 3D stability
python simulation/5-panda-stability-data.py
```

---

## Citation

```bibtex
@inproceedings{yang2023diffusion,
  title={{Compositional Diffusion-Based Continuous Constraint Solvers}},
  author={Yang, Zhutian and Mao, Jiayuan and Du, Yilun and Wu, Jiajun and Tenenbaum, Joshua B. and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie Pack},
  booktitle={Conference on Robot Learning},
  year={2023},
}
```

---

## Repository Structure

```
diffusion-ccsp/
├── networks/                    # Core neural architectures
│   ├── ddpm.py                  # GaussianDiffusion + Trainer
│   ├── denoise_fn.py            # ConstraintDiffuser, GeomAutoEncoder
│   ├── transformer.py           # StructDiffusion variant
│   └── data_transforms.py       # Preprocessing, normalization
├── flow_matching/               # Flow matching research branch
│   ├── train_flow*.py           # Flow training (v1-v4)
│   ├── solve_flow_ccsp.py       # Projected Flow-CCSP solver
│   ├── flow_message_passing.py  # Message-passing architecture
│   ├── eval_flow_iterative.py   # Iterative restart evaluation
│   ├── experiment_flow_fixes.py # SDE and energy-guided flow
│   └── datasets.py              # GraphDataset class
├── experiments/constraint_composition/  # Systematic method comparison
│   ├── runner.py                # Main experiment runner (15+ suites)
│   ├── methods.py               # All solving method implementations
│   ├── core.py                  # SceneSpec, barriers, constraint eval
│   ├── metrics.py               # Trajectory evaluation metrics
│   └── *_dataset.py             # Dataset builders for each method
├── simulation/                  # PyBullet simulation environment
│   ├── envs/                    # World builders, data collectors
│   └── pybullet_engine/         # Robot simulation (Panda, UR5)
├── train_ddpm.py                # DDPM training entry point
├── train_encoders.py            # CNN autoencoder training
├── solve_csp.py                 # Diffusion-CCSP evaluation
└── solve_csp_rejection.py       # Rejection sampling baseline
```
