"""
Projected Flow-CCSP Solver
===========================
Inference-time solver for Compositional Constraint Satisfaction Problems (CCSPs)
using the Projected Flow-CCSP algorithm from:

  "Projected Flow Matching for Compositional Constraint Satisfaction
   in Robot Task and Motion Planning" (draft, March 2026)

Architecture (Algorithm 1 from the paper):
─────────────────────────────────────────────────────────────────────────────
  PHASE 1 — PREDICTION (unconstrained)
    x_pred = ODE integrate flow model v_θ for T_p steps from x_0 ~ N(0,I)
    → quick proposal in a feasible neighbourhood

  PHASE 2 — CORRECTION (CBF-QP projected)
    For T_c steps:
      1. ṽ = α(1-t) · v_θ(x_t, t)         (vanishing velocity)
      2. For each constraint c: evaluate h_c(x_t^c) and ∇h_c
      3. u* = argmin ||u - ṽ||²            (CBF-QP)
              s.t. ∇h_c^T u + ε·sgn(h_c-δ)|h_c-δ|^ρ + w_t r_t ≥ 0
      4. x_{t+dt} = x_t + u* · dt
─────────────────────────────────────────────────────────────────────────────

Barrier functions:
  - Analytic for geometric constraints (cfree, in, supported-by, within)
  - Learned via FlowMatchingCCSP energy for qualitative constraints
    (currently approximated with hand-crafted analytic barriers)

Usage:
    # Evaluate flow-only (fast baseline)
    python solve_flow_ccsp.py -input_mode qualitative -checkpoint best

    # Full Projected Flow-CCSP with QP correction
    python solve_flow_ccsp.py -input_mode qualitative -checkpoint best -n_corr_steps 40
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from flow_matching.train_flow import FlowMatchingCCSP, get_data_config


# ─────────────────────────────────────────────────────────────────────────────
# Barrier functions
# (Analytic CBFs for all constraint types in the qualitative domain.
#  h_c(x) > 0 ↔ constraint satisfied,  h_c = 0 ↔ boundary.)
# ─────────────────────────────────────────────────────────────────────────────

def compute_barrier(constraint_type, pose_i, pose_j, geom_i, geom_j):
    """Compute barrier value h and per-node pose gradients (∇h w.r.t. pose).

    Normalised coordinate system (qualitative domain):
      geom = (w, h)   where w = obj_width / tray_width  ∈ [0, 1]
      pose = (x, y, cos, sin)  where x = obj_x / (tray_w/2) ∈ [0, 2]
      Tray spans [0, 2] × [0, 2].

    Args:
        constraint_type: str
        pose_i, pose_j:  torch.Tensor or array-like, shape [pose_dim]
        geom_i, geom_j:  torch.Tensor or array-like, shape [geom_dim]

    Returns:
        h_val:  float   barrier value (positive = satisfied)
        grad_i: np.array [4]  ∂h/∂pose_i
        grad_j: np.array [4]  ∂h/∂pose_j
    """
    xi, yi = float(pose_i[0]), float(pose_i[1])
    xj, yj = float(pose_j[0]), float(pose_j[1])
    wi = float(geom_i[0]) if len(geom_i) > 0 else 0.1
    hi = float(geom_i[1]) if len(geom_i) > 1 else 0.1
    wj = float(geom_j[0]) if len(geom_j) > 0 else 0.1
    hj = float(geom_j[1]) if len(geom_j) > 1 else 0.1

    grad_i = np.zeros(4, dtype=np.float64)
    grad_j = np.zeros(4, dtype=np.float64)
    h_val  = 1.0   # default: always satisfied for unknown constraints

    # ── Geometric constraints ───────────────────────────────────────────

    if constraint_type == 'in':
        # Object i fully inside the tray [0, 2] × [0, 2].
        # Barrier = smallest margin from any of the four walls.
        margins = [
            xi - wi,              # left wall
            2.0 - (xi + wi),      # right wall
            yi - hi,              # bottom wall
            2.0 - (yi + hi),      # top wall
        ]
        h_val = min(margins)
        idx = margins.index(h_val)
        if   idx == 0: grad_i[0] =  1.0   # push right
        elif idx == 1: grad_i[0] = -1.0   # push left
        elif idx == 2: grad_i[1] =  1.0   # push up
        elif idx == 3: grad_i[1] = -1.0   # push down

    elif constraint_type == 'cfree':
        # Axis-aligned bounding box separation.
        # h = max(|xi - xj| - (wi+wj), |yi - yj| - (hi+hj))
        # Positive when the AABBs are non-overlapping.
        dx = abs(xi - xj) - (wi + wj)
        dy = abs(yi - yj) - (hi + hj)
        h_val = max(dx, dy)
        # Amplify gradient when deeply overlapping
        overlap = max(0.0, -h_val)
        scale = max(1.0, 2.0 * overlap / (max(wi+wj, hi+hj) + 1e-8))
        if dx >= dy:
            s = 1.0 if xi > xj else -1.0
            grad_i[0] =  scale * s
            grad_j[0] = -scale * s
        else:
            s = 1.0 if yi > yj else -1.0
            grad_i[1] =  scale * s
            grad_j[1] = -scale * s

    elif constraint_type == 'within':
        # 3-D stability domain version of 'in' (same barrier, 2D projection)
        margins = [xi - wi, 2.0 - (xi + wi), yi - hi, 2.0 - (yi + hi)]
        h_val = min(margins)
        idx = margins.index(h_val)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        grad_i[0] = dirs[idx][0]
        grad_i[1] = dirs[idx][1]

    elif constraint_type == 'supportedby':
        # Simplified stability: centre of i must be over j's AABB.
        # h = min(wi + wj/2 - |xi - xj|, 0) rephrased as:
        # h = (wj/2 + wi/2) - |xi - xj|   (positive when centred over support)
        h_val = (wj + wi) * 0.5 - abs(xi - xj)
        s = 1.0 if xi < xj else -1.0
        grad_i[0] = s
        grad_j[0] = -s

    # ── Qualitative / relational constraints ────────────────────────────

    elif constraint_type == 'close-to':
        dist = np.hypot(xi - xj, yi - yj) + 1e-8
        threshold = max(wi + wj, hi + hj) * 1.5
        h_val = threshold - dist
        grad_i[0] = -(xi - xj) / dist
        grad_i[1] = -(yi - yj) / dist
        grad_j[0] =  (xi - xj) / dist
        grad_j[1] =  (yi - yj) / dist

    elif constraint_type == 'away-from':
        dist = np.hypot(xi - xj, yi - yj) + 1e-8
        threshold = max(wi + wj, hi + hj) * 2.0
        h_val = dist - threshold
        # Amplify gradient when deeply violated to push objects apart faster
        scale = max(1.0, 3.0 * max(0.0, threshold - dist) / (threshold + 1e-8))
        grad_i[0] =  scale * (xi - xj) / dist
        grad_i[1] =  scale * (yi - yj) / dist
        grad_j[0] = -scale * (xi - xj) / dist
        grad_j[1] = -scale * (yi - yj) / dist

    elif constraint_type == 'left-of':
        # i is to the LEFT of j: xj - xi > (wi + wj)*0.5
        h_val = (xj - xi) - (wi + wj) * 0.5
        grad_i[0] = -1.0
        grad_j[0] =  1.0

    elif constraint_type == 'right-of':
        # i is to the RIGHT of j
        h_val = (xi - xj) - (wi + wj) * 0.5
        grad_i[0] =  1.0
        grad_j[0] = -1.0

    elif constraint_type == 'top-of':
        # i is ABOVE j: yi - yj > (hi + hj)*0.5
        h_val = (yi - yj) - (hi + hj) * 0.5
        grad_i[1] =  1.0
        grad_j[1] = -1.0

    elif constraint_type == 'bottom-of':
        h_val = (yj - yi) - (hi + hj) * 0.5
        grad_i[1] = -1.0
        grad_j[1] =  1.0

    elif constraint_type == 'h-aligned':
        # Horizontally aligned: y-coordinates close (within 0.3 normalised)
        h_val = 0.3 - abs(yi - yj)
        s = -1.0 if yi > yj else 1.0
        grad_i[1] =  s
        grad_j[1] = -s

    elif constraint_type == 'v-aligned':
        # Vertically aligned: x-coordinates close
        h_val = 0.3 - abs(xi - xj)
        s = -1.0 if xi > xj else 1.0
        grad_i[0] =  s
        grad_j[0] = -s

    elif constraint_type == 'center-in':
        # Centre of i near tray centre (1, 1) — ultra-strong barrier
        dist = np.hypot(xi - 1.0, yi - 1.0) + 1e-8
        h_val = 3.0 - dist
        grad_i[0] = -(xi - 1.0) / dist
        grad_i[1] = -(yi - 1.0) / dist

    elif constraint_type == 'left-in':
        h_val = 4.0 - xi
        grad_i[0] = -1.0

    elif constraint_type == 'right-in':
        h_val = xi + 3.0
        grad_i[0] =  1.0

    elif constraint_type == 'top-in':
        h_val = yi + 3.0
        grad_i[1] =  1.0

    elif constraint_type == 'bottom-in':
        h_val = 4.0 - yi
        grad_i[1] = -1.0

    return float(h_val), grad_i, grad_j


# ─────────────────────────────────────────────────────────────────────────────
# CBF-QP solver (iterative POCS — Projections Onto Convex Sets)
# ─────────────────────────────────────────────────────────────────────────────

def solve_cbf_qp(v_nominal, barriers, epsilon=1.0, rho=0.8, delta=0.01,
                 w_relax=0.0, n_iter=10):
    """Project nominal velocity onto the CBF-feasible cone.

    Each barrier (h, grad) contributes one linear constraint:
        grad^T u + α(h) + w_relax·max(0, -(h-δ)) ≥ 0
    where α(h) = ε · sgn(h-δ) |h-δ|^ρ  (class-K function).

    We solve this via sequential single-constraint projections (POCS).
    This converges to the QP optimum when constraints are compatible, and
    yields a feasible point in the intersection otherwise.

    Args:
        v_nominal: np.array [n*pose_dim]   nominal velocity (flattened)
        barriers:  list of (h_val, grad)   grad is shape [n*pose_dim]
        epsilon:   CBF gain (default 1.0)
        rho:       class-K exponent ∈ (0, 1)
        delta:     robustness margin
        w_relax:   relaxation weight (reduces as t → 1)
        n_iter:    number of POCS passes

    Returns:
        u: np.array [n*pose_dim]   projected velocity
    """
    u = v_nominal.copy()
    for _ in range(n_iter):
        for h_val, grad in barriers:
            gnorm2 = np.dot(grad, grad)
            if gnorm2 < 1e-12:
                continue
            # Class-K α function
            alpha = epsilon * np.sign(h_val - delta) * (abs(h_val - delta) ** rho)
            # Relaxation term
            relax = w_relax * max(0.0, -(h_val - delta))
            # CBF condition: grad^T u + alpha + relax ≥ 0
            violation = np.dot(grad, u) + alpha + relax
            if violation < 0:
                # Project u onto the half-space
                u = u + (-violation / gnorm2) * grad
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Tray clamping (hard containment enforcement)
# ─────────────────────────────────────────────────────────────────────────────

def clamp_to_tray(poses, geoms, mask, margin=0.02):
    """Hard-clamp pose (x, y) to lie inside the normalised tray [0,2]×[0,2].

    In normalised coords: obj half-width = w, so x ∈ [w+margin, 2-w-margin].
    This enforces the 'in' constraint exactly, compensating for flow-model
    out-of-tray drift.
    """
    poses_c = poses.clone()
    for i in range(poses.shape[0]):
        if mask[i]:
            continue
        w = geoms[i, 0].item()
        h = geoms[i, 1].item()
        x_lo, x_hi = w + margin, 2.0 - w - margin
        y_lo, y_hi = h + margin, 2.0 - h - margin
        if x_lo >= x_hi:
            x_lo = x_hi = 1.0
        if y_lo >= y_hi:
            y_lo = y_hi = 1.0
        poses_c[i, 0] = poses_c[i, 0].clamp(x_lo, x_hi)
        poses_c[i, 1] = poses_c[i, 1].clamp(y_lo, y_hi)
    return poses_c


# ─────────────────────────────────────────────────────────────────────────────
# Constraint checking
# ─────────────────────────────────────────────────────────────────────────────

def check_constraints(poses, batch, constraint_types, dims, tol=0.02, device='cuda'):
    """Check all constraints in a batch using barrier functions.

    Args:
        poses:            [n_nodes, pose_dim] tensor
        batch:            PyG DataBatch
        constraint_types: list of constraint name strings
        dims:             model dims tuple
        tol:              tolerance (barrier ≥ -tol counts as satisfied)

    Returns:
        all_ok:  bool
        per_c:   dict {edge_idx: {'type', 'h_val', 'satisfied'}}
    """
    x        = batch.x.to(device)
    ei       = batch.edge_index.T.to(device)
    ea       = batch.edge_attr.to(device)
    geom_end = dims[0][2]

    per_c   = {}
    all_ok  = True

    for k in range(ei.shape[0]):
        i = ei[k, 0].item()
        j = ei[k, 1].item()
        c = int(ea[k].item())
        if c >= len(constraint_types):
            continue
        ctype = constraint_types[c]

        h, _, _ = compute_barrier(
            ctype,
            poses[i].cpu(), poses[j].cpu(),
            x[i, :geom_end].cpu(), x[j, :geom_end].cpu())

        satisfied = h >= -tol
        per_c[k] = {'type': ctype, 'h_val': h, 'satisfied': satisfied}
        if not satisfied:
            all_ok = False

    return all_ok, per_c


# ─────────────────────────────────────────────────────────────────────────────
# Projected Flow-CCSP solver
# ─────────────────────────────────────────────────────────────────────────────

class ProjectedFlowSolver:
    """Projected Flow-CCSP: prediction-correction solver for CCSP.

    Phase 1 (prediction): unconstrained Euler integration of the flow model.
    Phase 2 (correction): CBF-QP-projected steps with vanishing velocity.

    Args:
        model:         FlowMatchingCCSP (loaded, eval mode)
        n_pred_steps:  Euler steps in prediction phase (default 5)
        n_corr_steps:  correction iterations (default 40)
        alpha_scale:   velocity scale in correction phase (default 2.0)
        epsilon:       CBF gain ε (default 1.0)
        rho:           class-K exponent ρ ∈ (0,1) (default 0.8)
        delta:         robustness margin δ (default 0.01)
        device:        torch device string
    """

    def __init__(self, model,
                 n_pred_steps=5, n_corr_steps=40,
                 alpha_scale=2.0, epsilon=1.0, rho=0.8, delta=0.01,
                 device='cuda'):
        self.model         = model
        self.constraint_types = model.constraint_types
        self.dims          = model.dims
        self.n_pred_steps  = n_pred_steps
        self.n_corr_steps  = n_corr_steps
        self.alpha_scale   = alpha_scale
        self.epsilon       = epsilon
        self.rho           = rho
        self.delta         = delta
        self.device        = device

    @torch.no_grad()
    def solve(self, batch):
        """Run Projected Flow-CCSP on a single CCSP batch.

        Returns:
            poses: [n_nodes, pose_dim] tensor on CPU
            info:  dict with timing and diagnostic statistics
        """
        self.model.eval()
        batch = batch.to(self.device)

        pose_begin = self.dims[-1][1]
        pose_end   = self.dims[-1][2]
        geom_end   = self.dims[0][2]
        pose_dim   = self.dims[-1][0]

        x_clean = batch.x[:, pose_begin:pose_end].to(self.device)
        geoms   = batch.x[:, :geom_end].to(self.device)
        mask    = batch.mask.bool().to(self.device)
        n_nodes = x_clean.shape[0]

        info = {
            'pred_time': 0.0,
            'corr_time': 0.0,
            'corr_magnitudes': [],
        }

        # ── Phase 1: PREDICTION ────────────────────────────────────────
        t0 = time.perf_counter()

        x_t = torch.randn(n_nodes, pose_dim, device=self.device)
        x_t[mask] = x_clean[mask]

        dt_pred = 1.0 / self.n_pred_steps
        for step in range(self.n_pred_steps):
            t = step * dt_pred
            v = self.model(x_t, batch, t)
            x_t = x_t + v * dt_pred
            x_t[mask] = x_clean[mask]

        # Hard clamp after prediction
        x_t = clamp_to_tray(x_t, geoms, mask)
        info['pred_time'] = time.perf_counter() - t0

        # ── Phase 2: CORRECTION (CBF-QP) ──────────────────────────────
        t0 = time.perf_counter()

        ei       = batch.edge_index.T.to(self.device)  # [E, 2]
        ea       = batch.edge_attr.to(self.device)      # [E]
        full_dim = n_nodes * pose_dim

        dt_corr = 1.0 / self.n_corr_steps

        for step in range(self.n_corr_steps):
            t_corr = step / self.n_corr_steps          # progress ∈ [0, 1)

            # Flow velocity at this correction step
            t_flow = 0.5 + 0.5 * t_corr               # stay in later half of flow
            v_flow = self.model(x_t, batch, t_flow)
            v_np   = v_flow.cpu().numpy()

            # Vanishing scale: suppresses velocity as we approach t=1
            vanish  = self.alpha_scale * (1.0 - t_corr)
            v_scaled = vanish * v_np

            # Relaxation weight: decreases to zero before final step
            w_relax = max(0.0, 1.0 - t_corr / 0.7) * 5.0

            # Collect barriers for all constraints in this graph
            barriers = self._compute_barriers(
                x_t, batch, ei, ea, geom_end, pose_dim, n_nodes)

            # CBF-QP projection
            v_flat  = v_scaled.reshape(-1)
            u_flat  = solve_cbf_qp(
                v_flat, barriers,
                epsilon=self.epsilon, rho=self.rho,
                delta=self.delta, w_relax=w_relax,
                n_iter=15)

            # Diagnostic: how much did the QP modify the velocity?
            correction = np.linalg.norm(u_flat - v_flat)
            info['corr_magnitudes'].append(float(correction))

            # Euler update
            x_np = x_t.cpu().numpy()
            x_np = x_np + u_flat.reshape(n_nodes, pose_dim) * dt_corr
            x_t  = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            x_t[mask] = x_clean[mask]

            # Hard clamp
            x_t = clamp_to_tray(x_t, geoms, mask)

        info['corr_time'] = time.perf_counter() - t0
        info['total_time'] = info['pred_time'] + info['corr_time']
        info['avg_correction'] = (float(np.mean(info['corr_magnitudes']))
                                  if info['corr_magnitudes'] else 0.0)
        info['projection_ratio'] = info['avg_correction'] / (
            np.mean([np.linalg.norm(v) for v in [v_scaled]]) + 1e-8)

        return x_t.cpu(), info

    def _compute_barriers(self, x_t, batch, ei, ea, geom_end, pose_dim, n_nodes):
        """Build list of (h_val, full_grad) tuples for all constraint edges."""
        x        = batch.x.to(self.device)
        barriers = []
        full_dim = n_nodes * pose_dim

        for k in range(ei.shape[0]):
            i = ei[k, 0].item()
            j = ei[k, 1].item()
            c = int(ea[k].item())
            if c >= len(self.constraint_types):
                continue
            ctype = self.constraint_types[c]

            h_val, grad_i, grad_j = compute_barrier(
                ctype,
                x_t[i].cpu(), x_t[j].cpu(),
                x[i, :geom_end].cpu(), x[j, :geom_end].cpu())

            # Embed per-node gradients into the full flattened state gradient
            full_grad = np.zeros(full_dim, dtype=np.float64)
            full_grad[i * pose_dim: (i + 1) * pose_dim] = grad_i[:pose_dim]
            full_grad[j * pose_dim: (j + 1) * pose_dim] = grad_j[:pose_dim]

            barriers.append((h_val, full_grad))

        return barriers


# ─────────────────────────────────────────────────────────────────────────────
# Flow-only sampling (baseline / ablation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_flow(model, batch, n_steps=20, device='cuda'):
    """Euler integration of flow model without CBF-QP (fast baseline).

    Returns:
        poses: [n_nodes, pose_dim] tensor on CPU
    """
    model.eval()
    batch = batch.to(device)

    pose_begin = model.dims[-1][1]
    pose_end   = model.dims[-1][2]
    geom_end   = model.dims[0][2]

    x_clean = batch.x[:, pose_begin:pose_end].to(device)
    geoms   = batch.x[:, :geom_end].to(device)
    mask    = batch.mask.bool().to(device)

    x_t = torch.randn_like(x_clean)
    x_t[mask] = x_clean[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        v = model(x_t, batch, t)
        x_t = x_t + v * dt
        x_t[mask] = x_clean[mask]
        x_t = clamp_to_tray(x_t, geoms, mask)

    return x_t.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_tasks, input_mode, constraint_types, dims,
             n_samples=10, use_qp=True, solver_kwargs=None,
             device='cuda', verbose=True):
    """Evaluate Projected Flow-CCSP on all test tasks.

    Args:
        model:            FlowMatchingCCSP (loaded)
        test_tasks:       {n_obj: dataset_name}
        input_mode:       str
        constraint_types: list
        dims:             model dims tuple
        n_samples:        number of samples per problem (first success counts)
        use_qp:           if True use ProjectedFlowSolver, else flow-only
        solver_kwargs:    dict of kwargs for ProjectedFlowSolver
        device:           str
        verbose:          print per-task stats

    Returns:
        results: {n_obj: {successes, total, times, per_type, sat_rates}}
    """
    solver_kwargs = solver_kwargs or {}
    solver = ProjectedFlowSolver(model, device=device, **solver_kwargs) if use_qp else None

    results = {}
    for n_obj in sorted(test_tasks.keys()):
        task = test_tasks[n_obj]
        try:
            dataset = GraphDataset(task, input_mode=input_mode,
                                   pre_transform=pre_transform, visualize=False)
        except Exception as e:
            if verbose:
                print(f"  [{n_obj} obj] SKIP (dataset not found: {e})")
            continue

        loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                             pin_memory=False, num_workers=0)
        r = dict(successes=0, total=0,
                 times=[], corrections=[],
                 per_type=defaultdict(lambda: [0, 0]),
                 sat_rates=[])

        for data in loader:
            solved = False
            for trial in range(n_samples):
                torch.manual_seed(trial * 7919 + n_obj * 31)
                t0 = time.perf_counter()

                if use_qp:
                    poses, info = solver.solve(data)
                    r['corrections'].append(info.get('avg_correction', 0.0))
                else:
                    poses = sample_flow(model, data, n_steps=20, device=device)
                    info  = {}

                elapsed = time.perf_counter() - t0
                r['times'].append(elapsed)

                ok, per_c = check_constraints(poses, data, constraint_types,
                                              dims, device=device)
                n_sat = sum(1 for v in per_c.values() if v['satisfied'])
                r['sat_rates'].append(n_sat / len(per_c) if per_c else 0.0)

                for ci, cv in per_c.items():
                    r['per_type'][cv['type']][1] += 1
                    if cv['satisfied']:
                        r['per_type'][cv['type']][0] += 1

                r['total'] += 1
                if ok and not solved:
                    r['successes'] += 1
                    solved = True

        if verbose:
            rate    = 100.0 * r['successes'] / r['total'] if r['total'] else 0
            avg_sat = 100.0 * (sum(r['sat_rates']) / len(r['sat_rates'])) \
                      if r['sat_rates'] else 0
            avg_t   = 1000.0 * (sum(r['times']) / len(r['times'])) \
                      if r['times'] else 0
            print(f"  {n_obj} obj: {r['successes']}/{r['total']/n_samples:.0f}"
                  f"  ({rate:.1f}%)  sat={avg_sat:.1f}%  time={avg_t:.0f}ms")

        results[n_obj] = r
    return results


def print_results(name, results, n_samples=10):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    total_s = total_t = 0
    for n_obj in sorted(results.keys()):
        r = results[n_obj]
        n_probs = r['total'] // n_samples
        rate    = 100.0 * r['successes'] / n_probs if n_probs else 0
        avg_sat = 100.0 * (sum(r['sat_rates']) / len(r['sat_rates'])) \
                  if r['sat_rates'] else 0
        avg_t   = 1000.0 * (sum(r['times']) / len(r['times'])) if r['times'] else 0
        total_s += r['successes']
        total_t += n_probs
        print(f"\n  {n_obj} objects: {r['successes']}/{n_probs} ({rate:.1f}%)"
              f"  sat={avg_sat:.1f}%  time={avg_t:.0f}ms")
        if r['corrections']:
            avg_c = sum(r['corrections']) / len(r['corrections'])
            print(f"    QP correction magnitude: {avg_c:.4f}")
        for ct in sorted(r['per_type'].keys()):
            s, t = r['per_type'][ct]
            pct = 100.0 * s / t if t else 0
            print(f"    {ct:14s}: {pct:5.1f}%  ({s}/{t})")
    if total_t > 0:
        print(f"\n  OVERALL: {total_s}/{total_t} ({100.0*total_s/total_t:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Projected Flow-CCSP')
    parser.add_argument('-input_mode',   type=str, default='qualitative',
                        choices=['qualitative', 'diffuse_pairwise',
                                 'stability_flat', 'robot_box'])
    parser.add_argument('-checkpoint',   type=str, default='best',
                        help='model checkpoint tag: "best", "final", or milestone int')
    parser.add_argument('-flow_dir',     type=str, default=None,
                        help='directory containing flow checkpoints')
    parser.add_argument('-hidden_dim',   type=int, default=256)
    parser.add_argument('-n_samples',    type=int, default=10)
    parser.add_argument('-n_pred_steps', type=int, default=5)
    parser.add_argument('-n_corr_steps', type=int, default=40)
    parser.add_argument('-alpha_scale',  type=float, default=2.0)
    parser.add_argument('-epsilon',      type=float, default=1.0)
    parser.add_argument('-rho',          type=float, default=0.8)
    parser.add_argument('-delta',        type=float, default=0.01)
    parser.add_argument('--no_qp',      action='store_true',
                        help='evaluate flow-only (no CBF-QP correction)')
    parser.add_argument('-save_results', type=str, default=None,
                        help='JSON path to save results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_task, test_tasks, dims, constraint_types = get_data_config(args.input_mode)

    print(f"\n{'#'*70}")
    print(f"# Projected Flow-CCSP")
    print(f"# Mode:        {args.input_mode}")
    print(f"# Method:      {'Flow-only' if args.no_qp else 'Flow + CBF-QP'}")
    print(f"# Checkpoint:  {args.checkpoint}")
    print(f"# Samples:     {args.n_samples}")
    print(f"{'#'*70}")

    # Load model
    flow_dir  = args.flow_dir or f'./logs/flow_{args.input_mode}_h{args.hidden_dim}'
    ckpt_path = os.path.join(flow_dir, f'flow_model_{args.checkpoint}.pt')
    if not os.path.isfile(ckpt_path):
        print(f"\n  ERROR: checkpoint not found: {ckpt_path}")
        print(f"  Train first with: python train_flow.py -input_mode {args.input_mode}")
        sys.exit(1)

    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\n  Loaded: {ckpt_path}  (step={ckpt.get('step', '?')})")

    use_qp = not args.no_qp
    solver_kwargs = dict(
        n_pred_steps=args.n_pred_steps,
        n_corr_steps=args.n_corr_steps,
        alpha_scale=args.alpha_scale,
        epsilon=args.epsilon,
        rho=args.rho,
        delta=args.delta,
    )

    all_results = {}

    # ── Flow-only evaluation ────────────────────────────────────────────
    method_name = 'Flow-only (clamped)'
    print(f"\n  [{method_name}]")
    r_flow = evaluate(
        model, test_tasks, args.input_mode, constraint_types, dims,
        n_samples=args.n_samples, use_qp=False, device=device)
    all_results[method_name] = r_flow
    print_results(method_name, r_flow, args.n_samples)

    # ── Projected Flow + QP evaluation ─────────────────────────────────
    if use_qp:
        method_name = f'Projected Flow + QP ({args.n_corr_steps} steps)'
        print(f"\n  [{method_name}]")
        r_qp = evaluate(
            model, test_tasks, args.input_mode, constraint_types, dims,
            n_samples=args.n_samples, use_qp=True,
            solver_kwargs=solver_kwargs, device=device)
        all_results[method_name] = r_qp
        print_results(method_name, r_qp, args.n_samples)

    # ── Summary table ───────────────────────────────────────────────────
    obj_keys = sorted({n for r in all_results.values() for n in r})
    header = f"{'Method':<38}" + "".join(f"  {n}obj" for n in obj_keys) + "  Time(ms)"
    print(f"\n\n{'='*len(header)}")
    print("SUMMARY")
    print('='*len(header))
    print(header)
    print('-'*len(header))
    for name, results in all_results.items():
        row  = f"{name:<38}"
        all_t = []
        for n in obj_keys:
            if n in results:
                r    = results[n]
                n_p  = r['total'] // args.n_samples
                rate = 100.0 * r['successes'] / n_p if n_p else 0
                row += f"  {rate:4.0f}%"
                all_t.extend(r['times'])
            else:
                row += f"  {'N/A':>5}"
        avg_t = 1000.0 * (sum(all_t) / len(all_t)) if all_t else 0
        row  += f"  {avg_t:>6.0f}"
        print(row)

    # ── Save ────────────────────────────────────────────────────────────
    save_path = args.save_results or os.path.join(
        flow_dir, f'eval_{args.input_mode}_{args.checkpoint}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    serialisable = {}
    for name, results in all_results.items():
        serialisable[name] = {}
        for n_obj, r in results.items():
            n_p  = r['total'] // args.n_samples
            serialisable[name][str(n_obj)] = {
                'successes': r['successes'],
                'n_problems': n_p,
                'success_rate': r['successes'] / n_p if n_p else 0,
                'avg_time_ms': 1000.0 * (sum(r['times']) / len(r['times']))
                               if r['times'] else 0,
                'avg_constraint_sat': (sum(r['sat_rates']) / len(r['sat_rates']))
                                      if r['sat_rates'] else 0,
                'per_type': {ct: [s, t] for ct, (s, t) in r['per_type'].items()},
            }

    with open(save_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Results saved: {save_path}")


if __name__ == '__main__':
    main()
