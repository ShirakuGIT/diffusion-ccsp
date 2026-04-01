"""
FMIP vs CFM Comparison Evaluation
===================================
Compares five conditions on feasibility, diversity, runtime, and repair gain.

Conditions
----------
  A  Baseline CFM        — plain Euler rollout, no guidance, no mode
  B  Guided CFM          — CFM + constraint-energy gradient guidance
  F1 FMIP v1             — FMIP model, K-mode sampling, guided rollout
  F2 FMIP v2             — FMIP v1 + short local repair
  Ab Ablation            — FMIP model, K=1 (single mode), guided rollout

Metrics
-------
  feasibility:   fraction of scenes with all constraints satisfied (h ≥ 0)
  violation_sev: mean violation magnitude across violated constraints
  diversity:     mean pairwise L2 distance across K pose outputs
  n_clusters:    number of distinct mode clusters (k-means, k=n_modes)
  success_k:     best-of-K feasibility (at least one of K candidates satisfies all)
  repair_gain:   fraction of infeasible scenes repaired by the local repair stage
  runtime:       wall-clock inference time per scene (ms)

Usage
-----
  python eval_fmip.py
  python eval_fmip.py --cfm_ckpt logs/flow_qualitative_h256/flow_model_best.pt
  python eval_fmip.py --fmip_ckpt logs/fmip_qualitative_h256_m4/flow_model_best.pt \\
                      --n_modes 4 --K 4 --lambda_c 0.5 --n_repair 5 --n_scenes 50
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from flow_matching.train_flow import FlowMatchingCCSP, get_data_config
from flow_matching.fix_and_eval import clamp_to_tray, compute_barrier
from flow_matching.train_fmip import FlowMatchingCCSP_FMIP, _sample_fmip_simple

try:
    from flow_matching.train_flow_v4 import barrier_violation_loss
except ImportError:
    barrier_violation_loss = None


# ═══════════════════════════════════════════════════════════════════════════════
# Differentiable constraint energy  (for inference-time guidance)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_constraint_energy(poses, batch, constraint_types, device):
    """Differentiable violation energy — wraps barrier_violation_loss.

    Returns a scalar tensor with autograd graph through `poses`.
    Falls back to a minimal inlined implementation if import failed.
    """
    if barrier_violation_loss is not None:
        return barrier_violation_loss(poses, batch, constraint_types, device)

    # Inline fallback (subset of constraint types used in qualitative mode)
    x = batch.x.to(device)
    edge_index = batch.edge_index.T.to(device)
    edge_attr  = batch.edge_attr.to(device)
    geom_end   = 2

    total = torch.tensor(0.0, device=device)
    n     = 0
    for ei in range(edge_index.shape[0]):
        i  = int(edge_index[ei, 0].item())
        j  = int(edge_index[ei, 1].item())
        ci = int(edge_attr[ei].item())
        if ci >= len(constraint_types):
            continue
        ct = constraint_types[ci]
        pi, pj = poses[i], poses[j]
        gi, gj = x[i, :geom_end], x[j, :geom_end]
        wi, hi_g = gi[0], gi[1]
        wj, hj_g = gj[0], gj[1]

        if ct == 'cfree':
            dx = (pi[0] - pj[0]).abs() - (wi + wj)
            dy = (pi[1] - pj[1]).abs() - (hi_g + hj_g)
            h  = torch.stack([dx, dy]).max()
        elif ct == 'left-of':
            h  = pj[0] - pi[0] - (wi + wj) * 0.5
        elif ct == 'top-of':
            h  = pi[1] - pj[1] - (hi_g + hj_g) * 0.5
        else:
            continue   # skip unsupported types
        viol = F.relu(-h)
        total = total + viol
        n    += 1
    return total / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Guided rollout
# ═══════════════════════════════════════════════════════════════════════════════

def guided_rollout(model, batch, n_steps, z, lambda_c, constraint_types, device):
    """Euler rollout with constraint-energy gradient guidance.

    x_{t+dt} = x_t + (v_θ(x_t, t, z) - λ · ∇_x E(x_t)) · dt

    Args:
        model:            FlowMatchingCCSP or FlowMatchingCCSP_FMIP
        batch:            PyG DataBatch (single scene or batched)
        n_steps:          ODE steps
        z:                LongTensor [n_scenes] or None
        lambda_c:         guidance strength
        constraint_types: list of constraint type strings
        device:           'cuda' or 'cpu'

    Returns:
        x_t:  [N, pose_dim] final poses (on device)
    """
    batch = batch.to(device)

    pose_begin = model.dims[-1][1]
    pose_end   = model.dims[-1][2]
    geom_end   = model.dims[0][2]
    pose_dim   = model.dims[-1][0]

    x_clean = batch.x[:, pose_begin:pose_end].to(device)
    geoms   = batch.x[:, :geom_end].to(device)
    mask    = batch.mask.bool().to(device)

    x_t = torch.randn_like(x_clean)
    x_t[mask] = x_clean[mask]

    dt      = 1.0 / n_steps
    has_z   = hasattr(model, 'n_modes')

    for step in range(n_steps):
        t_val = step * dt

        # Velocity from model (no grad through model weights)
        with torch.no_grad():
            if has_z:
                v = model(x_t, batch, t_val, z=z)
            else:
                v = model(x_t, batch, t_val)

        # Constraint energy gradient (grad w.r.t. pose only)
        if lambda_c > 0:
            x_for_grad = x_t.detach().clone().requires_grad_(True)
            E = compute_constraint_energy(
                x_for_grad, batch, constraint_types, device)
            if E.item() > 1e-9:
                grad_E = torch.autograd.grad(E, x_for_grad)[0]
                grad_E[mask] = 0.0  # never perturb fixed nodes
            else:
                grad_E = torch.zeros_like(x_t)
        else:
            grad_E = torch.zeros_like(x_t)

        # Guided step
        x_t = x_t + (v - lambda_c * grad_E) * dt
        x_t[mask] = x_clean[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# K-mode sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_k_modes(model, batch, K, n_steps, lambda_c, constraint_types, device):
    """Sample K modes, score by constraint energy, return all and the best.

    For FMIP model: uses z=0,...,K-1 (cycling if K > n_modes).
    For plain CFM:  runs K independent rollouts with different random seeds.

    Returns:
        poses_all:   list of K pose tensors, each [N, pose_dim]
        energies:    list of K energy values (floats)
        best_idx:    index into poses_all of the best candidate
    """
    has_z   = hasattr(model, 'n_modes')
    n_modes = model.n_modes if has_z else 1

    batch_vec = batch.batch.to(device) if hasattr(batch, 'batch') else None
    n_scenes  = (int(batch_vec.max().item()) + 1) if batch_vec is not None else 1

    poses_all = []
    energies  = []

    for k in range(K):
        if has_z:
            z_k = torch.full((n_scenes,), k % n_modes, dtype=torch.long, device=device)
        else:
            z_k = None
            torch.manual_seed(k * 42)  # different initial noise per trial

        if lambda_c > 0:
            poses_k = guided_rollout(
                model, batch, n_steps, z_k, lambda_c, constraint_types, device)
        else:
            with torch.no_grad():
                poses_k = _sample_fmip_simple(model, batch, n_steps, z=z_k, device=device) \
                          if has_z else _cfm_sample(model, batch, n_steps, device)

        # Score by energy (no grad needed)
        with torch.no_grad():
            x_ev = poses_k.detach().clone().requires_grad_(False)
            # Use numpy-level scoring for clean numbers
            energy = _scene_energy_np(poses_k.cpu().numpy(), batch, constraint_types)

        poses_all.append(poses_k)
        energies.append(energy)

    best_idx = int(np.argmin(energies))
    return poses_all, energies, best_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Local repair stage
# ═══════════════════════════════════════════════════════════════════════════════

def repair_poses(poses_np, geoms_np, edge_index_np, edge_attr_np,
                 constraint_types, mask_np, n_steps=5, step_size=0.05):
    """Short analytic gradient repair using compute_barrier.

    Only moves free (non-masked) nodes.
    Clamps to tray after each step.

    Args:
        poses_np:       [N, pose_dim] numpy array
        geoms_np:       [N, geom_dim] numpy array
        edge_index_np:  [E, 2] numpy int array
        edge_attr_np:   [E]    numpy int array
        constraint_types: list of strings
        mask_np:        [N] bool numpy array (True = fixed)
        n_steps:        number of repair iterations
        step_size:      gradient step size

    Returns:
        repaired poses [N, pose_dim] numpy array
    """
    poses = poses_np.copy()
    n_nodes = poses.shape[0]
    pose_dim = poses.shape[1]

    for _ in range(n_steps):
        grad = np.zeros_like(poses)
        for ei in range(edge_index_np.shape[0]):
            i  = int(edge_index_np[ei, 0])
            j  = int(edge_index_np[ei, 1])
            ci = int(edge_attr_np[ei])
            if ci >= len(constraint_types):
                continue

            # Convert to tensors for compute_barrier interface
            pi = torch.tensor(poses[i], dtype=torch.float32)
            pj = torch.tensor(poses[j], dtype=torch.float32)
            gi = torch.tensor(geoms_np[i], dtype=torch.float32)
            gj = torch.tensor(geoms_np[j], dtype=torch.float32)

            h_val, g_i, g_j = compute_barrier(constraint_types[ci], pi, pj, gi, gj)
            if h_val < 0:  # violated — apply repair step
                grad[i, :len(g_i)] += step_size * np.array(g_i)[:pose_dim]
                grad[j, :len(g_j)] += step_size * np.array(g_j)[:pose_dim]

        # Apply gradient and clamp
        for i in range(n_nodes):
            if mask_np[i]:
                continue
            poses[i] += grad[i]
            w = geoms_np[i, 0]; h = geoms_np[i, 1]
            poses[i, 0] = np.clip(poses[i, 0], w + 0.02, 2.0 - w - 0.02)
            poses[i, 1] = np.clip(poses[i, 1], h + 0.02, 2.0 - h - 0.02)

    return poses


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _scene_energy_np(poses_np, batch, constraint_types):
    """Numpy constraint energy (sum of violations). Non-differentiable."""
    edge_index = batch.edge_index.T.numpy() if hasattr(batch.edge_index, 'numpy') \
                 else batch.edge_index.T.cpu().numpy()
    edge_attr  = batch.edge_attr.numpy() if hasattr(batch.edge_attr, 'numpy') \
                 else batch.edge_attr.cpu().numpy()
    x_np       = batch.x.numpy() if hasattr(batch.x, 'numpy') \
                 else batch.x.cpu().numpy()
    geoms_np   = x_np[:, :2]

    total = 0.0
    for ei in range(edge_index.shape[0]):
        i, j = int(edge_index[ei, 0]), int(edge_index[ei, 1])
        ci   = int(edge_attr[ei])
        if ci >= len(constraint_types):
            continue
        pi = torch.tensor(poses_np[i], dtype=torch.float32)
        pj = torch.tensor(poses_np[j], dtype=torch.float32)
        gi = torch.tensor(geoms_np[i], dtype=torch.float32)
        gj = torch.tensor(geoms_np[j], dtype=torch.float32)
        h, _, _ = compute_barrier(constraint_types[ci], pi, pj, gi, gj)
        total += max(0.0, -h)
    return total


def score_scene(poses_np, batch, constraint_types):
    """Full per-scene metrics.

    Returns dict with:
        feasible:     bool (all h >= 0)
        n_violated:   int
        n_total:      int
        viol_sev:     mean |violation| for violated constraints
    """
    edge_index = batch.edge_index.T.cpu().numpy()
    edge_attr  = batch.edge_attr.cpu().numpy()
    geoms_np   = batch.x.cpu().numpy()[:, :2]

    violations = []
    for ei in range(edge_index.shape[0]):
        i, j = int(edge_index[ei, 0]), int(edge_index[ei, 1])
        ci   = int(edge_attr[ei])
        if ci >= len(constraint_types):
            continue
        pi = torch.tensor(poses_np[i], dtype=torch.float32)
        pj = torch.tensor(poses_np[j], dtype=torch.float32)
        gi = torch.tensor(geoms_np[i], dtype=torch.float32)
        gj = torch.tensor(geoms_np[j], dtype=torch.float32)
        h, _, _ = compute_barrier(constraint_types[ci], pi, pj, gi, gj)
        violations.append(h)

    if not violations:
        return {'feasible': True, 'n_violated': 0, 'n_total': 0, 'viol_sev': 0.0}

    n_violated = sum(1 for h in violations if h < -0.02)
    sev = float(np.mean([-h for h in violations if h < -0.02])) if n_violated > 0 else 0.0
    return {
        'feasible':   n_violated == 0,
        'n_violated': n_violated,
        'n_total':    len(violations),
        'viol_sev':   sev,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Diversity metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_diversity(poses_list, mask):
    """Pairwise L2 diversity and cluster count across K pose tensors.

    Args:
        poses_list: list of K tensors, each [N, pose_dim]
        mask:       [N] bool tensor (True = fixed — exclude from diversity)

    Returns:
        mean_pairwise_dist: average L2 between pairs of outputs (free nodes only)
        n_clusters:         number of distinct clusters via threshold-based grouping
    """
    if len(poses_list) < 2:
        return 0.0, 1

    free  = ~mask
    vecs  = [p[free].cpu().numpy().flatten() for p in poses_list]
    K     = len(vecs)

    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(np.linalg.norm(vecs[i] - vecs[j]))

    mean_dist = float(np.mean(dists))

    # Threshold-based cluster count: two samples in same cluster if dist < mean_dist/2
    threshold  = mean_dist / 2.0
    cluster_id = list(range(K))
    for i in range(K):
        for j in range(i + 1, K):
            if np.linalg.norm(vecs[i] - vecs[j]) < threshold:
                # Merge clusters
                old_id = cluster_id[j]
                new_id = cluster_id[i]
                cluster_id = [new_id if c == old_id else c for c in cluster_id]

    n_clusters = len(set(cluster_id))
    return mean_dist, n_clusters


# ═══════════════════════════════════════════════════════════════════════════════
# CFM baseline sampler (plain, no mode)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _cfm_sample(model, batch, n_steps, device):
    """Plain Euler rollout for FlowMatchingCCSP (no mode variable)."""
    batch = batch.to(device)

    pose_begin = model.dims[-1][1]
    pose_end   = model.dims[-1][2]
    geom_end   = model.dims[0][2]
    pose_dim   = model.dims[-1][0]

    x_clean = batch.x[:, pose_begin:pose_end].to(device)
    geoms   = batch.x[:, :geom_end].to(device)
    mask    = batch.mask.bool().to(device)

    x_t = torch.randn_like(x_clean)
    x_t[mask] = x_clean[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        v   = model(x_t, batch, step * dt)
        x_t = x_t + v * dt
        x_t[mask] = x_clean[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
    return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# Per-condition runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_condition(name, model, batch, K, n_steps, lambda_c, n_repair,
                  constraint_types, device):
    """Run a single inference condition, return metrics dict.

    name: 'A' | 'B' | 'F1' | 'F2' | 'Ab'
    """
    is_fmip  = hasattr(model, 'n_modes')
    n_scenes = (int(batch.batch.max().item()) + 1) if hasattr(batch, 'batch') else 1
    mask     = batch.mask.bool().to(device)

    t0 = time.perf_counter()

    if name == 'A':
        # Plain CFM — single rollout, no guidance
        if is_fmip:
            with torch.no_grad():
                poses = _sample_fmip_simple(model, batch, n_steps, z=None, device=device)
        else:
            poses = _cfm_sample(model, batch, n_steps, device)
        poses_all = [poses]
        energies  = [_scene_energy_np(poses.cpu().numpy(), batch, constraint_types)]
        best_idx  = 0

    elif name == 'B':
        # CFM + guidance, single rollout
        poses = guided_rollout(model, batch, n_steps, None, lambda_c,
                               constraint_types, device)
        poses_all = [poses]
        energies  = [_scene_energy_np(poses.cpu().numpy(), batch, constraint_types)]
        best_idx  = 0

    elif name in ('F1', 'Ab'):
        # FMIP: K-mode sampling with guided rollout
        effective_K = K if name == 'F1' else 1
        effective_lc = lambda_c
        poses_all, energies, best_idx = sample_k_modes(
            model, batch, effective_K, n_steps, effective_lc,
            constraint_types, device)

    elif name == 'F2':
        # FMIP v1 + repair
        poses_all, energies, best_idx = sample_k_modes(
            model, batch, K, n_steps, lambda_c, constraint_types, device)

    else:
        raise ValueError(f"Unknown condition: {name}")

    t_rollout = (time.perf_counter() - t0) * 1000  # ms

    # Score best candidate
    best_poses = poses_all[best_idx]
    sc_before  = score_scene(best_poses.cpu().numpy(), batch, constraint_types)

    # Repair stage (only for F2)
    sc_after   = sc_before
    t_repair   = 0.0
    if name == 'F2' and not sc_before['feasible'] and n_repair > 0:
        t1 = time.perf_counter()
        x_np    = batch.x.cpu().numpy()
        geoms_n = x_np[:, :2]
        ei_np   = batch.edge_index.T.cpu().numpy()
        ea_np   = batch.edge_attr.cpu().numpy()
        mask_np = batch.mask.bool().cpu().numpy()

        repaired = repair_poses(
            best_poses.cpu().numpy(), geoms_n, ei_np, ea_np,
            constraint_types, mask_np, n_steps=n_repair)
        sc_after = score_scene(repaired, batch, constraint_types)
        t_repair = (time.perf_counter() - t1) * 1000

    # Diversity
    mean_pairwise, n_clusters = compute_diversity(poses_all, mask.cpu())

    # Best-of-K feasibility
    feasible_any = any(
        score_scene(p.cpu().numpy(), batch, constraint_types)['feasible']
        for p in poses_all
    )

    return {
        'feasible':         int(sc_before['feasible']),
        'feasible_any_k':   int(feasible_any),
        'feasible_repaired':int(sc_after['feasible']),
        'n_violated':       sc_before['n_violated'],
        'n_total':          sc_before['n_total'],
        'viol_sev':         sc_before['viol_sev'],
        'viol_sev_repaired':sc_after['viol_sev'],
        'diversity':        mean_pairwise,
        'n_clusters':       n_clusters,
        'runtime_ms':       t_rollout,
        'repair_ms':        t_repair,
        'best_energy':      energies[best_idx],
        'K_used':           len(poses_all),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_cfm_model(ckpt_path, dims, constraint_types, device):
    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=256, constraint_types=constraint_types,
        normalize=True, device=device,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  Loaded CFM: {ckpt_path}")
    return model


def load_fmip_model(ckpt_path, dims, constraint_types, n_modes, device):
    model = FlowMatchingCCSP_FMIP(
        dims=dims, hidden_dim=256, constraint_types=constraint_types,
        normalize=True, device=device, n_modes=n_modes,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get('model_state_dict', ck)
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [FMIP] {len(missing)} missing keys (new parameters)")
    model.eval()
    print(f"  Loaded FMIP: {ckpt_path}  (n_modes={n_modes})")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate(records):
    """Aggregate a list of per-scene metric dicts → mean floats."""
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


def run_evaluation(models_dict, conditions, loader, constraint_types,
                   K, n_steps, lambda_c, n_repair, device, n_scenes):
    """Run all conditions × all scenes.

    models_dict: {condition_name: model or None}
    conditions:  list of condition names to run
    """
    results = defaultdict(list)

    for sc_idx, batch in enumerate(loader):
        if sc_idx >= n_scenes:
            break
        batch = batch.to(device)

        for cond in conditions:
            model = models_dict.get(cond)
            if model is None:
                continue
            try:
                m = run_condition(
                    cond, model, batch, K, n_steps, lambda_c, n_repair,
                    constraint_types, device)
                results[cond].append(m)
            except Exception as e:
                print(f"  [scene {sc_idx}] condition {cond} failed: {e}")

        if (sc_idx + 1) % 10 == 0:
            print(f"  {sc_idx+1}/{n_scenes} scenes done …", flush=True)

    return {cond: _aggregate(recs) for cond, recs in results.items()}


def print_table(summary):
    """Pretty-print comparison table."""
    cols = ['feasible', 'feasible_any_k', 'feasible_repaired',
            'viol_sev', 'diversity', 'n_clusters', 'runtime_ms']

    header = f"{'Cond':6s} " + " ".join(f"{c:18s}" for c in cols)
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for cond, row in sorted(summary.items()):
        vals = [f"{row.get(c, float('nan')):.3f}" for c in cols]
        print(f"{cond:6s} " + " ".join(f"{v:18s}" for v in vals))
    print("═" * len(header))


def main():
    parser = argparse.ArgumentParser(description='FMIP vs CFM evaluation')
    parser.add_argument('--cfm_ckpt',  type=str,
                        default='logs/flow_qualitative_h256/flow_model_best.pt')
    parser.add_argument('--fmip_ckpt', type=str, default=None,
                        help='Path to trained FMIP checkpoint (optional)')
    parser.add_argument('--n_modes',   type=int,   default=4)
    parser.add_argument('--K',         type=int,   default=4,
                        help='K candidates per scene for FMIP')
    parser.add_argument('--n_steps',   type=int,   default=20,
                        help='ODE rollout steps')
    parser.add_argument('--lambda_c',  type=float, default=0.5,
                        help='Constraint guidance strength')
    parser.add_argument('--n_repair',  type=int,   default=5,
                        help='Repair iterations (0 = skip repair)')
    parser.add_argument('--n_scenes',  type=int,   default=50,
                        help='Number of test scenes to evaluate')
    parser.add_argument('--out',       type=str,
                        default='logs/fmip_eval_results.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nFMIP Evaluation  (device={device})")
    print(f"  K={args.K}  lambda_c={args.lambda_c}  "
          f"n_steps={args.n_steps}  n_repair={args.n_repair}  "
          f"n_scenes={args.n_scenes}")

    _, _, dims, constraint_types = get_data_config('qualitative')

    # ── Load models ────────────────────────────────────────────────────────────
    cfm_model = None
    if os.path.exists(args.cfm_ckpt):
        cfm_model = load_cfm_model(args.cfm_ckpt, dims, constraint_types, device)
    else:
        print(f"  WARNING: CFM checkpoint not found: {args.cfm_ckpt}")

    fmip_model = None
    if args.fmip_ckpt and os.path.exists(args.fmip_ckpt):
        fmip_model = load_fmip_model(
            args.fmip_ckpt, dims, constraint_types, args.n_modes, device)
    elif args.fmip_ckpt:
        print(f"  WARNING: FMIP checkpoint not found: {args.fmip_ckpt}")
        print(f"  → Train first:  python train_fmip.py")

    # ── Determine which conditions to run ────────────────────────────────────
    models_dict = {
        'A':  cfm_model,
        'B':  cfm_model,
        'F1': fmip_model,
        'F2': fmip_model,
        'Ab': fmip_model,
    }
    conditions = [c for c, m in models_dict.items() if m is not None]
    if not conditions:
        print("  ERROR: no models loaded. Provide at least --cfm_ckpt or --fmip_ckpt.")
        return

    print(f"  Conditions to run: {conditions}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_task, test_tasks, _, _ = get_data_config('qualitative')
    # Pick smallest test split for speed
    smallest_task = sorted(test_tasks.items())[0][1]
    ds = GraphDataset(smallest_task, input_mode='qualitative',
                      pre_transform=pre_transform, visualize=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"  Dataset: {smallest_task}  ({len(ds)} scenes)")

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\n  Running {len(conditions)} condition(s) × {args.n_scenes} scenes …")
    summary = run_evaluation(
        models_dict, conditions, loader, constraint_types,
        K=args.K, n_steps=args.n_steps, lambda_c=args.lambda_c,
        n_repair=args.n_repair, device=device, n_scenes=args.n_scenes)

    # ── Results ───────────────────────────────────────────────────────────────
    print_table(summary)

    # Interpret results
    print("\n  Interpretation:")
    if 'A' in summary and 'F1' in summary:
        delta_feas = summary['F1']['feasible'] - summary['A']['feasible']
        delta_any  = summary['F1']['feasible_any_k'] - summary['A']['feasible']
        print(f"  FMIP v1 vs CFM baseline:")
        print(f"    Feasibility delta:      {delta_feas:+.3f}")
        print(f"    Best-of-K delta:        {delta_any:+.3f}")
        print(f"    Diversity (FMIP):       {summary['F1']['diversity']:.4f}")
        print(f"    Diversity (CFM):        {summary['A']['diversity']:.4f}")
    if 'F2' in summary:
        rep = summary['F2']['feasible_repaired'] - summary['F2']['feasible']
        print(f"  Repair gain:              {rep:+.3f}")
    if 'B' in summary and 'F1' in summary:
        g_vs_b = summary['F1']['feasible'] - summary['B']['feasible']
        print(f"  Mode branching gain:      {g_vs_b:+.3f}  (FMIP v1 vs guided-CFM)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'args': vars(args),
            'conditions': conditions,
            'summary': summary,
        }, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    # Verdict
    print("\n  ── Pilot Verdict ─────────────────────────────────────────────")
    if 'F1' in summary and 'A' in summary:
        fmip_better = summary['F1']['feasible_any_k'] > summary['A']['feasible']
        diverse     = summary['F1']['diversity'] > summary['A']['diversity'] * 1.2
        if fmip_better and diverse:
            print("  POSITIVE: FMIP shows better best-of-K feasibility AND diversity.")
        elif fmip_better:
            print("  PARTIAL: FMIP better feasibility but not clearly more diverse.")
        elif diverse:
            print("  PARTIAL: FMIP more diverse but no feasibility gain.")
        else:
            print("  NEGATIVE: FMIP does not improve over CFM baseline.")
    else:
        print("  Incomplete: need both CFM and FMIP checkpoints for full verdict.")


if __name__ == '__main__':
    main()
