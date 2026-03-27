"""
PCFM: Physics-Constrained Flow Matching for CCSP
==================================================
Inference-time Gauss-Newton projection on top of the pretrained flow v1 model.
No retraining needed — projects intermediate flow states onto the constraint manifold.

Based on: "Physics-Constrained Flow Matching" (NeurIPS 2025)
Key equation: x_proj = x - J^T (J J^T + λI)^{-1} h(x)

Uses analytic Jacobians (each constraint touches ≤2 nodes in xy only) for speed.

Usage:
    python eval_pcfm.py
    python eval_pcfm.py --n_steps 50 --n_proj_iters 3 --alpha 0.8
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'networks'))

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from train_flow import FlowMatchingCCSP, get_data_config
from fix_and_eval import clamp_to_tray, check_constraints, compute_barrier


# ═══════════════════════════════════════════════════════════════════════════════
# Analytic Gauss-Newton projection (fast, no autograd)
# ═══════════════════════════════════════════════════════════════════════════════

def gauss_newton_project(poses_np, edge_index, edge_attr, geoms_np,
                         constraint_types, free_mask,
                         n_iters=3, damping=1e-4, pose_dim=4):
    """
    Project poses onto constraint manifold using Gauss-Newton with analytic Jacobians.
        x_proj = x - J^T (J J^T + λI)^{-1} h(x)

    Uses compute_barrier() from fix_and_eval.py for h and gradients.
    Only projects violated constraints (h < 0 in barrier convention → violated).

    Args:
        poses_np: [N, pose_dim] numpy array
        edge_index: [E, 2] numpy array
        edge_attr: [E] numpy array (constraint type indices)
        geoms_np: [N, geom_dim] numpy array
        constraint_types: list of constraint type strings
        free_mask: [N] bool numpy array (True = free node)
        n_iters: GN iterations
        damping: regularization
        pose_dim: dimension of pose (4 for qualitative)

    Returns:
        projected poses [N, pose_dim] numpy array
    """
    n_nodes = poses_np.shape[0]
    free_indices = np.where(free_mask)[0]
    n_free = len(free_indices)
    if n_free == 0:
        return poses_np.copy()

    # Map: node index → position in free vector
    node_to_free = {int(idx): k for k, idx in enumerate(free_indices)}
    free_dim = n_free * pose_dim

    poses = poses_np.copy()

    for iteration in range(n_iters):
        # Collect violated constraints
        h_vals = []
        J_rows = []

        for ei in range(edge_index.shape[0]):
            i = int(edge_index[ei, 0])
            j = int(edge_index[ei, 1])
            ctype_idx = int(edge_attr[ei])
            if ctype_idx >= len(constraint_types):
                continue
            ctype = constraint_types[ctype_idx]

            pose_i = torch.tensor(poses[i], dtype=torch.float32)
            pose_j = torch.tensor(poses[j], dtype=torch.float32)
            geom_i = torch.tensor(geoms_np[i], dtype=torch.float32)
            geom_j = torch.tensor(geoms_np[j], dtype=torch.float32)

            h_val, grad_i, grad_j = compute_barrier(
                ctype, pose_i, pose_j, geom_i, geom_j)

            # barrier convention: h >= 0 satisfied, h < 0 violated
            # We want to project violated ones (h < 0) to h = 0
            if h_val >= 0:
                continue

            # Build Jacobian row (only for free variables)
            J_row = np.zeros(free_dim)
            if i in node_to_free:
                k = node_to_free[i]
                J_row[k*pose_dim:(k+1)*pose_dim] = grad_i[:pose_dim]
            if j in node_to_free:
                k = node_to_free[j]
                J_row[k*pose_dim:(k+1)*pose_dim] = grad_j[:pose_dim]

            h_vals.append(h_val)  # negative value (violated)
            J_rows.append(J_row)

        if len(h_vals) == 0:
            break  # all satisfied

        m = len(h_vals)
        h = np.array(h_vals)  # [m], negative values
        J = np.array(J_rows)  # [m, free_dim]

        # We want to push h from negative to 0:
        # x_new = x - J^T (J J^T + λI)^{-1} * h
        # Since h < 0 and grad points toward increasing h,
        # -J^T * (negative h) = +J^T * |h| → moves x to increase h toward 0
        JJT = J @ J.T + damping * np.eye(m)

        try:
            z = np.linalg.solve(JJT, h)  # h is negative
            dx = -J.T @ z  # [free_dim]
        except np.linalg.LinAlgError:
            dx = -J.T @ np.linalg.lstsq(JJT, h, rcond=None)[0]

        # Apply correction
        dx_reshaped = dx.reshape(n_free, pose_dim)
        for k, idx in enumerate(free_indices):
            poses[idx] += dx_reshaped[k]

    return poses


# ═══════════════════════════════════════════════════════════════════════════════
# PCFM Flow Sampler
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_pcfm(model, batch, constraint_types, n_steps=20,
                n_proj_iters=3, alpha=0.5, final_proj_iters=5,
                damping=1e-4, proj_start=0.3, device='cuda'):
    """
    PCFM sampler: Euler flow integration + Gauss-Newton projection.

    At each step:
        1. x_t+dt = x_t + v_θ(x_t, t) * dt     (Euler step)
        2. x_proj = GaussNewton(x_t+dt)          (project onto constraints)
        3. x_t+dt = x_t+dt + alpha*(x_proj - x_t+dt)  (relaxed correction)

    At t=1: run final_proj_iters of full Gauss-Newton projection.
    """
    model.eval()
    batch = batch.to(device)

    x = batch.x.to(device)
    pose_dim = model.dims[-1][0]
    pose_begin = model.dims[-1][1]
    pose_end = model.dims[-1][2]
    geom_end = model.dims[0][2]
    n_nodes = x.shape[0]
    mask = batch.mask.bool().to(device)
    clean_poses = x[:, pose_begin:pose_end]
    geoms = x[:, :geom_end]

    # Precompute numpy arrays for GN projection
    edge_index_np = batch.edge_index.T.cpu().numpy()
    edge_attr_np = batch.edge_attr.cpu().numpy()
    geoms_np = geoms.cpu().numpy()
    free_mask = (~mask).cpu().numpy()

    # Start from noise
    x_t = torch.randn(n_nodes, pose_dim, device=device)
    x_t[mask] = clean_poses[mask]

    dt = 1.0 / n_steps

    for step in range(n_steps):
        t = step * dt

        # 1. Euler step
        v = model(x_t, batch, t)
        x_t = x_t + v * dt
        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        # 2. Gauss-Newton projection (after initial noise phase)
        if t >= proj_start and alpha > 0:
            poses_np = x_t.cpu().numpy()
            poses_proj = gauss_newton_project(
                poses_np, edge_index_np, edge_attr_np, geoms_np,
                constraint_types, free_mask,
                n_iters=n_proj_iters, damping=damping, pose_dim=pose_dim)

            x_proj = torch.tensor(poses_proj, dtype=torch.float32, device=device)

            # 3. Relaxed correction
            x_t = x_t + alpha * (x_proj - x_t)
            x_t[mask] = clean_poses[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    # Final projection: iterate more aggressively
    if final_proj_iters > 0:
        poses_np = x_t.cpu().numpy()
        poses_proj = gauss_newton_project(
            poses_np, edge_index_np, edge_attr_np, geoms_np,
            constraint_types, free_mask,
            n_iters=final_proj_iters, damping=damping, pose_dim=pose_dim)
        x_t = torch.tensor(poses_proj, dtype=torch.float32, device=device)
        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_method(model, constraint_types, sampler_fn, input_mode='qualitative',
                    n_samples=10, device='cuda', label=""):
    """Generic evaluator for any sampler function."""
    test_tasks = {i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
                  for i in range(2, 6)}

    results = {}
    for n_obj, task_name in test_tasks.items():
        print(f"  {n_obj} objects...", end=" ", flush=True)
        dataset = GraphDataset(task_name, input_mode=input_mode,
                               pre_transform=pre_transform, visualize=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        stats = {'successes': 0, 'total': 0, 'times': [],
                 'per_type': defaultdict(lambda: [0, 0]),
                 'constraint_sat_rates': []}
        scene_succeeded = set()
        scene_first_try = set()
        n_scenes = 0

        for scene_idx, data in enumerate(loader):
            n_scenes += 1
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj * 100 + scene_idx)

                t0 = time.time()
                poses = sampler_fn(model, data, device=device)
                elapsed = time.time() - t0
                stats['times'].append(elapsed)

                all_ok, per_c = check_constraints(
                    poses, data, constraint_types, device)

                stats['total'] += 1
                if all_ok:
                    stats['successes'] += 1
                    scene_succeeded.add(scene_idx)
                    if trial == 0:
                        scene_first_try.add(scene_idx)

                n_sat = sum(1 for v in per_c.values() if v['satisfied'])
                stats['constraint_sat_rates'].append(
                    n_sat / len(per_c) if per_c else 0)

                for ci, cinfo in per_c.items():
                    stats['per_type'][cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        stats['per_type'][cinfo['type']][0] += 1

        trial_rate = 100 * stats['successes'] / stats['total']
        scene_top1 = 100 * len(scene_first_try) / n_scenes if n_scenes else 0
        scene_topk = 100 * len(scene_succeeded) / n_scenes if n_scenes else 0
        avg_time = 1000 * np.mean(stats['times'])

        print(f"trial={trial_rate:.1f}%  top1={scene_top1:.1f}%  "
              f"topk={scene_topk:.1f}%  time={avg_time:.0f}ms")

        results[n_obj] = {
            'stats': stats,
            'n_scenes': n_scenes,
            'trial_rate': trial_rate,
            'scene_top1': scene_top1,
            'scene_topk': scene_topk,
            'avg_time_ms': avg_time,
        }

    return results


def print_comparison(all_results):
    """Print side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f"  PCFM Evaluation Results")
    print(f"{'='*90}")

    methods = list(all_results.keys())
    header = f"{'n_obj':>5}"
    for m in methods:
        header += f"  |  {m:^24}"
    sub = f"{'':>5}"
    for _ in methods:
        sub += f"  |  {'trial':>6} {'top1':>6} {'time':>8}"
    print(f"\n{header}")
    print(f"{sub}")
    print("-" * 90)

    for n_obj in range(2, 6):
        line = f"{n_obj:>5}"
        for m in methods:
            r = all_results[m].get(n_obj, {})
            trial = f"{r.get('trial_rate', 0):.1f}%" if r else "—"
            top1 = f"{r.get('scene_top1', 0):.1f}%" if r else "—"
            t = f"{r.get('avg_time_ms', 0):.0f}ms" if r else "—"
            line += f"  |  {trial:>6} {top1:>6} {t:>8}"
        print(line)

    # Per-constraint breakdown for last method (PCFM)
    last_method = methods[-1]
    last_results = all_results[last_method]
    print(f"\n{'-'*90}")
    print(f"  {last_method} — per-constraint satisfaction rates:")
    for n_obj in sorted(last_results.keys()):
        if 'stats' not in last_results[n_obj]:
            continue
        pt = last_results[n_obj]['stats']
        print(f"\n  {n_obj} objects:")
        for ct in sorted(pt['per_type'].keys()):
            s, t = pt['per_type'][ct]
            print(f"    {ct:14s}: {100*s/t:5.1f}% ({s}/{t})")

    print(f"\n  Reference: Diffusion-CCSP + ULA (10 tries):")
    print(f"    2-obj: 96%, 3-obj: 92%, 4-obj: 70%, 5-obj: 44%")
    print(f"{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PCFM evaluation')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--n_proj_iters', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--final_proj_iters', type=int, default=10)
    parser.add_argument('--damping', type=float, default=1e-4)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--proj_start', type=float, default=0.3)
    parser.add_argument('--skip_pure', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_mode = 'qualitative'

    print(f"\n{'#'*70}")
    print(f"# PCFM: Physics-Constrained Flow Matching for CCSP")
    print(f"# Steps={args.n_steps}, ProjIters={args.n_proj_iters}, "
          f"Alpha={args.alpha}, FinalIters={args.final_proj_iters}")
    print(f"# Damping={args.damping}, ProjStart={args.proj_start}, "
          f"Samples={args.n_samples}")
    print(f"{'#'*70}")

    # Load pretrained flow v1 model
    _, _, dims, constraint_types = get_data_config(input_mode)
    flow_dir = f'./logs/flow_{input_mode}_h{args.hidden_dim}'
    ckpt_path = os.path.join(flow_dir, 'flow_model_best.pt')

    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded: {ckpt_path}")

    all_results = {}

    # Pure flow baseline
    if not args.skip_pure:
        print(f"\n{'─'*50}")
        print(f"  Pure Flow (no projection)")
        print(f"{'─'*50}")

        from fix_and_eval import sample_flow_fixed
        def pure_sampler(model, data, device='cuda'):
            return sample_flow_fixed(model, data, n_steps=args.n_steps, device=device)

        all_results['Pure Flow'] = evaluate_method(
            model, constraint_types, pure_sampler,
            n_samples=args.n_samples, device=device)

    # PCFM
    print(f"\n{'─'*50}")
    print(f"  PCFM (α={args.alpha}, proj={args.n_proj_iters}, "
          f"final={args.final_proj_iters})")
    print(f"{'─'*50}")

    def pcfm_sampler(model, data, device='cuda'):
        return sample_pcfm(
            model, data, constraint_types,
            n_steps=args.n_steps, n_proj_iters=args.n_proj_iters,
            alpha=args.alpha, final_proj_iters=args.final_proj_iters,
            damping=args.damping, proj_start=args.proj_start,
            device=device)

    all_results['PCFM'] = evaluate_method(
        model, constraint_types, pcfm_sampler,
        n_samples=args.n_samples, device=device)

    # Also test projection-only (no flow, just project from noise)
    print(f"\n{'─'*50}")
    print(f"  Projection Only (GN from random noise, no flow)")
    print(f"{'─'*50}")

    def proj_only_sampler(model, data, device='cuda'):
        batch = data.to(device)
        x = batch.x.to(device)
        pose_dim = model.dims[-1][0]
        pose_begin = model.dims[-1][1]
        pose_end = model.dims[-1][2]
        geom_end = model.dims[0][2]
        mask = batch.mask.bool().to(device)
        clean_poses = x[:, pose_begin:pose_end]
        geoms = x[:, :geom_end]

        # Start from random poses in [0, 2]
        x_t = torch.rand(x.shape[0], pose_dim, device=device) * 2.0
        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        # Just project
        poses_np = x_t.cpu().numpy()
        edge_index_np = batch.edge_index.T.cpu().numpy()
        edge_attr_np = batch.edge_attr.cpu().numpy()
        geoms_np = geoms.cpu().numpy()
        free_mask = (~mask).cpu().numpy()

        poses_proj = gauss_newton_project(
            poses_np, edge_index_np, edge_attr_np, geoms_np,
            constraint_types, free_mask,
            n_iters=50, damping=args.damping, pose_dim=pose_dim)

        x_t = torch.tensor(poses_proj, dtype=torch.float32, device=device)
        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
        return x_t

    all_results['Proj Only'] = evaluate_method(
        model, constraint_types, proj_only_sampler,
        n_samples=args.n_samples, device=device)

    # Print comparison
    print_comparison(all_results)

    # Save results
    save_path = './logs/eval_pcfm.json'
    os.makedirs('./logs', exist_ok=True)
    serializable = {'config': vars(args)}
    for method_name, results in all_results.items():
        serializable[method_name] = {}
        for k, v in results.items():
            sv = {kk: vv for kk, vv in v.items() if kk != 'stats'}
            if 'stats' in v:
                sv['per_type'] = {ct: list(counts)
                                  for ct, counts in v['stats']['per_type'].items()}
            serializable[method_name][str(k)] = sv

    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
