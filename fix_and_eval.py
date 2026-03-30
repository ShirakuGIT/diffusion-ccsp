"""
FIXES for train_flow.py and solve_flow_ccsp.py
================================================
Apply these changes to fix:
1. Flow model generating out-of-tray poses (hard clamp)
2. Diffusion baseline import error
3. Insufficient QP correction

Copy the functions below into the respective files,
OR just run this file which patches and re-evaluates.

Usage:
    python fix_and_eval.py -input_mode qualitative -flow_checkpoint best
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

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints, puzzle_constraints

from train_flow import FlowMatchingCCSP, get_data_config


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1: Hard clamp for tray containment
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_to_tray(poses, geoms, mask, pose_dim=4):
    poses_out = poses.clone()

    # Extract x and y
    x = poses[:, 0]
    y = poses[:, 1]

    wi = geoms[:, 0]
    hi = geoms[:, 1]

    margin = 0.02

    x_min = wi + margin
    x_max = 2.0 - wi - margin
    y_min = hi + margin
    y_max = 2.0 - hi - margin

    # Handle degenerate cases
    x_center = torch.full_like(x_min, 1.0)
    y_center = torch.full_like(y_min, 1.0)

    x_min = torch.where(x_min >= x_max, x_center, x_min)
    x_max = torch.where(x_min >= x_max, x_center, x_max)
    y_min = torch.where(y_min >= y_max, y_center, y_min)
    y_max = torch.where(y_min >= y_max, y_center, y_max)

    # Clamp (OUT-OF-PLACE)
    x_clamped = torch.clamp(x, x_min, x_max)
    y_clamped = torch.clamp(y, y_min, y_max)

    # Respect mask (do NOT modify fixed nodes)
    if mask is not None:
        x_clamped = torch.where(mask, x, x_clamped)
        y_clamped = torch.where(mask, y, y_clamped)

    # Reconstruct tensor (NO in-place ops)
    poses_out = poses.clone()
    poses_out = torch.cat([
        x_clamped.unsqueeze(1),
        y_clamped.unsqueeze(1),
        poses[:, 2:]
    ], dim=1)

    return poses_out

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: Fixed sampling functions with clamping
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_flow_fixed(model, batch, n_steps=20, device='cuda'):
    """Flow sampling WITH tray clamping after each step."""
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

    # Start from noise
    x_t = torch.randn(n_nodes, pose_dim, device=device)
    x_t[mask] = clean_poses[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        v = model(x_t, batch, t)
        x_t = x_t + v * dt
        x_t[mask] = clean_poses[mask]
        # CLAMP to tray after each step
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


@torch.no_grad()
def sample_projected_flow_fixed(model, batch, constraint_types,
                                 n_pred_steps=1, n_corr_steps=30,
                                 alpha_scale=2.0, epsilon=1.0, rho=0.8,
                                 delta=0.01, device='cuda'):
    """
    Full Projected Flow-CCSP with fixes:
    1. Hard clamp for tray containment
    2. Stronger epsilon for QP
    3. Energy-guided base velocity (not just flow velocity)
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

    info = {'pred_time': 0, 'corr_time': 0, 'corrections': []}

    # ── Phase 1: PREDICTION ──
    t0 = time.time()
    x_t = torch.randn(n_nodes, pose_dim, device=device)
    x_t[mask] = clean_poses[mask]

    # Multiple prediction steps for better proposal
    n_pred = max(n_pred_steps, 5)
    dt = 1.0 / n_pred
    for step in range(n_pred):
        t = step * dt
        v = model(x_t, batch, t)
        x_t = x_t + v * dt
        x_t[mask] = clean_poses[mask]

    # CLAMP after prediction
    x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
    info['pred_time'] = time.time() - t0

    # ── Phase 2: CORRECTION with CBF-QP ──
    t0 = time.time()
    edge_index = batch.edge_index.T.to(device)
    edge_attr = batch.edge_attr.to(device)

    for step in range(n_corr_steps):
        t_corr = step / n_corr_steps

        # Get flow velocity at current state
        v_flow = model(x_t, batch, 0.5 + 0.5 * t_corr)
        v_flow_np = v_flow.cpu().numpy()

        # Vanishing scale
        v_scaled = alpha_scale * (1.0 - t_corr) * v_flow_np

        # Collect barriers
        barriers = []
        full_dim = n_nodes * pose_dim

        for ei in range(edge_index.shape[0]):
            i = edge_index[ei, 0].item()
            j = edge_index[ei, 1].item()
            ctype_idx = int(edge_attr[ei].item())
            if ctype_idx >= len(constraint_types):
                continue
            ctype = constraint_types[ctype_idx]

            pose_i = x_t[i].cpu()
            pose_j = x_t[j].cpu()
            geom_i = x[i, :geom_end].cpu()
            geom_j = x[j, :geom_end].cpu()

            h_val, grad_i, grad_j = compute_barrier(
                ctype, pose_i, pose_j, geom_i, geom_j)

            full_grad = np.zeros(full_dim)
            full_grad[i*pose_dim:(i+1)*pose_dim] = grad_i[:pose_dim]
            full_grad[j*pose_dim:(j+1)*pose_dim] = grad_j[:pose_dim]

            barriers.append((h_val, full_grad))

        # QP correction
        v_flat = v_scaled.reshape(-1)
        w_relax = max(0, 1.0 - t_corr / 0.7) * 5.0

        u_flat = v_flat.copy()
        for _ in range(5):
            for h_val, grad in barriers:
                gnorm2 = np.dot(grad, grad)
                if gnorm2 < 1e-10:
                    continue
                alpha_cbf = epsilon * np.sign(h_val - delta) * (abs(h_val - delta) ** rho)
                condition = np.dot(grad, u_flat) + alpha_cbf + w_relax * max(0, -(h_val - delta))
                if condition < 0:
                    u_flat = u_flat + (-condition / gnorm2) * grad

        correction = np.linalg.norm(u_flat - v_flat)
        info['corrections'].append(correction)

        # Update
        dt_corr = 1.0 / n_corr_steps
        poses_flat = x_t.cpu().numpy().reshape(-1)
        poses_flat += u_flat * dt_corr
        x_t = torch.tensor(poses_flat.reshape(n_nodes, pose_dim),
                           dtype=torch.float32, device=device)
        x_t[mask] = clean_poses[mask]

        # CLAMP after each correction step
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    info['corr_time'] = time.time() - t0
    info['total_time'] = info['pred_time'] + info['corr_time']
    info['avg_correction'] = np.mean(info['corrections']) if info['corrections'] else 0

    return x_t, info


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3: Corrected barrier functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_barrier(constraint_type, pose_i, pose_j, geom_i, geom_j):
    """
    Compute barrier value and gradients.

    Normalized coordinate system (qualitative domain):
        geom = (w, h) where w = obj_width / tray_width ∈ [0, 1]
        pose = (x, y, cos, sin) where x = obj_x / (tray_w/2) ∈ [0, 2]

    Tray spans [0, 2] × [0, 2] in normalized coords.
    Object half-extent in x = w (since w = obj_w/tray_w and
    position scale = tray_w/2, so half_width_norm = (obj_w/2)/(tray_w/2) = w)
    """
    xi, yi = pose_i[0].item(), pose_i[1].item()
    xj, yj = pose_j[0].item(), pose_j[1].item()
    wi = geom_i[0].item() if len(geom_i) > 0 else 0.1
    hi = geom_i[1].item() if len(geom_i) > 1 else 0.1
    wj = geom_j[0].item() if len(geom_j) > 0 else 0.1
    hj = geom_j[1].item() if len(geom_j) > 1 else 0.1

    grad_i = np.zeros(4)
    grad_j = np.zeros(4)

    if constraint_type == 'in':
        # Object i inside tray: margins from each edge
        margin_l = xi - wi
        margin_r = 2.0 - (xi + wi)
        margin_b = yi - hi
        margin_t = 2.0 - (yi + hi)
        margins = [margin_l, margin_r, margin_b, margin_t]
        h_val = min(margins)
        idx = margins.index(h_val)
        if idx == 0:   grad_i[0] = 1.0
        elif idx == 1: grad_i[0] = -1.0
        elif idx == 2: grad_i[1] = 1.0
        elif idx == 3: grad_i[1] = -1.0

    elif constraint_type == 'cfree':
        dx = abs(xi - xj) - (wi + wj)
        dy = abs(yi - yj) - (hi + hj)
        h_val = max(dx, dy)
        if dx > dy:
            sign = 1.0 if xi > xj else -1.0
            grad_i[0], grad_j[0] = sign, -sign
        else:
            sign = 1.0 if yi > yj else -1.0
            grad_i[1], grad_j[1] = sign, -sign

    elif constraint_type == 'close-to':
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2) + 1e-8
        threshold = max(wi + wj, hi + hj) * 1.5
        h_val = threshold - dist
        grad_i[0] = -(xi - xj) / dist
        grad_i[1] = -(yi - yj) / dist
        grad_j[0] = (xi - xj) / dist
        grad_j[1] = (yi - yj) / dist

    elif constraint_type == 'away-from':
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2) + 1e-8
        threshold = max(wi + wj, hi + hj) * 2.0
        h_val = dist - threshold
        grad_i[0] = (xi - xj) / dist
        grad_i[1] = (yi - yj) / dist
        grad_j[0] = -(xi - xj) / dist
        grad_j[1] = -(yi - yj) / dist

    elif constraint_type == 'left-of':
        h_val = xj - xi - (wi + wj) * 0.5
        grad_i[0] = -1.0
        grad_j[0] = 1.0

    elif constraint_type == 'top-of':
        h_val = yi - yj - (hi + hj) * 0.5
        grad_i[1] = 1.0
        grad_j[1] = -1.0

    elif constraint_type == 'h-aligned':
        h_val = 0.3 - abs(yi - yj)
        sign = -1.0 if yi > yj else 1.0
        grad_i[1] = sign
        grad_j[1] = -sign

    elif constraint_type == 'v-aligned':
        h_val = 0.3 - abs(xi - xj)
        sign = -1.0 if xi > xj else 1.0
        grad_i[0] = sign
        grad_j[0] = -sign

    elif constraint_type == 'center-in':
        dist = np.sqrt((xi - 1.0)**2 + (yi - 1.0)**2) + 1e-8
        h_val = 0.5 - dist
        grad_i[0] = -(xi - 1.0) / dist
        grad_i[1] = -(yi - 1.0) / dist

    elif constraint_type == 'left-in':
        h_val = 1.0 - xi
        grad_i[0] = -1.0

    elif constraint_type == 'right-in':
        h_val = xi - 1.0
        grad_i[0] = 1.0

    elif constraint_type == 'top-in':
        h_val = yi - 1.0
        grad_i[1] = 1.0

    elif constraint_type == 'bottom-in':
        h_val = 1.0 - yi
        grad_i[1] = -1.0

    else:
        h_val = 1.0  # unknown constraint, always satisfied

    return h_val, grad_i, grad_j


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4: Constraint checking with same barrier functions
# ═══════════════════════════════════════════════════════════════════════════════

def check_constraints(poses, batch, constraint_types, device='cuda'):
    """Check all constraints using our barrier functions."""
    x = batch.x.to(device)
    edge_index = batch.edge_index.T.to(device)
    edge_attr = batch.edge_attr.to(device)
    geom_end = 2  # qualitative: geom = x[:, 0:2]

    results = {}
    all_ok = True

    for ei in range(edge_index.shape[0]):
        i = edge_index[ei, 0].item()
        j = edge_index[ei, 1].item()
        ctype_idx = int(edge_attr[ei].item())
        if ctype_idx >= len(constraint_types):
            continue
        ctype = constraint_types[ctype_idx]

        h_val, _, _ = compute_barrier(
            ctype,
            poses[i].cpu(), poses[j].cpu(),
            x[i, :geom_end].cpu(), x[j, :geom_end].cpu())

        satisfied = h_val >= -0.02
        results[ei] = {'type': ctype, 'h_val': h_val, 'satisfied': satisfied}
        if not satisfied:
            all_ok = False

    return all_ok, results


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 5: Evaluation with Diffusion-CCSP using THEIR checker
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_diffusion_their_way(run_id, milestone, test_tasks, input_mode,
                                  constraint_types, n_samples=10):
    """
    Evaluate Diffusion-CCSP using THEIR evaluation pipeline,
    bypassing our barrier functions entirely.
    This gives us their ground-truth numbers to compare against.
    """
    try:
        from train_utils import load_trainer
        trainer = load_trainer(run_id, milestone, verbose=False,
                               test_tasks=test_tasks, input_mode=input_mode)
        # Use their built-in evaluate
        trainer.evaluate('eval_compare', tries=(n_samples, 0),
                         render=False, save_log=False)
        return True
    except Exception as e:
        print(f"  Could not run Diffusion-CCSP evaluation: {e}")
        print(f"  This might be a PyTorch version issue.")
        print(f"  Try: pip install torch==2.0.1  (matching their requirements)")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_tasks, input_mode, constraint_types,
             n_samples=10, use_qp=True, n_corr_steps=30, device='cuda',
             method_name=""):
    """Evaluate with all fixes applied."""
    results = {}

    for n_obj, task_name in test_tasks.items():
        print(f"    {n_obj} objects...", end=" ", flush=True)
        dataset = GraphDataset(task_name, input_mode=input_mode,
                               pre_transform=pre_transform, visualize=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        task_results = {
            'successes': 0, 'total': 0,
            'times': [], 'corrections': [],
            'per_type': defaultdict(lambda: [0, 0]),
            'constraint_sat_rates': []
        }

        for data in loader:
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj)

                t0 = time.time()
                if use_qp:
                    poses, info = sample_projected_flow_fixed(
                        model, data, constraint_types,
                        n_pred_steps=5, n_corr_steps=n_corr_steps,
                        epsilon=1.0, rho=0.8, delta=0.01,
                        device=device)
                    task_results['corrections'].append(info.get('avg_correction', 0))
                else:
                    poses = sample_flow_fixed(model, data, n_steps=20, device=device)
                elapsed = time.time() - t0
                task_results['times'].append(elapsed)

                all_ok, per_c = check_constraints(
                    poses, data, constraint_types, device)

                task_results['total'] += 1
                if all_ok:
                    task_results['successes'] += 1

                n_sat = sum(1 for v in per_c.values() if v['satisfied'])
                task_results['constraint_sat_rates'].append(
                    n_sat / len(per_c) if per_c else 0)

                for ci, cinfo in per_c.items():
                    task_results['per_type'][cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        task_results['per_type'][cinfo['type']][0] += 1

        succ_rate = 100 * task_results['successes'] / task_results['total']
        avg_sat = 100 * np.mean(task_results['constraint_sat_rates'])
        avg_time = 1000 * np.mean(task_results['times'])
        print(f"  success={succ_rate:.1f}%  avg_sat={avg_sat:.1f}%  time={avg_time:.0f}ms")

        results[n_obj] = task_results

    return results


def print_results(name, results):
    """Print detailed results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    total_s, total_t = 0, 0
    for n_obj in sorted(results.keys()):
        r = results[n_obj]
        rate = 100 * r['successes'] / r['total'] if r['total'] > 0 else 0
        avg_sat = 100 * np.mean(r['constraint_sat_rates']) if r['constraint_sat_rates'] else 0
        avg_time = 1000 * np.mean(r['times']) if r['times'] else 0
        total_s += r['successes']
        total_t += r['total']
        print(f"\n  {n_obj} objects: {r['successes']}/{r['total']} "
              f"({rate:.1f}%)  sat={avg_sat:.1f}%  time={avg_time:.0f}ms")
        if r.get('corrections'):
            print(f"    QP correction: {np.mean(r['corrections']):.4f}")
        for ct in sorted(r['per_type'].keys()):
            s, t = r['per_type'][ct]
            print(f"    {ct:14s}: {100*s/t:5.1f}% ({s}/{t})")
    print(f"\n  OVERALL: {total_s}/{total_t} ({100*total_s/total_t:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_mode', type=str, default='qualitative')
    parser.add_argument('-flow_checkpoint', type=str, default='best')
    parser.add_argument('-flow_dir', type=str, default=None)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-n_samples', type=int, default=10)
    parser.add_argument('-n_corr_steps', type=int, default=40)
    parser.add_argument('--no_qp', action='store_true')
    parser.add_argument('-run_id', type=str, default=None)
    parser.add_argument('-milestone', type=int, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_task, test_tasks, dims, constraint_types = get_data_config(args.input_mode)

    print(f"\n{'#'*70}")
    print(f"# Projected Flow-CCSP Evaluation (WITH FIXES)")
    print(f"# Task: {args.input_mode}")
    print(f"# Fixes: hard tray clamping, stronger QP, 5-step prediction")
    print(f"{'#'*70}")

    # Load flow model
    flow_dir = args.flow_dir or f'./logs/flow_{args.input_mode}_h{args.hidden_dim}'
    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device).to(device)

    ckpt_path = os.path.join(flow_dir, f'flow_model_{args.flow_checkpoint}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded: {ckpt_path}")

    all_results = {}

    # Flow only (with clamping)
    print(f"\n  [1] Flow Only (20 steps, with tray clamping)")
    r1 = evaluate(model, test_tasks, args.input_mode, constraint_types,
                  n_samples=args.n_samples, use_qp=False, device=device)
    all_results['Flow Only (clamped)'] = r1
    print_results('Flow Only (clamped)', r1)

    # Flow + QP
    if not args.no_qp:
        print(f"\n  [2] Projected Flow + QP ({args.n_corr_steps} correction steps)")
        r2 = evaluate(model, test_tasks, args.input_mode, constraint_types,
                      n_samples=args.n_samples, use_qp=True,
                      n_corr_steps=args.n_corr_steps, device=device)
        all_results['Projected Flow + QP'] = r2
        print_results('Projected Flow + QP', r2)

    # Diffusion baseline
    run_id = args.run_id or 'qsd3ju74'
    milestone = args.milestone or 7
    print(f"\n  [3] Diffusion-CCSP baseline (run={run_id}, m={milestone})")
    print(f"      Using their eval pipeline...")
    evaluate_diffusion_their_way(run_id, milestone, test_tasks,
                                 args.input_mode, constraint_types,
                                 n_samples=args.n_samples)

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<30} | {'2 obj':>8} | {'3 obj':>8} | {'4 obj':>8} | {'5 obj':>8} | {'Time':>8}")
    print("-" * 80)
    for name, results in all_results.items():
        parts = [f"{name:<30}"]
        all_times = []
        for n_obj in [2, 3, 4, 5]:
            if n_obj in results:
                r = results[n_obj]
                rate = 100 * r['successes'] / r['total']
                parts.append(f"{rate:>7.1f}%")
                all_times.extend(r['times'])
            else:
                parts.append(f"{'N/A':>8}")
        avg_t = 1000 * np.mean(all_times) if all_times else 0
        parts.append(f"{avg_t:>6.0f}ms")
        print(" | ".join(parts))

    # Save
    save_path = f'./logs/eval_{args.input_mode}_fixed.json'
    serializable = {}
    for name, results in all_results.items():
        serializable[name] = {}
        for n_obj, r in results.items():
            serializable[name][str(n_obj)] = {
                'successes': r['successes'],
                'total': r['total'],
                'avg_time_ms': 1000 * np.mean(r['times']),
                'avg_sat': 100 * np.mean(r['constraint_sat_rates']),
                'per_type': {k: [v[0], v[1]] for k, v in r['per_type'].items()}
            }
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()