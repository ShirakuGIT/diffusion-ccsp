"""
Model Diagnostics: Projection Residual + Rollout Constraint Curve
=================================================================
Tests whether the current flow model has learned good global structure,
or whether the GN projection is doing all the heavy lifting.

Key metrics:
  1. Projection Residual: ||x_proj - x_flow|| per step — how hard GN works
  2. Rollout Curve: constraint violation vs timestep t
  3. Per-constraint breakdown of violations across t

Usage:
    python diagnose_model.py
"""

import os, sys, time, json
import numpy as np
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader


from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from flow_matching.train_flow import FlowMatchingCCSP, get_data_config
from flow_matching.fix_and_eval import clamp_to_tray, compute_barrier
from eval_pcfm import gauss_newton_project


def load_model(device='cuda'):
    _, _, dims, constraint_types = get_data_config('qualitative')
    ckpt = torch.load('./logs/flow_qualitative_h256/flow_model_best.pt', map_location=device)
    model = FlowMatchingCCSP(dims=dims, hidden_dim=256, constraint_types=constraint_types,
                              normalize=True, device=device).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, dims, constraint_types


def constraint_violation(poses_np, edge_index, edge_attr, geoms_np, constraint_types):
    """Returns mean |h| over all violated constraints (h < 0 = violated)."""
    violations = []
    for ei in range(edge_index.shape[0]):
        i, j = int(edge_index[ei, 0]), int(edge_index[ei, 1])
        ctype_idx = int(edge_attr[ei])
        if ctype_idx >= len(constraint_types):
            continue
        ctype = constraint_types[ctype_idx]
        pose_i = torch.tensor(poses_np[i], dtype=torch.float32)
        pose_j = torch.tensor(poses_np[j], dtype=torch.float32)
        geom_i = torch.tensor(geoms_np[i], dtype=torch.float32)
        geom_j = torch.tensor(geoms_np[j], dtype=torch.float32)
        h_val, _, _ = compute_barrier(ctype, pose_i, pose_j, geom_i, geom_j)
        violations.append(min(h_val, 0.0))  # 0 if satisfied, negative if violated
    if not violations:
        return 0.0
    return -np.mean(violations)  # positive = avg violation magnitude


def run_diagnostics(model, constraint_types, n_scenes=50, n_steps=20, device='cuda'):
    test_task = "RandomSplitQualitativeWorld(100)_qualitative_test_3_split"
    dataset = GraphDataset(test_task, input_mode='qualitative',
                           pre_transform=pre_transform, visualize=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rollout_violations = defaultdict(list)   # step -> list of violations
    proj_residuals = defaultdict(list)       # step -> list of ||x_proj - x_flow||
    final_violations = []
    final_violations_after_proj = []

    n_done = 0
    for data in loader:
        if n_done >= n_scenes:
            break
        n_done += 1
        torch.manual_seed(n_done * 42)

        batch = data.to(device)
        x = batch.x.to(device)
        pose_dim = model.dims[-1][0]
        pose_begin = model.dims[-1][1]
        pose_end = model.dims[-1][2]
        geom_end = model.dims[0][2]
        n_nodes = x.shape[0]
        mask = batch.mask.bool().to(device)
        clean_poses = x[:, pose_begin:pose_end]
        geoms = x[:, :geom_end]

        edge_index_np = batch.edge_index.T.cpu().numpy()
        edge_attr_np = batch.edge_attr.cpu().numpy()
        geoms_np = geoms.cpu().numpy()
        free_mask = (~mask).cpu().numpy()

        x_t = torch.randn(n_nodes, pose_dim, device=device)
        x_t[mask] = clean_poses[mask]

        dt = 1.0 / n_steps

        with torch.no_grad():
            for step in range(n_steps):
                t = step * dt

                # Euler step
                v = model(x_t, batch, t)
                x_t = x_t + v * dt
                x_t[mask] = clean_poses[mask]
                x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

                # Measure constraint violation at this step
                poses_np = x_t.cpu().numpy()
                viol = constraint_violation(poses_np, edge_index_np, edge_attr_np,
                                            geoms_np, constraint_types)
                rollout_violations[step].append(viol)

                # Measure how much GN projection would change things
                poses_proj = gauss_newton_project(
                    poses_np, edge_index_np, edge_attr_np, geoms_np,
                    constraint_types, free_mask,
                    n_iters=5, damping=1e-4, pose_dim=pose_dim)
                residual = np.linalg.norm(poses_proj - poses_np)
                proj_residuals[step].append(residual)

        # Final state violations
        final_violations.append(viol)
        viol_after = constraint_violation(poses_proj, edge_index_np, edge_attr_np,
                                          geoms_np, constraint_types)
        final_violations_after_proj.append(viol_after)

    return rollout_violations, proj_residuals, final_violations, final_violations_after_proj


def print_report(rollout_violations, proj_residuals, final_violations, final_after_proj):
    print("\n" + "="*65)
    print("  MODEL DIAGNOSTIC REPORT")
    print("="*65)

    # Rollout curve (every 5 steps)
    print("\n  Constraint Violation vs Timestep (avg |h| over violated constraints)")
    print("  step    flow_viol   proj_residual")
    print("  " + "-"*45)
    steps = sorted(rollout_violations.keys())
    for s in steps:
        viol = np.mean(rollout_violations[s])
        res = np.mean(proj_residuals[s])
        bar = "█" * int(viol * 50)
        print(f"  t={s/max(steps):.2f}   {viol:.4f}     {res:.4f}   {bar}")

    # Final state summary
    print(f"\n  Final state (t=1.0):")
    print(f"    avg violation (flow only):        {np.mean(final_violations):.4f}")
    print(f"    avg violation (after GN proj):    {np.mean(final_after_proj):.4f}")
    print(f"    avg proj residual (final step):   {np.mean(proj_residuals[max(steps)]):.4f}")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    final_res = np.mean(proj_residuals[max(steps)])
    final_viol = np.mean(final_violations)
    final_viol_proj = np.mean(final_after_proj)

    if final_res < 0.05:
        print(f"    ✓ Projection residual is SMALL ({final_res:.4f})")
        print(f"      → Model is near feasible. Problem is in sampler, not model.")
    elif final_res < 0.2:
        print(f"    ~ Projection residual is MODERATE ({final_res:.4f})")
        print(f"      → Model has some global structure but leaves gaps.")
        print(f"      → Feasibility loss in training would help.")
    else:
        print(f"    ✗ Projection residual is LARGE ({final_res:.4f})")
        print(f"      → Model is NOT near feasible. Training objective is wrong.")
        print(f"      → Add feasibility loss: ||h(x_t + v*dt)|| to training.")

    if final_viol_proj < final_viol * 0.5:
        proj_pct = 100 * (final_viol - final_viol_proj) / (final_viol + 1e-8)
        print(f"    → GN projection reduces violation by {proj_pct:.0f}%")
        print(f"      → PCFM is doing significant work — model needs improvement")
    else:
        print(f"    → GN projection barely helps — model is already near feasible")

    print("="*65)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model...")
    model, dims, constraint_types = load_model(device)
    print("Running diagnostics on 50 scenes (3-obj)...")

    rollout_violations, proj_residuals, final_viol, final_after_proj = \
        run_diagnostics(model, constraint_types, n_scenes=50, n_steps=20, device=device)

    print_report(rollout_violations, proj_residuals, final_viol, final_after_proj)

    # Save
    out = {
        'rollout_violations': {str(k): float(np.mean(v)) for k, v in rollout_violations.items()},
        'proj_residuals': {str(k): float(np.mean(v)) for k, v in proj_residuals.items()},
        'final_violation_flow': float(np.mean(final_viol)),
        'final_violation_after_proj': float(np.mean(final_after_proj)),
    }
    with open('./logs/model_diagnostics.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("  Saved: ./logs/model_diagnostics.json")


if __name__ == '__main__':
    main()
