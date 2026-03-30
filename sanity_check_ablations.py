"""
Ablation Study: Diagnosing Flow Matching Failures
==================================================
Tests the following hypotheses:
  H1: Too few sampling steps (20 → 50, 100, 200)
  H2: Per-step clamping interferes with the ODE trajectory
  H3: Later checkpoint (epoch 100) might be better than "best" (epoch 60)
  H4: Constraint checker mismatch between diffusion and flow

Usage:
    python sanity_check_ablations.py
"""

import os
import sys
import time
import json
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
from fix_and_eval import clamp_to_tray, check_constraints


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_flow_variants(model, batch, n_steps=20, clamp_mode='every_step',
                         device='cuda'):
    """
    Flow sampling with configurable steps and clamping.

    clamp_mode:
      'every_step' — clamp after every Euler step (current default)
      'final_only' — only clamp the final output
      'none'       — no clamping at all
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

    # Start from noise
    x_t = torch.randn(n_nodes, pose_dim, device=device)
    x_t[mask] = clean_poses[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        v = model(x_t, batch, t)
        x_t = x_t + v * dt
        x_t[mask] = clean_poses[mask]

        if clamp_mode == 'every_step':
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    # Final clamp
    if clamp_mode in ('final_only', 'every_step'):
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


@torch.no_grad()
def sample_flow_heun(model, batch, n_steps=20, device='cuda'):
    """
    Flow sampling with Heun's method (2nd-order) + final-only clamp.
    Heun's: x_{n+1} = x_n + dt/2 * (v(x_n, t_n) + v(x_n + dt*v(x_n,t_n), t_{n+1}))
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

    x_t = torch.randn(n_nodes, pose_dim, device=device)
    x_t[mask] = clean_poses[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        t_next = (step + 1) * dt

        v1 = model(x_t, batch, t)
        x_pred = x_t + v1 * dt
        x_pred[mask] = clean_poses[mask]

        v2 = model(x_pred, batch, t_next)
        x_t = x_t + 0.5 * dt * (v1 + v2)
        x_t[mask] = clean_poses[mask]

    # Final clamp only
    x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
    return x_t


@torch.no_grad()
def sample_flow_stochastic(model, batch, n_steps=100, noise_scale=0.01,
                           device='cuda'):
    """
    Flow sampling with added stochasticity (pseudo-Langevin).
    Adds small noise at each step, similar to what makes diffusion robust.
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

    x_t = torch.randn(n_nodes, pose_dim, device=device)
    x_t[mask] = clean_poses[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        v = model(x_t, batch, t)
        # Add decreasing noise (anneal from noise_scale to 0)
        noise_level = noise_scale * (1.0 - t)
        noise = noise_level * torch.randn_like(x_t)
        x_t = x_t + v * dt + noise
        x_t[mask] = clean_poses[mask]

    x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
    return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_variant(model, constraint_types, sampler_fn, sampler_kwargs,
                     input_mode='qualitative', n_samples=10, device='cuda',
                     test_n_objs=(2, 3, 4, 5)):
    """Evaluate a sampling variant on all test sets."""
    test_tasks = {i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
                  for i in test_n_objs}

    results = {}
    for n_obj, task_name in test_tasks.items():
        print(f"      {n_obj}-obj: ", end="", flush=True)
        dataset = GraphDataset(task_name, input_mode=input_mode,
                               pre_transform=pre_transform, visualize=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        successes = 0
        total = 0
        scene_succeeded = set()
        scene_first_try = set()
        n_scenes = 0
        times = []
        per_type = defaultdict(lambda: [0, 0])

        for scene_idx, data in enumerate(loader):
            n_scenes += 1
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj * 100 + scene_idx)

                t0 = time.time()
                poses = sampler_fn(model, data, device=device, **sampler_kwargs)
                elapsed = time.time() - t0
                times.append(elapsed)

                all_ok, per_c = check_constraints(
                    poses, data, constraint_types, device)

                total += 1
                if all_ok:
                    successes += 1
                    scene_succeeded.add(scene_idx)
                    if trial == 0:
                        scene_first_try.add(scene_idx)

                for ci, cinfo in per_c.items():
                    per_type[cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        per_type[cinfo['type']][0] += 1

        trial_rate = 100 * successes / total if total else 0
        scene_top1 = 100 * len(scene_first_try) / n_scenes if n_scenes else 0
        scene_topk = 100 * len(scene_succeeded) / n_scenes if n_scenes else 0
        avg_time = 1000 * np.mean(times) if times else 0

        print(f"trial={trial_rate:.1f}%  top1={scene_top1:.1f}%  "
              f"topk={scene_topk:.1f}%  time={avg_time:.0f}ms")

        results[n_obj] = {
            'trial_rate': trial_rate,
            'scene_top1': scene_top1,
            'scene_topk': scene_topk,
            'avg_time_ms': avg_time,
            'per_type': {ct: list(counts) for ct, counts in per_type.items()},
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT CHECKER COMPARISON (H4)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_checkers(model, constraint_types, input_mode='qualitative',
                     device='cuda'):
    """
    Compare our barrier-based checker vs diffusion's render_world_from_graph
    on the same flow-generated poses, to verify they agree.
    """
    from denoise_fn import constraint_from_edge_attr
    from data_utils import render_world_from_graph

    test_task = "RandomSplitQualitativeWorld(100)_qualitative_test_2_split"
    dataset = GraphDataset(test_task, input_mode=input_mode,
                           pre_transform=pre_transform, visualize=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    agree = 0
    disagree = 0
    our_pass_their_fail = 0
    our_fail_their_pass = 0

    for scene_idx, data in enumerate(loader):
        if scene_idx >= 20:  # just check 20 scenes
            break
        torch.manual_seed(42 + scene_idx)

        poses = sample_flow_variants(model, data, n_steps=20,
                                     clamp_mode='every_step', device=device)

        # Our checker
        our_ok, _ = check_constraints(poses, data, constraint_types, device)

        # Their checker
        batch = data.to(device)
        x = batch.x.clone()
        # Reconstruct full features
        geom_end = model.dims[0][2]
        pose_begin = model.dims[-1][1]
        pose_end = model.dims[-1][2]
        all_features = torch.cat([
            x[:, :pose_begin].cpu(),
            poses.detach().cpu(),
            x[:, pose_end:].cpu(),
        ], dim=1)
        all_features.clamp_(-1., 1.)

        world_dims = batch.world_dims[0]
        edge_index = batch.edge_index[:, torch.where(batch.edge_extract == 0)[0]]
        edge_attr = batch.edge_attr[torch.where(batch.edge_extract == 0)]
        offset = edge_index.min()
        edge_index -= offset
        constraints = constraint_from_edge_attr(edge_attr, edge_index)

        evaluations = render_world_from_graph(
            all_features, world_dims=world_dims,
            world_name='RandomSplitQualitativeWorld',
            constraints=constraints, save=False, log=False, show=False)
        their_ok = (len(evaluations) == 0)

        if our_ok == their_ok:
            agree += 1
        else:
            disagree += 1
            if our_ok and not their_ok:
                our_pass_their_fail += 1
                print(f"    Scene {scene_idx}: OUR=pass, THEIRS=fail "
                      f"({evaluations})")
            else:
                our_fail_their_pass += 1
                print(f"    Scene {scene_idx}: OUR=fail, THEIRS=pass")

    print(f"\n    Checker comparison (20 scenes, 2-obj):")
    print(f"      Agree: {agree}/20")
    print(f"      Disagree: {disagree}/20")
    print(f"        Ours pass, theirs fail: {our_pass_their_fail}")
    print(f"        Ours fail, theirs pass: {our_fail_their_pass}")
    return agree, disagree


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    """Print comparison table of all ablations."""
    print("\n" + "=" * 85)
    print("  ABLATION SUMMARY")
    print("=" * 85)

    header = f"{'Variant':<35} | {'2-obj':>8} {'3-obj':>8} {'4-obj':>8} {'5-obj':>8}"
    print(f"\n{header}")
    print("-" * 85)

    # Reference
    print(f"{'Diffusion (Reverse, no ULA)':35} | {'68.0%':>8} {'43.0%':>8} {'11.0%':>8} {'3.0%':>8}  [top-1]")
    print(f"{'':35} | {'100.0%':>8} {'92.0%':>8} {'69.0%':>8} {'34.0%':>8}  [top-10]")
    print("-" * 85)

    for name, results in all_results.items():
        vals_top1 = []
        vals_topk = []
        vals_trial = []
        for n_obj in (2, 3, 4, 5):
            if n_obj in results:
                vals_top1.append(f"{results[n_obj]['scene_top1']:.1f}%")
                vals_topk.append(f"{results[n_obj]['scene_topk']:.1f}%")
                vals_trial.append(f"{results[n_obj]['trial_rate']:.1f}%")
            else:
                vals_top1.append("—")
                vals_topk.append("—")
                vals_trial.append("—")
        print(f"{name:35} | {vals_top1[0]:>8} {vals_top1[1]:>8} {vals_top1[2]:>8} {vals_top1[3]:>8}  [top-1]")
        print(f"{'':35} | {vals_topk[0]:>8} {vals_topk[1]:>8} {vals_topk[2]:>8} {vals_topk[3]:>8}  [top-10]")

    print("=" * 85)

    # Per-constraint breakdown for key variants
    for name, results in all_results.items():
        if 2 in results and results[2]['per_type']:
            print(f"\n  {name} — per-constraint (2-obj):")
            for ct in sorted(results[2]['per_type'].keys()):
                s, t = results[2]['per_type'][ct]
                print(f"    {ct:14s}: {100*s/t:5.1f}% ({s}/{t})")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = 10
    _, _, dims, constraint_types = get_data_config('qualitative')

    print(f"\n{'#'*70}")
    print(f"# ABLATION STUDY: Diagnosing Flow Matching Failures")
    print(f"# {n_samples} tries per scene, 100 scenes per n_obj")
    print(f"{'#'*70}")

    all_results = {}

    # ── Load best model ──
    flow_dir = './logs/flow_qualitative_h256'
    def load_model(ckpt_name):
        model = FlowMatchingCCSP(
            dims=dims, hidden_dim=256,
            constraint_types=constraint_types,
            normalize=True, device=device).to(device)
        ckpt_path = os.path.join(flow_dir, f'flow_model_{ckpt_name}.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded: {ckpt_path}")
        return model

    model = load_model('best')

    # ══════════════════════════════════════════════════════════════════════
    # H1: Sampling Steps
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("H1: Effect of sampling steps (Euler, clamp every step)")
    print(f"{'─'*60}")

    for n_steps in [20, 50, 100, 200]:
        name = f"Euler {n_steps} steps (clamp every)"
        print(f"\n  {name}:")
        all_results[name] = evaluate_variant(
            model, constraint_types,
            sampler_fn=sample_flow_variants,
            sampler_kwargs={'n_steps': n_steps, 'clamp_mode': 'every_step'},
            n_samples=n_samples, device=device)

    # ══════════════════════════════════════════════════════════════════════
    # H2: Clamping Mode
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("H2: Effect of clamping mode (100 Euler steps)")
    print(f"{'─'*60}")

    for clamp_mode in ['every_step', 'final_only', 'none']:
        name = f"Euler 100 steps (clamp={clamp_mode})"
        print(f"\n  {name}:")
        all_results[name] = evaluate_variant(
            model, constraint_types,
            sampler_fn=sample_flow_variants,
            sampler_kwargs={'n_steps': 100, 'clamp_mode': clamp_mode},
            n_samples=n_samples, device=device)

    # ══════════════════════════════════════════════════════════════════════
    # H2b: Heun's Method (2nd order)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("H2b: Heun's method (2nd-order integrator)")
    print(f"{'─'*60}")

    for n_steps in [20, 50, 100]:
        name = f"Heun {n_steps} steps (final clamp)"
        print(f"\n  {name}:")
        all_results[name] = evaluate_variant(
            model, constraint_types,
            sampler_fn=sample_flow_heun,
            sampler_kwargs={'n_steps': n_steps},
            n_samples=n_samples, device=device)

    # ══════════════════════════════════════════════════════════════════════
    # H2c: Stochastic Flow (pseudo-Langevin)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("H2c: Stochastic flow (noise injection during sampling)")
    print(f"{'─'*60}")

    for noise_scale in [0.005, 0.01, 0.02, 0.05]:
        name = f"Stochastic 100 steps (σ={noise_scale})"
        print(f"\n  {name}:")
        all_results[name] = evaluate_variant(
            model, constraint_types,
            sampler_fn=sample_flow_stochastic,
            sampler_kwargs={'n_steps': 100, 'noise_scale': noise_scale},
            n_samples=n_samples, device=device)

    # ══════════════════════════════════════════════════════════════════════
    # H3: Later Checkpoint
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("H3: Checkpoint comparison")
    print(f"{'─'*60}")

    for ckpt in ['best', '60', '80', '100']:
        ckpt_path = os.path.join(flow_dir, f'flow_model_{ckpt}.pt')
        if not os.path.isfile(ckpt_path):
            print(f"  Skipping {ckpt} (not found)")
            continue
        model_ckpt = load_model(ckpt)
        name = f"Checkpoint {ckpt} (100 steps)"
        print(f"\n  {name}:")
        all_results[name] = evaluate_variant(
            model_ckpt, constraint_types,
            sampler_fn=sample_flow_variants,
            sampler_kwargs={'n_steps': 100, 'clamp_mode': 'every_step'},
            n_samples=n_samples, device=device,
            test_n_objs=(2, 3))  # Quick check on 2,3 obj only
        del model_ckpt

    # ══════════════════════════════════════════════════════════════════════
    # H4: Constraint Checker Comparison
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("H4: Constraint checker comparison")
    print(f"{'─'*60}")
    try:
        compare_checkers(model, constraint_types, device=device)
    except Exception as e:
        print(f"  Checker comparison failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print_summary(all_results)

    # Save all results
    save_path = './logs/sanity_check_ablations.json'
    with open(save_path, 'w') as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
