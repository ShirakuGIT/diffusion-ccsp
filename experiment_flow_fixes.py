"""
Experiment: Stochastic Flow (SDE) + Energy-Guided Flow vs Diffusion Baselines
===============================================================================
Tests two hypotheses for why pure flow matching underperforms DDPM:

  Idea A — Stochastic Flow (SDE): Add Gaussian noise at each Euler step,
           turning the deterministic ODE into an SDE. This gives exploration
           like DDPM's reverse process.

  Idea B — Energy-Guided Flow: Use per-constraint MLPs to compute energy
           E = Σ_i ||MLP_i(x_t) - x_t||², take autograd gradients, and add
           as a correction to the flow velocity. This mimics ULA composition.

Baselines:
  - Diffusion-CCSP (reverse only, no ULA)
  - Diffusion-CCSP (with ULA)

Usage:
    python experiment_flow_fixes.py
    python experiment_flow_fixes.py --skip_diffusion   # skip slow diffusion baselines
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'networks'))

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from train_flow import FlowMatchingCCSP, get_data_config
from fix_and_eval import clamp_to_tray, check_constraints


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA A: Stochastic Flow (SDE)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_stochastic_flow(model, batch, n_steps=50, sigma_max=0.3,
                            sigma_decay='linear', device='cuda'):
    """
    Flow sampling as SDE: add noise at each step for exploration.

    x_{t+dt} = x_t + v_θ(x_t, t) * dt + σ(t) * √dt * noise

    σ(t) decays over time so we explore early and converge late.
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
    sqrt_dt = dt ** 0.5

    for step in range(n_steps):
        t = step * dt

        # Noise schedule: high early, zero at end
        if sigma_decay == 'linear':
            sigma = sigma_max * (1.0 - t)
        elif sigma_decay == 'cosine':
            sigma = sigma_max * np.cos(t * np.pi / 2)
        else:
            sigma = sigma_max * (1.0 - t)

        v = model(x_t, batch, t)
        noise = torch.randn_like(x_t) if step < n_steps - 1 else 0.0
        x_t = x_t + v * dt + sigma * sqrt_dt * noise
        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA B: Energy-Guided Flow
# ═══════════════════════════════════════════════════════════════════════════════

def compute_flow_energy(model, x_t, batch, t):
    """
    Compute composed energy from per-constraint MLPs.

    E(x) = Σ_c ||v_c(x_t, t)||²

    Each constraint MLP predicts a velocity. The energy is the sum of squared
    velocity norms — this penalises states where constraints are unsatisfied
    (which produce large correction velocities).

    Actually, closer to Diffusion-CCSP's energy:
    E(x) = Σ_edges ||decoded_pose_c - x_t||²

    We compute this by running each constraint MLP and measuring how far its
    output (predicted "clean" velocity) wants to move each node.
    """
    import jactorch

    device = model.device
    t_tensor = model._t_tensor(t, device)
    pose_dim = model.dims[-1][0]

    # Need gradients for x_t
    x_t_grad = x_t.clone().detach().requires_grad_(True)

    geom_emb, pose_emb, edge_index = model._encode(x_t_grad, batch)

    total_energy = torch.tensor(0.0, device=device)

    for i, mlp in enumerate(model.constraint_mlps):
        edge_mask = (batch.edge_attr.to(device) == i)
        if edge_mask.sum() == 0:
            continue
        edges = edge_index[edge_mask]
        src, dst = edges[:, 0], edges[:, 1]
        n_edges = src.shape[0]

        t_emb = model.time_mlp(
            jactorch.add_dim(t_tensor, 0, n_edges)[:, 0])
        inputs = torch.cat([
            geom_emb[src], geom_emb[dst],
            pose_emb[src], pose_emb[dst], t_emb], dim=-1)
        out = mlp(inputs)

        # Decode to pose predictions
        v_src = model.pose_decoder(out[:, :model.hidden_dim])
        v_dst = model.pose_decoder(out[:, model.hidden_dim:])

        # Energy: how far does each constraint want to move each node?
        # This is ||predicted_velocity||² per constraint edge
        total_energy = total_energy + (v_src ** 2).sum() + (v_dst ** 2).sum()

    # Compute gradient of energy w.r.t. input poses
    if total_energy.requires_grad:
        grad = torch.autograd.grad(total_energy, x_t_grad,
                                    retain_graph=False)[0]
    else:
        grad = torch.zeros_like(x_t)

    return total_energy.item(), grad.detach()


def sample_energy_guided_flow(model, batch, n_steps=50, energy_scale=0.5,
                               n_langevin=3, langevin_lr=0.01,
                               device='cuda'):
    """
    Energy-guided flow: Euler step + Langevin correction on composed energy.

    At each step:
      1. x_proposal = x_t + v_θ(x_t, t) * dt           (flow step)
      2. For k Langevin steps:                            (energy correction)
           E, ∇E = compute_energy(x, t)
           x = x - lr * ∇E + noise
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

        # 1. Euler flow step (no grad needed)
        with torch.no_grad():
            v = model(x_t, batch, t)
            x_t = x_t + v * dt
            x_t[mask] = clean_poses[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        # 2. Langevin correction (needs grad)
        # Scale decreases over time — less correction as we converge
        scale = energy_scale * (1.0 - t)
        lr = langevin_lr * scale

        if lr > 1e-6 and step < n_steps - 1:
            for _ in range(n_langevin):
                _, grad = compute_flow_energy(model, x_t, batch, t)
                noise = torch.randn_like(x_t) * (2 * lr) ** 0.5
                x_t = x_t - lr * grad + noise
                x_t[mask] = clean_poses[mask]
                x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sampler(sample_fn, model, constraint_types, n_samples=10,
                     device='cuda', label=""):
    """Generic evaluation for any flow-based sampler."""
    test_tasks = {
        i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
        for i in [2, 3]
    }

    results = {}
    for n_obj, task_name in test_tasks.items():
        print(f"    {n_obj} objects...", end=" ", flush=True)
        dataset = GraphDataset(task_name, input_mode='qualitative',
                               pre_transform=pre_transform, visualize=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        stats = {
            'successes': 0, 'total': 0, 'times': [],
            'per_type': defaultdict(lambda: [0, 0]),
            'scene_succeeded': set(), 'scene_first_try': set(),
        }
        n_scenes = 0

        for si, data in enumerate(loader):
            n_scenes += 1
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj * 100 + si)

                t0 = time.time()
                poses = sample_fn(model, data, device=device)
                elapsed = time.time() - t0
                stats['times'].append(elapsed)

                all_ok, per_c = check_constraints(
                    poses, data, constraint_types, device)

                stats['total'] += 1
                if all_ok:
                    stats['successes'] += 1
                    stats['scene_succeeded'].add(si)
                    if trial == 0:
                        stats['scene_first_try'].add(si)

                for ci, cinfo in per_c.items():
                    stats['per_type'][cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        stats['per_type'][cinfo['type']][0] += 1

        trial_rate = 100 * stats['successes'] / max(stats['total'], 1)
        top1 = 100 * len(stats['scene_first_try']) / max(n_scenes, 1)
        topk = 100 * len(stats['scene_succeeded']) / max(n_scenes, 1)
        avg_ms = 1000 * np.mean(stats['times']) if stats['times'] else 0

        print(f"trial={trial_rate:.1f}%  top1={top1:.1f}%  "
              f"topk={topk:.1f}%  time={avg_ms:.0f}ms")

        # Per-constraint breakdown
        for ct in ['cfree', 'away-from', 'left-of', 'top-of']:
            if ct in stats['per_type']:
                s, t = stats['per_type'][ct]
                print(f"      {ct:14s}: {100*s/t:.1f}% ({s}/{t})")

        results[n_obj] = {
            'trial_rate': trial_rate,
            'scene_top1': top1,
            'scene_topk': topk,
            'avg_time_ms': avg_ms,
            'per_type': {k: list(v) for k, v in stats['per_type'].items()},
        }

    return results


def evaluate_diffusion(ebm_mode, n_tries=10):
    """Run Diffusion-CCSP evaluation with their pipeline."""
    from train_utils import load_trainer

    test_tasks = {
        i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
        for i in [2, 3]
    }

    run_id = 'qsd3ju74'
    milestone = 7

    trainer = load_trainer(run_id, milestone, verbose=False,
                           input_mode='qualitative', test_tasks=test_tasks)

    if ebm_mode == 'none':
        trainer.model.EBM = False
        label = 'diffusion_no_ebm'
    else:
        # Keep ULA as-is
        label = 'diffusion_ula'

    print(f"    EBM={trainer.model.EBM}")

    json_name = f'experiment_{label}'
    trainer.evaluate(json_name, tries=(n_tries, 0), verbose=True, save_log=True)

    # Read results
    json_path = os.path.join(trainer.render_dir,
                              f'denoised_t={json_name}.json')
    results = {}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            log = json.load(f)
        for n_obj_str in ['2', '3']:
            if n_obj_str in log:
                entry = log[n_obj_str]
                n_obj = int(n_obj_str)
                results[n_obj] = {
                    'scene_top1': 100 * entry.get('success_rate', 0),
                    'scene_topk': 100 * entry.get('success_rate_top3',
                                                    entry.get('success_rate', 0)),
                    'trial_rate': 100 * entry.get('success_rate', 0),
                }
                # Try to get per-type
                if 'per_type' in entry:
                    results[n_obj]['per_type'] = entry['per_type']
    else:
        print(f"    WARNING: Could not find {json_path}")

    del trainer
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_diffusion', action='store_true')
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = args.n_samples

    print(f"\n{'#'*70}")
    print(f"# Flow vs Diffusion: Stochastic SDE + Energy-Guided Experiments")
    print(f"# {n_samples} tries per scene, 100 scenes, 2-obj and 3-obj")
    print(f"{'#'*70}")

    _, _, dims, constraint_types = get_data_config('qualitative')

    # Load flow model
    flow_dir = './logs/flow_qualitative_h256'
    ckpt_path = os.path.join(flow_dir, 'flow_model_best.pt')
    flow_model = FlowMatchingCCSP(
        dims=dims, hidden_dim=256, constraint_types=constraint_types,
        normalize=True, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    flow_model.load_state_dict(ckpt['model_state_dict'])
    flow_model.eval()
    print(f"  Loaded flow model: {ckpt_path}")

    all_results = {}

    # ── 1. Pure Flow (20 steps, deterministic) — baseline ──
    print(f"\n{'='*60}")
    print(f"  1. Pure Flow (20 steps, deterministic ODE)")
    print(f"{'='*60}")

    def pure_flow_fn(model, data, device='cuda'):
        from fix_and_eval import sample_flow_fixed
        return sample_flow_fixed(model, data, n_steps=20, device=device)

    all_results['Pure Flow (20 steps)'] = evaluate_sampler(
        pure_flow_fn, flow_model, constraint_types,
        n_samples=n_samples, device=device)

    # ── 2. Stochastic Flow SDE (50 steps) ──
    print(f"\n{'='*60}")
    print(f"  2. Stochastic Flow SDE (50 steps, σ=0.3)")
    print(f"{'='*60}")

    def sde_flow_fn(model, data, device='cuda'):
        return sample_stochastic_flow(model, data, n_steps=50,
                                       sigma_max=0.3, device=device)

    all_results['Stochastic Flow SDE (50)'] = evaluate_sampler(
        sde_flow_fn, flow_model, constraint_types,
        n_samples=n_samples, device=device)

    # ── 3. Stochastic Flow SDE (100 steps) ──
    print(f"\n{'='*60}")
    print(f"  3. Stochastic Flow SDE (100 steps, σ=0.3)")
    print(f"{'='*60}")

    def sde_flow_100_fn(model, data, device='cuda'):
        return sample_stochastic_flow(model, data, n_steps=100,
                                       sigma_max=0.3, device=device)

    all_results['Stochastic Flow SDE (100)'] = evaluate_sampler(
        sde_flow_100_fn, flow_model, constraint_types,
        n_samples=n_samples, device=device)

    # ── 4. Energy-Guided Flow ──
    print(f"\n{'='*60}")
    print(f"  4. Energy-Guided Flow (50 steps, 3 Langevin/step)")
    print(f"{'='*60}")

    def energy_flow_fn(model, data, device='cuda'):
        return sample_energy_guided_flow(model, data, n_steps=50,
                                          energy_scale=0.5, n_langevin=3,
                                          langevin_lr=0.01, device=device)

    all_results['Energy-Guided Flow (50)'] = evaluate_sampler(
        energy_flow_fn, flow_model, constraint_types,
        n_samples=n_samples, device=device)

    # ── 5. Diffusion baselines ──
    if not args.skip_diffusion:
        print(f"\n{'='*60}")
        print(f"  5. Diffusion-CCSP (no ULA)")
        print(f"{'='*60}")
        all_results['Diffusion (no ULA)'] = evaluate_diffusion(
            'none', n_tries=n_samples)

        print(f"\n{'='*60}")
        print(f"  6. Diffusion-CCSP (ULA)")
        print(f"{'='*60}")
        all_results['Diffusion (ULA)'] = evaluate_diffusion(
            'ULA', n_tries=n_samples)

    # ── Print comparison ──
    print(f"\n\n{'='*80}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*80}")

    header = f"{'Method':<30} | {'2-obj top1':>10} {'2-obj topk':>10} | {'3-obj top1':>10} {'3-obj topk':>10} | {'2-obj ms':>8}"
    print(f"\n{header}")
    print("-" * 95)

    for name, res in all_results.items():
        t1_2 = res.get(2, {}).get('scene_top1', 0)
        tk_2 = res.get(2, {}).get('scene_topk', 0)
        t1_3 = res.get(3, {}).get('scene_top1', 0)
        tk_3 = res.get(3, {}).get('scene_topk', 0)
        ms = res.get(2, {}).get('avg_time_ms', 0)
        print(f"{name:<30} | {t1_2:>9.1f}% {tk_2:>9.1f}% | {t1_3:>9.1f}% {tk_3:>9.1f}% | {ms:>7.0f}")

    # Per-constraint breakdown
    print(f"\n{'─'*80}")
    print(f"  Per-constraint breakdown (3-obj):")
    print(f"{'─'*80}")
    for name, res in all_results.items():
        if 3 not in res or 'per_type' not in res[3]:
            continue
        pt = res[3]['per_type']
        parts = []
        for ct in ['cfree', 'away-from', 'left-of', 'top-of']:
            if ct in pt:
                s, total = pt[ct]
                parts.append(f"{ct}={100*s/total:.0f}%")
        print(f"  {name:<30}: {', '.join(parts)}")

    print(f"{'='*80}")

    # Save
    save_path = './logs/experiment_flow_fixes.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
