"""
Sanity Check: Pure Diffusion (Reverse) vs Pure Flow Matching
=============================================================
Compare the two base generative models WITHOUT any global iterative correction:
  - Diffusion-CCSP (Reverse): standard reverse diffusion, NO ULA/Langevin
  - Flow Matching: ODE integration only, NO CBF-QP projection

This tests whether the base model alone can solve compositional constraints,
or whether a global iterative solver (ULA for diffusion, CBF-QP for flow)
is fundamentally necessary.

Usage:
    python sanity_check_pure.py
"""

import os
import sys
import time
import json
import numpy as np
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader

# Add local modules

from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from flow_matching.train_flow import FlowMatchingCCSP, get_data_config
from flow_matching.fix_and_eval import clamp_to_tray, check_constraints, sample_flow_fixed


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFUSION (REVERSE ONLY — no ULA)
# ═══════════════════════════════════════════════════════════════════════════════

def load_diffusion_reverse_only(run_id='qsd3ju74', milestone=7):
    """
    Load the Diffusion-CCSP model and force EBM=False so that
    p_sample_loop uses standard reverse diffusion (no ULA/Langevin).
    """
    from train_utils import load_trainer

    test_tasks = {i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
                  for i in range(2, 6)}

    trainer = load_trainer(run_id, milestone, verbose=False,
                           input_mode='qualitative', test_tasks=test_tasks)

    # CRITICAL: override EBM to False — pure reverse diffusion, no ULA
    trainer.model.EBM = False
    print(f"  Loaded Diffusion-CCSP (run={run_id}, milestone={milestone})")
    print(f"  EBM set to: {trainer.model.EBM} (pure reverse diffusion)")
    return trainer


def evaluate_diffusion_reverse(trainer, n_tries=10, verbose=True):
    """
    Evaluate Diffusion-CCSP with REVERSE ONLY (EBM=False).
    Uses their own evaluation pipeline for apples-to-apples comparison.

    Returns scene-level results:
      - top1: fraction of scenes solved on first try
      - topk: fraction of scenes solved in at least 1 of n_tries attempts
    """
    print(f"\n  Running Diffusion (Reverse Only) with {n_tries} tries...")
    trainer.evaluate('sanity_reverse', tries=(n_tries, 0),
                     verbose=verbose, save_log=True)

    # Read the saved log
    json_path = os.path.join(trainer.render_dir, 'denoised_t=sanity_reverse.json')
    if os.path.isfile(json_path):
        with open(json_path) as f:
            log = json.load(f)
        return log
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW MATCHING (PURE — no QP)
# ═══════════════════════════════════════════════════════════════════════════════

def load_flow_model(hidden_dim=256, input_mode='qualitative', device='cuda'):
    """Load the Flow Matching model."""
    _, _, dims, constraint_types = get_data_config(input_mode)
    flow_dir = f'./logs/flow_{input_mode}_h{hidden_dim}'
    ckpt_path = os.path.join(flow_dir, 'flow_model_best.pt')

    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded Flow model: {ckpt_path}")
    return model, constraint_types


def evaluate_flow_pure(model, constraint_types, input_mode='qualitative',
                       n_samples=10, n_steps=20, device='cuda'):
    """
    Evaluate Flow Matching with pure ODE integration (no QP).
    Uses per-trial success evaluation (matching fix_and_eval.py).
    Also computes scene-level top-1/top-k for fair comparison with diffusion.
    """
    test_tasks = {i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
                  for i in range(2, 6)}

    results = {}
    for n_obj, task_name in test_tasks.items():
        print(f"    {n_obj} objects...", end=" ", flush=True)
        dataset = GraphDataset(task_name, input_mode=input_mode,
                               pre_transform=pre_transform, visualize=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        per_trial = {'successes': 0, 'total': 0, 'times': [],
                     'per_type': defaultdict(lambda: [0, 0]),
                     'constraint_sat_rates': []}
        # Scene-level tracking (to match diffusion metrics)
        scene_succeeded = set()
        scene_first_try = set()
        n_scenes = 0

        for scene_idx, data in enumerate(loader):
            n_scenes += 1
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj * 100 + scene_idx)

                t0 = time.time()
                with torch.no_grad():
                    poses = sample_flow_fixed(model, data, n_steps=n_steps,
                                             device=device)
                elapsed = time.time() - t0
                per_trial['times'].append(elapsed)

                all_ok, per_c = check_constraints(
                    poses, data, constraint_types, device)

                per_trial['total'] += 1
                if all_ok:
                    per_trial['successes'] += 1
                    scene_succeeded.add(scene_idx)
                    if trial == 0:
                        scene_first_try.add(scene_idx)

                n_sat = sum(1 for v in per_c.values() if v['satisfied'])
                per_trial['constraint_sat_rates'].append(
                    n_sat / len(per_c) if per_c else 0)

                for ci, cinfo in per_c.items():
                    per_trial['per_type'][cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        per_trial['per_type'][cinfo['type']][0] += 1

        trial_rate = 100 * per_trial['successes'] / per_trial['total']
        scene_top1 = 100 * len(scene_first_try) / n_scenes if n_scenes else 0
        scene_topk = 100 * len(scene_succeeded) / n_scenes if n_scenes else 0
        avg_time = 1000 * np.mean(per_trial['times'])

        print(f"  trial={trial_rate:.1f}%  scene_top1={scene_top1:.1f}%  "
              f"scene_topk={scene_topk:.1f}%  time={avg_time:.0f}ms")

        results[n_obj] = {
            'per_trial': per_trial,
            'n_scenes': n_scenes,
            'scene_top1': scene_top1,
            'scene_topk': scene_topk,
            'avg_time_ms': avg_time,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(diff_log, flow_results, n_tries=10):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("  SANITY CHECK: Pure Diffusion (Reverse) vs Pure Flow Matching")
    print("  No ULA, No CBF-QP — just the base generative models")
    print("=" * 80)

    header = (f"{'n_obj':>5}  |  {'Diffusion (Reverse)':^30}  |  "
              f"{'Flow Matching (Pure)':^30}")
    sub = (f"{'':>5}  |  {'top1':>6} {'topk':>6} {'time':>8}  |  "
           f"{'top1':>6} {'topk':>6} {'trial':>6} {'time':>8}")
    print(f"\n{header}")
    print(f"{sub}")
    print("-" * 80)

    for n_obj in range(2, 6):
        # Diffusion results from their log
        d_top1, d_topk, d_time = '—', '—', '—'
        if diff_log and str(n_obj) in diff_log:
            entry = diff_log[str(n_obj)]
            d_top1 = f"{100*entry.get('success_rate', 0):.1f}%"
            d_topk = f"{100*entry.get('success_rate_top3', 0):.1f}%"
            times = entry.get('sampling_time', [])
            if times:
                d_time = f"{1000*np.mean([t[2] for t in times]):.0f}ms"

        # Flow results
        f_top1, f_topk, f_trial, f_time = '—', '—', '—', '—'
        if n_obj in flow_results:
            fr = flow_results[n_obj]
            f_top1 = f"{fr['scene_top1']:.1f}%"
            f_topk = f"{fr['scene_topk']:.1f}%"
            pt = fr['per_trial']
            f_trial = f"{100*pt['successes']/pt['total']:.1f}%"
            f_time = f"{fr['avg_time_ms']:.0f}ms"

        print(f"{n_obj:>5}  |  {d_top1:>6} {d_topk:>6} {d_time:>8}  |  "
              f"{f_top1:>6} {f_topk:>6} {f_trial:>6} {f_time:>8}")

    # Also print per-constraint-type breakdown for flow
    print("\n" + "-" * 80)
    print("  Flow Matching — per-constraint satisfaction rates:")
    for n_obj in sorted(flow_results.keys()):
        pt = flow_results[n_obj]['per_trial']
        print(f"\n  {n_obj} objects:")
        for ct in sorted(pt['per_type'].keys()):
            s, t = pt['per_type'][ct]
            print(f"    {ct:14s}: {100*s/t:5.1f}% ({s}/{t})")

    # Reference: Diffusion + ULA numbers
    print("\n" + "-" * 80)
    print("  Reference: Diffusion-CCSP (ULA, 10 tries) — from previous eval:")
    print("    2-obj: 96%, 3-obj: 92%, 4-obj: 70%, 5-obj: 44%")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_tries = 10

    print(f"\n{'#'*70}")
    print(f"# SANITY CHECK: Pure generative model comparison")
    print(f"# Diffusion (Reverse, no ULA) vs Flow Matching (no QP)")
    print(f"# {n_tries} tries per scene, 100 scenes per n_obj")
    print(f"{'#'*70}")

    # ── Step 1: Evaluate Diffusion (Reverse Only) ──
    print(f"\n{'─'*50}")
    print("Step 1: Diffusion-CCSP (Reverse Only, no ULA)")
    print(f"{'─'*50}")
    trainer = load_diffusion_reverse_only()
    diff_log = evaluate_diffusion_reverse(trainer, n_tries=n_tries, verbose=True)
    del trainer
    torch.cuda.empty_cache()

    # ── Step 2: Evaluate Flow Matching (Pure) ──
    print(f"\n{'─'*50}")
    print("Step 2: Flow Matching (Pure, no QP)")
    print(f"{'─'*50}")
    flow_model, constraint_types = load_flow_model(device=device)
    flow_results = evaluate_flow_pure(
        flow_model, constraint_types, n_samples=n_tries, device=device)

    # Save flow results
    save_path = './logs/sanity_check_pure_flow.json'
    serializable = {}
    for k, v in flow_results.items():
        sv = {kk: vv for kk, vv in v.items() if kk != 'per_trial'}
        pt = v['per_trial']
        sv['per_trial'] = {
            'successes': pt['successes'],
            'total': pt['total'],
            'times': pt['times'],
            'constraint_sat_rates': pt['constraint_sat_rates'],
            'per_type': {ct: list(counts) for ct, counts in pt['per_type'].items()},
        }
        serializable[k] = sv
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {save_path}")

    # ── Step 3: Print comparison ──
    print_comparison(diff_log, flow_results, n_tries=n_tries)


if __name__ == '__main__':
    main()
