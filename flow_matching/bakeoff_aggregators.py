"""
5000-step Bake-Off: sum vs attention vs maxpool aggregation for Flow-CCSP.
===========================================================================
Tests whether attention-weighted or max-pooling aggregation breaks the
colocation singularity where opposing constraint velocities cancel to zero.

Trains 3 models for 5000 steps each on the full qualitative training set,
then evaluates on 2-obj and 3-obj test sets (100 scenes, 10 trials).
Focus metric: away-from constraint success rate.

Usage:
    python bakeoff_aggregators.py
    python bakeoff_aggregators.py --steps 10000 --hidden_dim 256
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
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'networks'))

from torch_geometric.loader import DataLoader

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from train_flow import FlowMatchingCCSP, get_data_config, _sample_flow_simple
from fix_and_eval import check_constraints, sample_flow_fixed


def train_model(aggregator, dims, constraint_types, train_dataset,
                hidden_dim=256, lr=5e-4, batch_size=128, n_steps=5000,
                device='cuda'):
    """Train a FlowMatchingCCSP model with the given aggregator."""
    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device,
        aggregator=aggregator).to(device)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {aggregator}: {n_p:,} params")

    opt = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                    pin_memory=True, num_workers=0)
    dl_iter = iter(dl)

    model.train()
    losses = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        opt.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % 1000 == 0:
            avg = np.mean(losses[-200:])
            elapsed = time.time() - t0
            print(f"      step {step}/{n_steps}  loss={avg:.5f}  "
                  f"({elapsed:.0f}s)")

    avg_loss = np.mean(losses[-500:])
    print(f"    Final loss: {avg_loss:.5f} ({time.time()-t0:.0f}s total)")
    return model, avg_loss


def evaluate_model(model, constraint_types, n_samples=10, device='cuda',
                   n_steps=20):
    """Evaluate on 2-obj and 3-obj test sets."""
    results = {}

    for n_obj in [2, 3]:
        task = f"RandomSplitQualitativeWorld(100)_qualitative_test_{n_obj}_split"
        ds = GraphDataset(task, input_mode='qualitative',
                          pre_transform=pre_transform, visualize=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        stats = {
            'successes': 0, 'total': 0,
            'per_type': defaultdict(lambda: [0, 0]),
            'scene_succeeded': set(),
        }
        n_scenes = 0

        for si, data in enumerate(loader):
            n_scenes += 1
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj * 100 + si)

                with torch.no_grad():
                    poses = sample_flow_fixed(model, data, n_steps=n_steps,
                                             device=device)

                all_ok, per_c = check_constraints(
                    poses, data, constraint_types, device)

                stats['total'] += 1
                if all_ok:
                    stats['successes'] += 1
                    stats['scene_succeeded'].add(si)

                for ci, cinfo in per_c.items():
                    stats['per_type'][cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        stats['per_type'][cinfo['type']][0] += 1

        trial_rate = 100 * stats['successes'] / stats['total']
        scene_rate = 100 * len(stats['scene_succeeded']) / n_scenes

        print(f"    {n_obj}-obj: trial={trial_rate:.1f}%  "
              f"scene={scene_rate:.1f}%")

        # Print key constraint types
        for ct in ['away-from', 'cfree', 'left-of', 'top-of']:
            if ct in stats['per_type']:
                s, t = stats['per_type'][ct]
                print(f"      {ct:14s}: {100*s/t:.1f}% ({s}/{t})")

        results[n_obj] = {
            'trial_rate': trial_rate,
            'scene_rate': scene_rate,
            'per_type': {k: list(v) for k, v in stats['per_type'].items()},
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_mode = 'qualitative'

    print(f"\n{'#'*70}")
    print(f"# Aggregator Bake-Off: sum vs attention vs maxpool")
    print(f"# Steps={args.steps}, Hidden={args.hidden_dim}, "
          f"LR={args.lr}, BS={args.batch_size}")
    print(f"{'#'*70}")

    _, _, dims, constraint_types = get_data_config(input_mode)

    # Load training data
    train_task = "RandomSplitQualitativeWorld(30000)_qualitative_train"
    train_ds = GraphDataset(train_task, input_mode=input_mode,
                            pre_transform=pre_transform, visualize=False)
    print(f"  Train set: {len(train_ds):,} scenes")

    all_results = {}

    for aggregator in ['sum', 'attention', 'maxpool']:
        print(f"\n{'='*60}")
        print(f"  Training: {aggregator}")
        print(f"{'='*60}")

        model, final_loss = train_model(
            aggregator, dims, constraint_types, train_ds,
            hidden_dim=args.hidden_dim, lr=args.lr,
            batch_size=args.batch_size, n_steps=args.steps,
            device=device)

        # Save checkpoint
        save_dir = f'./logs/bakeoff_{aggregator}_h{args.hidden_dim}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'aggregator': aggregator,
            'step': args.steps,
            'loss': final_loss,
        }, save_path)
        print(f"    Saved: {save_path}")

        print(f"\n  Evaluating: {aggregator}")
        eval_results = evaluate_model(
            model, constraint_types, n_samples=args.n_samples,
            device=device)
        eval_results['final_loss'] = final_loss
        all_results[aggregator] = eval_results

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  BAKE-OFF RESULTS ({args.steps} steps)")
    print(f"{'='*80}")

    header = f"{'Aggregator':>12} | {'Loss':>8}"
    for n_obj in [2, 3]:
        header += f" | {n_obj}-obj trial"
        header += f" | {n_obj}-obj away"
        header += f" | {n_obj}-obj cfree"
    print(f"\n{header}")
    print("-" * 100)

    for agg in ['sum', 'attention', 'maxpool']:
        r = all_results[agg]
        line = f"{agg:>12} | {r['final_loss']:>8.5f}"
        for n_obj in [2, 3]:
            nr = r.get(n_obj, {})
            trial = nr.get('trial_rate', 0)
            away = 0
            cfree = 0
            pt = nr.get('per_type', {})
            if 'away-from' in pt and pt['away-from'][1] > 0:
                away = 100 * pt['away-from'][0] / pt['away-from'][1]
            if 'cfree' in pt and pt['cfree'][1] > 0:
                cfree = 100 * pt['cfree'][0] / pt['cfree'][1]
            line += f" |    {trial:>5.1f}%"
            line += f" |    {away:>5.1f}%"
            line += f" |    {cfree:>5.1f}%"
        print(line)

    print(f"\n  Success criterion: away-from on 3-obj goes from 0% (sum)")
    print(f"  to >20% (attention/maxpool)")
    print(f"{'='*80}")

    # Save results
    save_path = './logs/bakeoff_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
