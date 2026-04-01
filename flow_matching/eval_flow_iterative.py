"""
Evaluate pure flow vs iterative restart flow.

Direction B:
  treat the current flow sampler as a reusable correction primitive, then
  restart from its own output with a small perturbation to get diffusion-like
  globally informed refinement without retraining.

Usage:
    python eval_flow_iterative.py --model baseline
    python eval_flow_iterative.py --model baseline --compare_pure
    python eval_flow_iterative.py --model message_passing --n_rounds 3
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'envs'))
sys.path.insert(0, str(ROOT / 'networks'))
sys.path.insert(0, str(ROOT.parent / 'Jacinle'))

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from fix_and_eval import clamp_to_tray, check_constraints
from train_flow import FlowMatchingCCSP, get_best_device, get_data_config
from flow_message_passing import MessagePassingFlowMatchingCCSP


def validate_dataset_dir(task_name):
    root = ROOT / 'data' / task_name
    raw_dir = root / 'raw'
    processed_dir = root / 'processed'

    raw_files = list(raw_dir.glob('*')) if raw_dir.exists() else []
    processed_files = list(processed_dir.glob('*')) if processed_dir.exists() else []

    if raw_files or processed_files:
        return

    raise FileNotFoundError(
        f'Dataset "{task_name}" is empty or missing.\n'
        f'Checked: {raw_dir} and {processed_dir}\n'
        f'Populate the dataset first, e.g. run `python download_data_checkpoints.py` '
        f'or generate the test splits before evaluation.'
    )


@torch.no_grad()
def sample_flow(model, batch, n_steps=20, device='cuda'):
    model.eval()
    batch = batch.to(device)

    x = batch.x.to(device)
    pose_dim = model.dims[-1][0]
    pose_begin = model.dims[-1][1]
    pose_end = model.dims[-1][2]
    geom_end = model.dims[0][2]
    mask = batch.mask.bool().to(device)
    clean_poses = x[:, pose_begin:pose_end]
    geoms = x[:, :geom_end]

    x_t = torch.randn_like(clean_poses)
    x_t[mask] = clean_poses[mask]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        v = model(x_t, batch, step * dt)
        x_t = x_t + v * dt
        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

    return x_t


@torch.no_grad()
def sample_iterative_restart_flow(model, batch, n_steps=20, n_restarts=3,
                                  restart_noise=0.10, noise_decay=0.5,
                                  device='cuda'):
    model.eval()
    batch = batch.to(device)

    x = batch.x.to(device)
    pose_dim = model.dims[-1][0]
    pose_begin = model.dims[-1][1]
    pose_end = model.dims[-1][2]
    geom_end = model.dims[0][2]
    mask = batch.mask.bool().to(device)
    clean_poses = x[:, pose_begin:pose_end]
    geoms = x[:, :geom_end]

    x_prev = None
    for restart_idx in range(n_restarts):
        if x_prev is None:
            x_t = torch.randn_like(clean_poses)
        else:
            sigma = restart_noise * (noise_decay ** max(restart_idx - 1, 0))
            x_t = x_prev + sigma * torch.randn_like(x_prev)

        x_t[mask] = clean_poses[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        dt = 1.0 / n_steps
        for step in range(n_steps):
            v = model(x_t, batch, step * dt)
            x_t = x_t + v * dt
            x_t[mask] = clean_poses[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        x_prev = x_t

    return x_prev


def build_model(args, dims, constraint_types, device):
    if args.model == 'baseline':
        return FlowMatchingCCSP(
            dims=dims,
            hidden_dim=args.hidden_dim,
            constraint_types=constraint_types,
            normalize=True,
            device=device,
        ).to(device)

    return MessagePassingFlowMatchingCCSP(
        dims=dims,
        hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True,
        device=device,
        n_rounds=args.n_rounds,
    ).to(device)


def default_checkpoint_path(args):
    if args.checkpoint:
        return args.checkpoint
    if args.model == 'baseline':
        return os.path.join(
            f'./logs/flow_{args.input_mode}_h{args.hidden_dim}',
            f'flow_model_{args.checkpoint_tag}.pt',
        )
    return os.path.join(
        f'./logs/flow_mp_{args.input_mode}_h{args.hidden_dim}_r{args.n_rounds}',
        f'flow_model_{args.checkpoint_tag}.pt',
    )


def evaluate_sampler(model, sampler_fn, constraint_types, input_mode='qualitative',
                     n_samples=10, device='cuda', label='', verbose=False):
    _, test_tasks, _, _ = get_data_config(input_mode)

    results = {}
    for n_obj, task_name in test_tasks.items():
        if not os.path.isdir(f'./data/{task_name}'):
            continue
        validate_dataset_dir(task_name)
        print(f'  {label} | {n_obj} objects...', end=' ', flush=True)
        dataset = GraphDataset(task_name, input_mode=input_mode,
                               pre_transform=pre_transform, visualize=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        stats = {
            'successes': 0,
            'total': 0,
            'times': [],
            'per_type': defaultdict(lambda: [0, 0]),
            'constraint_sat_rates': [],
        }
        scene_succeeded = set()
        scene_first_try = set()
        n_scenes = 0

        for scene_idx, data in enumerate(loader):
            n_scenes += 1
            for trial in range(n_samples):
                torch.manual_seed(trial * 1000 + n_obj * 100 + scene_idx)

                t0 = time.time()
                poses = sampler_fn(model, data, device=device)
                stats['times'].append(time.time() - t0)

                all_ok, per_c = check_constraints(poses, data, constraint_types, device)
                stats['total'] += 1
                if all_ok:
                    stats['successes'] += 1
                    scene_succeeded.add(scene_idx)
                    if trial == 0:
                        scene_first_try.add(scene_idx)

                n_sat = sum(1 for _, cinfo in per_c.items() if cinfo['satisfied'])
                stats['constraint_sat_rates'].append(
                    n_sat / max(len(per_c), 1) if per_c else 0.0)
                for _, cinfo in per_c.items():
                    stats['per_type'][cinfo['type']][1] += 1
                    if cinfo['satisfied']:
                        stats['per_type'][cinfo['type']][0] += 1

        trial_rate = 100.0 * stats['successes'] / max(stats['total'], 1)
        top1 = 100.0 * len(scene_first_try) / max(n_scenes, 1)
        topk = 100.0 * len(scene_succeeded) / max(n_scenes, 1)
        avg_time = 1000.0 * np.mean(stats['times']) if stats['times'] else 0.0
        avg_sat = 100.0 * np.mean(stats['constraint_sat_rates']) if stats['constraint_sat_rates'] else 0.0

        print(f'trial={trial_rate:.1f}%  top1={top1:.1f}%  topk={topk:.1f}%  '
              f'avg_sat={avg_sat:.1f}%  time={avg_time:.0f}ms')
        if verbose:
            print('    per-constraint:')
            for ct in sorted(stats['per_type'].keys()):
                sat, tot = stats['per_type'][ct]
                print(f'      {ct:14s}: {100.0 * sat / max(tot, 1):5.1f}% ({sat}/{tot})')
        results[n_obj] = {
            'trial_rate': trial_rate,
            'scene_top1': top1,
            'scene_topk': topk,
            'avg_time_ms': avg_time,
            'avg_constraint_sat': avg_sat,
            'stats': stats,
        }

    return results


def print_summary(name, results):
    print(f'\n{name}')
    print('-' * len(name))
    for n_obj in sorted(results.keys()):
        r = results[n_obj]
        print(f"  {n_obj} obj: trial={r['trial_rate']:.1f}%  "
              f"top1={r['scene_top1']:.1f}%  topk={r['scene_topk']:.1f}%  "
              f"avg_sat={r['avg_constraint_sat']:.1f}%  "
              f"time={r['avg_time_ms']:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description='Evaluate iterative restart flow')
    parser.add_argument('--model', choices=['baseline', 'message_passing'], default='baseline')
    parser.add_argument('--input_mode', default='qualitative')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_rounds', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_tag', type=str, default='best')
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--restart_noise', type=float, default=0.10)
    parser.add_argument('--noise_decay', type=float, default=0.5)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--compare_pure', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = get_best_device(args.device)
    _, _, dims, constraint_types = get_data_config(args.input_mode)

    model = build_model(args, dims, constraint_types, device)
    ckpt_path = default_checkpoint_path(args)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Loaded {args.model} checkpoint: {ckpt_path}')
    print(f'Config: model={args.model} input_mode={args.input_mode} hidden_dim={args.hidden_dim} '
          f'n_rounds={args.n_rounds}')
    print(f'Sampler: n_steps={args.n_steps} n_restarts={args.n_restarts} '
          f'restart_noise={args.restart_noise} noise_decay={args.noise_decay}')
    print(f'Eval   : n_samples={args.n_samples} device={device} verbose={args.verbose}')

    all_results = {}

    if args.compare_pure:
        def pure_sampler(m, data, device='cuda'):
            return sample_flow(m, data, n_steps=args.n_steps, device=device)

        all_results['Pure Flow'] = evaluate_sampler(
            model, pure_sampler, constraint_types,
            input_mode=args.input_mode,
            n_samples=args.n_samples,
            device=device,
            label='Pure',
            verbose=args.verbose,
        )

    def iterative_sampler(m, data, device='cuda'):
        return sample_iterative_restart_flow(
            m, data,
            n_steps=args.n_steps,
            n_restarts=args.n_restarts,
            restart_noise=args.restart_noise,
            noise_decay=args.noise_decay,
            device=device,
        )

    all_results['Iterative Restart Flow'] = evaluate_sampler(
        model, iterative_sampler, constraint_types,
        input_mode=args.input_mode,
        n_samples=args.n_samples,
        device=device,
        label='Restart',
        verbose=args.verbose,
    )

    for name, results in all_results.items():
        print_summary(name, results)


if __name__ == '__main__':
    main()
