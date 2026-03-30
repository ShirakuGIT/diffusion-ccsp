"""
Evaluate the learned optimizer against the diffusion baseline.

Examples:
    python eval_learned_optimizer.py --checkpoint logs/learned_optimizer_v1/best.pt
    python eval_learned_optimizer.py --checkpoint logs/learned_optimizer_v1/best.pt --diffusion_run_id qsd3ju74 --diffusion_milestone 7
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'envs'))
sys.path.insert(0, str(ROOT / 'networks'))

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from fix_and_eval import check_constraints
from models.learned_optimizer import LearnedOptimizer, solve
from train_flow_v3 import compute_constraint_violations, get_data_config
from train_learned_optimizer import get_pose_dim, initialize_poses, maybe_project_to_tray
from train_utils import load_trainer


def compute_constraint_violation(x, batch, constraint_types, device):
    loss, _ = compute_constraint_violations(x, batch, constraint_types, device=device)
    return float(loss.item())


def summarize(results):
    if results['total'] == 0:
        return {
            'avg_violation': float('nan'),
            'total_violation': 0.0,
            'success_rate': 0.0,
            'avg_runtime_ms': 0.0,
            'count': 0,
        }

    return {
        'avg_violation': results['total_violation'] / results['total'],
        'total_violation': results['total_violation'],
        'success_rate': results['successes'] / results['total'],
        'avg_runtime_ms': 1000.0 * np.mean(results['times']),
        'count': results['total'],
    }


@torch.no_grad()
def evaluate_learned(model, loader, constraint_types, device, solve_steps):
    model.eval()
    results = {'total': 0, 'successes': 0, 'total_violation': 0.0, 'times': []}

    for batch in loader:
        batch = batch.to(device)
        pose_dim = get_pose_dim(batch)
        x_init = initialize_poses(batch, pose_dim, device=device)

        t0 = time.time()
        x_final = solve(model, x_init, batch, steps=solve_steps)
        x_final = maybe_project_to_tray(x_final, batch)
        elapsed = time.time() - t0

        violation = compute_constraint_violation(x_final, batch, constraint_types, device=device)
        success, _ = check_constraints(x_final, batch, constraint_types, device=device)

        results['total'] += 1
        results['successes'] += int(success)
        results['total_violation'] += violation
        results['times'].append(elapsed)

    return summarize(results)


@torch.no_grad()
def evaluate_diffusion(diffusion, loader, constraint_types, device):
    results = {'total': 0, 'successes': 0, 'total_violation': 0.0, 'times': []}

    for batch in loader:
        batch = batch.to(device)

        t0 = time.time()
        x_final = diffusion.sample(batch)
        elapsed = time.time() - t0

        violation = compute_constraint_violation(x_final, batch, constraint_types, device=device)
        success, _ = check_constraints(x_final, batch, constraint_types, device=device)

        results['total'] += 1
        results['successes'] += int(success)
        results['total_violation'] += violation
        results['times'].append(elapsed)

    return summarize(results)


def load_learned_optimizer(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _, _, dims, constraint_types = get_data_config('qualitative')
    args = checkpoint.get('args', {})
    hidden_dim = args.get('hidden_dim', 256)

    model = LearnedOptimizer(
        pose_dim=dims[-1][0],
        num_edge_types=len(constraint_types),
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, constraint_types


def print_summary(name, metrics):
    print(
        f'{name:20s} '
        f'avg_violation={metrics["avg_violation"]:.6f} '
        f'success_rate={100.0 * metrics["success_rate"]:.1f}% '
        f'avg_runtime={metrics["avg_runtime_ms"]:.2f}ms '
        f'n={metrics["count"]}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--eval_task', type=str, default=None)
    parser.add_argument('--solve_steps', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--diffusion_run_id', type=str, default=None)
    parser.add_argument('--diffusion_milestone', type=int, default=None)
    parser.add_argument('--output_json', type=str, default='logs/learned_optimizer_v1/eval_results.json')
    args = parser.parse_args()

    _, test_tasks, _, constraint_types = get_data_config('qualitative')
    eval_task = args.eval_task or test_tasks[min(test_tasks.keys())]

    dataset = GraphDataset(
        eval_task,
        input_mode='qualitative',
        pre_transform=pre_transform,
        visualize=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model, constraint_types = load_learned_optimizer(args.checkpoint, args.device)
    learned_metrics = evaluate_learned(
        model,
        loader,
        constraint_types=constraint_types,
        device=args.device,
        solve_steps=args.solve_steps,
    )
    print_summary('learned_optimizer', learned_metrics)

    results = {
        'eval_task': eval_task,
        'learned_optimizer': learned_metrics,
    }

    if args.diffusion_run_id and args.diffusion_milestone is not None:
        trainer = load_trainer(
            args.diffusion_run_id,
            args.diffusion_milestone,
            verbose=False,
            input_mode='qualitative',
            test_tasks={0: eval_task},
        )
        diffusion_metrics = evaluate_diffusion(
            trainer.model,
            loader,
            constraint_types=constraint_types,
            device=args.device,
        )
        print_summary('diffusion_baseline', diffusion_metrics)
        results['diffusion_baseline'] = diffusion_metrics

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'saved {output_path}')


if __name__ == '__main__':
    main()
