"""
Minimal training loop for a learned graph optimizer that directly minimizes
constraint violations without trajectory or target supervision.

Example:
    python train_learned_optimizer.py --train_steps 10000 --unroll_steps 2
"""

import argparse
import json
import os
import random
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim import Adam
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent

from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints
from flow_matching.fix_and_eval import clamp_to_tray
from models.learned_optimizer import LearnedOptimizer, apply_fixed_poses, solve
from flow_matching.train_flow_v3 import compute_constraint_violations, get_data_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pose_dim(batch):
    if hasattr(batch, 'mask') and hasattr(batch, 'x'):
        return batch.x.shape[-1] - 2
    return batch.x.shape[-1]


def get_clean_pose_targets(batch, pose_dim, device):
    if hasattr(batch, 'x') and batch.x.size(-1) >= pose_dim:
        return batch.x[:, -pose_dim:].to(device)
    raise ValueError('Batch does not expose pose targets needed for conditioned nodes.')


def initialize_poses(batch, pose_dim, device, noise_scale=0.5):
    x0 = noise_scale * torch.randn(batch.num_nodes, pose_dim, device=device)
    if hasattr(batch, 'mask'):
        fixed_mask = batch.mask.bool().to(device)
        fixed_values = get_clean_pose_targets(batch, pose_dim, device)
        x0[fixed_mask] = fixed_values[fixed_mask]
    return x0


def maybe_project_to_tray(x, batch):
    if not hasattr(batch, 'mask'):
        return x
    if not hasattr(batch, 'x') or batch.x.size(-1) < x.size(-1) + 2:
        return x
    geoms = batch.x[:, :2].to(x.device)
    return clamp_to_tray(x, geoms, batch.mask.bool().to(x.device), pose_dim=x.size(-1))


def compute_constraint_violation(x, batch, constraint_types, device):
    loss, _ = compute_constraint_violations(x, batch, constraint_types, device=device)
    return loss


def rollout(model, batch, unroll_steps, noise_scale, device, step_size=0.1, x_init=None):
    pose_dim = get_pose_dim(batch)
    if x_init is None:
        x = initialize_poses(batch, pose_dim, device=device, noise_scale=noise_scale)
    else:
        x = x_init.clone()
    fixed_mask = batch.mask.bool().to(device) if hasattr(batch, 'mask') else None
    fixed_values = get_clean_pose_targets(batch, pose_dim, device) if fixed_mask is not None else None

    for _ in range(unroll_steps):
        delta = model(x, batch.edge_index, batch.edge_attr)
        delta = torch.tanh(delta)
        x = x + step_size * delta
        x = apply_fixed_poses(x, fixed_mask, fixed_values)
        x = maybe_project_to_tray(x, batch)

    return x


@torch.no_grad()
def evaluate(model, loader, constraint_types, device, solve_steps=5, max_batches=10):
    model.eval()
    losses = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        pose_dim = get_pose_dim(batch)
        x_init = initialize_poses(batch, pose_dim, device=device)
        x_final = solve(model, x_init, batch, steps=solve_steps, step_size=0.1)
        x_final = maybe_project_to_tray(x_final, batch)
        loss = compute_constraint_violation(x_final, batch, constraint_types, device=device)
        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses)) if losses else float('nan')


def build_dataloaders(train_task, eval_task, batch_size, num_workers):
    dataset_kwargs = dict(input_mode='qualitative', pre_transform=pre_transform, visualize=False)
    train_dataset = GraphDataset(train_task, **dataset_kwargs)
    eval_dataset = GraphDataset(eval_task, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    return train_loader, eval_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_task', type=str, default=None)
    parser.add_argument('--eval_task', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--unroll_steps', type=int, default=3)
    parser.add_argument('--solve_steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noise_scale', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='logs/learned_optimizer_v1')
    args = parser.parse_args()

    set_seed(args.seed)
    _, test_tasks, dims, constraint_types = get_data_config('qualitative')
    train_task = args.train_task or get_data_config('qualitative')[0]
    eval_task = args.eval_task or test_tasks[min(test_tasks.keys())]

    train_loader, eval_loader = build_dataloaders(
        train_task=train_task,
        eval_task=eval_task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pose_dim = dims[-1][0]
    model = LearnedOptimizer(
        pose_dim=pose_dim,
        num_edge_types=len(constraint_types),
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_iter = cycle(train_loader)
    best_eval = float('inf')

    for step in range(1, args.train_steps + 1):
        batch = next(train_iter).to(args.device)
        optimizer.zero_grad(set_to_none=True)

        pose_dim = get_pose_dim(batch)
        x0 = initialize_poses(batch, pose_dim, device=args.device, noise_scale=args.noise_scale)
        before = compute_constraint_violation(x0, batch, constraint_types, device=args.device).detach()

        x_final = rollout(
            model,
            batch,
            unroll_steps=args.unroll_steps,
            noise_scale=args.noise_scale,
            device=args.device,
            x_init=x0,
        )
        loss = compute_constraint_violation(x_final, batch, constraint_types, device=args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        after = loss.detach()
        improvement = before - after

        if step % args.log_every == 0 or step == 1:
            print(
                f'step={step:06d} '
                f'train_violation={loss.item():.6f} '
                f'improvement={improvement.item():.6f}'
            )

        if step % args.eval_every == 0 or step == args.train_steps:
            eval_loss = evaluate(
                model,
                eval_loader,
                constraint_types=constraint_types,
                device=args.device,
                solve_steps=args.solve_steps,
            )
            print(f'step={step:06d} eval_violation={eval_loss:.6f}')

            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'constraint_types': constraint_types,
            }
            torch.save(checkpoint, save_dir / 'last.pt')

            if eval_loss < best_eval:
                best_eval = eval_loss
                torch.save(checkpoint, save_dir / 'best.pt')

        if step % args.save_every == 0:
            torch.save(
                {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                },
                save_dir / f'step_{step:06d}.pt',
            )

    with open(save_dir / 'train_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f'saved checkpoints to {save_dir}')


if __name__ == '__main__':
    main()
