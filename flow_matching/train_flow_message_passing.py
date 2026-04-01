"""
Train a flow model with multi-round constraint interaction.

Direction A:
  replace one-shot independent edge composition with K rounds of message
  passing so node states can negotiate conflicting constraints in one forward.

Usage:
    python train_flow_message_passing.py -input_mode qualitative
    python train_flow_message_passing.py -input_mode qualitative -n_rounds 4
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent

from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from flow_matching.train_flow import FlowTrainer, get_best_device, get_data_config
from flow_matching.flow_message_passing import MessagePassingFlowMatchingCCSP


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
        f'or generate the qualitative data before training.'
    )


def main():
    parser = argparse.ArgumentParser(description='Train message-passing flow model for CCSP')
    parser.add_argument('-input_mode', type=str, default='qualitative',
                        choices=['qualitative', 'diffuse_pairwise', 'stability_flat', 'robot_box'])
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-n_rounds', type=int, default=3)
    parser.add_argument('-lr', type=float, default=5e-4)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-train_num_steps', type=int, default=200000)
    parser.add_argument('-save_every', type=int, default=10000)
    parser.add_argument('-results_dir', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-print_every', type=int, default=250)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-eval_n_samples', type=int, default=1)
    parser.add_argument('-eval_n_steps', type=int, default=20)
    args = parser.parse_args()

    device = get_best_device(args.device)
    print(f'  Device : {device}')

    train_task, test_tasks, dims, constraint_types = get_data_config(args.input_mode)
    results_dir = args.results_dir or (
        f'./logs/flow_mp_{args.input_mode}_h{args.hidden_dim}_r{args.n_rounds}'
    )
    print(f'  Config : input_mode={args.input_mode} hidden_dim={args.hidden_dim} '
          f'n_rounds={args.n_rounds} batch_size={args.batch_size} lr={args.lr} '
          f'steps={args.train_num_steps:,}')
    print(f'  Runtime: num_workers={args.num_workers} print_every={args.print_every} '
          f'eval_n_samples={args.eval_n_samples} eval_n_steps={args.eval_n_steps} '
          f'verbose={args.verbose}')
    print(f'  Train task: {train_task}')
    print(f'  Test tasks : {test_tasks}')
    print(f'  Dims      : {dims}')
    print(f'  Constraints ({len(constraint_types)}): {constraint_types}')
    print(f'  Results   : {results_dir}')

    validate_dataset_dir(train_task)
    for _, task_name in test_tasks.items():
        task_dir = ROOT / 'data' / task_name
        if task_dir.exists():
            validate_dataset_dir(task_name)

    ds_kw = dict(input_mode=args.input_mode, pre_transform=pre_transform, visualize=False)
    train_dataset = GraphDataset(train_task, **ds_kw)
    test_datasets = {
        k: GraphDataset(t, **ds_kw)
        for k, t in test_tasks.items()
        if os.path.isdir(f'./data/{t}')
    }
    print(f'  Train: {len(train_dataset):,}   Tests: {list(test_datasets.keys())}')

    model = MessagePassingFlowMatchingCCSP(
        dims=dims,
        hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True,
        device=device,
        n_rounds=args.n_rounds,
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {n_p:,}')

    batch = next(iter(DataLoader(train_dataset, batch_size=4)))
    loss = model.compute_loss(batch)
    print(f'  Sanity loss: {loss.item():.4f}')

    trainer = FlowTrainer(
        model,
        train_dataset,
        test_datasets,
        lr=args.lr,
        batch_size=args.batch_size,
        train_num_steps=args.train_num_steps,
        save_every=args.save_every,
        results_folder=results_dir,
        num_workers=args.num_workers,
        print_every=args.print_every,
        verbose=args.verbose,
        eval_n_samples=args.eval_n_samples,
        eval_n_steps=args.eval_n_steps,
    )

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            trainer.step = ckpt.get('step', 0)
            print(f'  [load] step={trainer.step} from {resume_path}')
        else:
            trainer.load(args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
