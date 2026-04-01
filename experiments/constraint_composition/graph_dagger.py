from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.constraint_composition.core import SceneSpec
from experiments.constraint_composition.graph_noise_dataset import build_graph_noise_dataset
from experiments.constraint_composition.graph_noise_model import (
    LearnedGraphVectorField,
    train_graph_vector_field,
)


def _make_graph_sample(scene: SceneSpec, poses: np.ndarray, tau: float, target_v: np.ndarray) -> Data:
    tau_column = np.full((scene.num_nodes, 1), tau, dtype=np.float32)
    mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
    node_features = np.concatenate([
        scene.geoms.astype(np.float32, copy=False),
        poses.astype(np.float32, copy=False),
        mask_column,
        tau_column,
    ], axis=1)
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(scene.edge_index.T, dtype=torch.long),
        edge_attr=torch.tensor(scene.edge_attr, dtype=torch.long),
        target_v=torch.tensor(target_v.astype(np.float32, copy=False), dtype=torch.float32),
        mask=torch.tensor(scene.mask, dtype=torch.bool),
    )


def _predict_update(scene: SceneSpec, poses: np.ndarray, bundle: LearnedGraphVectorField, tau: float) -> np.ndarray:
    node_features = np.concatenate([
        scene.geoms.astype(np.float32, copy=False),
        poses.astype(np.float32, copy=False),
        scene.mask.astype(np.float32).reshape(-1, 1),
        np.full((scene.num_nodes, 1), tau, dtype=np.float32),
    ], axis=1)
    x_norm = (node_features - bundle.node_mean) / bundle.node_std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    edge_index = torch.tensor(scene.edge_index.T, dtype=torch.long)
    edge_attr = torch.tensor(scene.edge_attr, dtype=torch.long)
    with torch.no_grad():
        pred = bundle.model(x_tensor, edge_index, edge_attr).cpu().numpy().astype(np.float32, copy=False)
    pred[scene.mask] = 0.0
    return pred


def train_graph_dagger(
    scenes: Iterable[SceneSpec],
    base_num_trajectories: int = 1000,
    rollout_steps: int = 40,
    rounds: int = 2,
    rollout_trajectories: int = 250,
    seed: int = 0,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> tuple[LearnedGraphVectorField, dict[str, object]]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('train_graph_dagger() requires at least one scene.')

    rng = np.random.default_rng(seed)
    dataset = build_graph_noise_dataset(
        scenes,
        num_trajectories=base_num_trajectories,
        rollout_steps=rollout_steps,
        seed=seed,
    )
    history = []
    bundle = None

    for round_idx in range(max(rounds, 1)):
        bundle, train_stats = train_graph_vector_field(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + round_idx,
            device=device,
        )
        history.append({
            'round': int(round_idx),
            'dataset_size': int(len(dataset)),
            'train_stats': train_stats,
        })

        if round_idx == max(rounds, 1) - 1:
            break

        new_samples = []
        for traj_idx in range(max(rollout_trajectories, 0)):
            scene = scenes[traj_idx % len(scenes)]
            poses = scene.initialize_state(rng).astype(np.float32, copy=False)
            x_star = scene.target_poses.astype(np.float32, copy=False)

            for step_idx in range(max(rollout_steps, 0)):
                tau = float(step_idx) / float(max(rollout_steps, 1))
                target_v = (x_star[:, :2] - poses[:, :2]).astype(np.float32, copy=False)
                target_v[scene.mask] = 0.0
                new_samples.append(_make_graph_sample(scene, poses, tau, target_v))

                pred_v = _predict_update(scene, poses, bundle, tau)
                update = np.zeros_like(poses, dtype=np.float32)
                update[:, :2] = pred_v
                poses = scene.clamp(poses + update).astype(np.float32, copy=False)

        dataset.extend(new_samples)

    return bundle, {
        'rounds': int(max(rounds, 1)),
        'base_num_trajectories': int(base_num_trajectories),
        'rollout_trajectories': int(rollout_trajectories),
        'rollout_steps': int(rollout_steps),
        'history': history,
        'final_dataset_size': int(len(dataset)),
    }
