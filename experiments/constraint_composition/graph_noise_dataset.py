from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.constraint_composition.core import SceneSpec


def build_graph_noise_dataset(
    scenes: Iterable[SceneSpec],
    num_trajectories: int = 1000,
    rollout_steps: int = 40,
    seed: int = 0,
) -> list[Data]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_graph_noise_dataset() requires at least one scene.')

    rng = np.random.default_rng(seed)
    dataset: list[Data] = []

    for traj_idx in range(max(num_trajectories, 0)):
        scene = scenes[traj_idx % len(scenes)]
        x_star = scene.target_poses.astype(np.float32, copy=False)

        for step_idx in range(max(rollout_steps, 0)):
            tau = float(step_idx) / float(max(rollout_steps, 1))
            sigma = 0.75 * (1.0 - tau)
            noise = rng.normal(size=x_star.shape).astype(np.float32) * sigma
            noise[scene.mask] = 0.0
            noisy_poses = scene.clamp(x_star + noise).astype(np.float32, copy=False)

            target_v = (x_star[:, :2] - noisy_poses[:, :2]).astype(np.float32, copy=False)
            target_v[scene.mask] = 0.0

            tau_column = np.full((scene.num_nodes, 1), tau, dtype=np.float32)
            mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
            node_features = np.concatenate([
                scene.geoms.astype(np.float32, copy=False),
                noisy_poses.astype(np.float32, copy=False),
                mask_column,
                tau_column,
            ], axis=1)

            dataset.append(
                Data(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    edge_index=torch.tensor(scene.edge_index.T, dtype=torch.long),
                    edge_attr=torch.tensor(scene.edge_attr, dtype=torch.long),
                    target_v=torch.tensor(target_v, dtype=torch.float32),
                    mask=torch.tensor(scene.mask, dtype=torch.bool),
                )
            )

    return dataset
