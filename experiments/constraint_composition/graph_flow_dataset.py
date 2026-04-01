from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.constraint_composition.core import SceneSpec


def build_graph_flow_dataset(
    scenes: Iterable[SceneSpec],
    num_trajectories: int = 1000,
    rollout_steps: int = 40,
    step_size: float = 0.1,
    fd_eps: float = 1e-3,
    seed: int = 0,
) -> list[Data]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_graph_flow_dataset() requires at least one scene.')

    rng = np.random.default_rng(seed)
    dataset: list[Data] = []

    for traj_idx in range(max(num_trajectories, 0)):
        scene = scenes[traj_idx % len(scenes)]
        x0 = scene.initialize_state(rng).astype(np.float32, copy=False)
        x1 = scene.target_poses.astype(np.float32, copy=False)
        v_target = (x1[:, :2] - x0[:, :2]).astype(np.float32, copy=False)
        v_target[scene.mask] = 0.0

        for step_idx in range(max(rollout_steps, 0)):
            tau = float(step_idx) / float(max(rollout_steps, 1))
            poses = ((1.0 - tau) * x0 + tau * x1).astype(np.float32, copy=False)

            tau_column = np.full((scene.num_nodes, 1), tau, dtype=np.float32)
            mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
            node_features = np.concatenate([
                scene.geoms.astype(np.float32, copy=False),
                poses,
                mask_column,
                tau_column,
            ], axis=1)

            dataset.append(
                Data(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    edge_index=torch.tensor(scene.edge_index.T, dtype=torch.long),
                    edge_attr=torch.tensor(scene.edge_attr, dtype=torch.long),
                    target_v=torch.tensor(v_target, dtype=torch.float32),
                    mask=torch.tensor(scene.mask, dtype=torch.bool),
                )
            )

    return dataset
