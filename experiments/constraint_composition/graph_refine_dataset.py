from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.constraint_composition.core import SceneSpec, total_violation
from experiments.constraint_composition.graph_score_proj_dataset import selective_project_state_linear
from experiments.constraint_composition.methods import graph_score_plus_step
from experiments.constraint_composition.prototypes import numerical_grad
from experiments.constraint_composition.graph_noise_model import LearnedGraphVectorField


def build_graph_refine_dataset(
    scenes: Iterable[SceneSpec],
    coarse_bundle: LearnedGraphVectorField,
    num_trajectories: int = 1000,
    rollout_steps: int = 40,
    switch_threshold: float = 1.0,
    noise_scale: float = 0.05,
    step_size: float = 0.1,
    sigma: float = 0.1,
    fd_eps: float = 1e-3,
    projection_passes: int = 1,
    seed: int = 0,
) -> list[Data]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_graph_refine_dataset() requires at least one scene.')

    rng = np.random.default_rng(seed)
    dataset: list[Data] = []

    for traj_idx in range(max(num_trajectories, 0)):
        scene = scenes[traj_idx % len(scenes)]
        poses = scene.initialize_state(rng).astype(np.float32, copy=False)

        for _ in range(max(rollout_steps, 0)):
            current_violation = total_violation(scene, poses)
            if current_violation < switch_threshold:
                sample_poses = poses.copy()
                if noise_scale > 0.0:
                    noise = rng.standard_normal(size=sample_poses[:, :2].shape).astype(np.float32) * float(noise_scale)
                    noise[scene.mask] = 0.0
                    sample_poses[:, :2] += noise
                    sample_poses = scene.clamp(sample_poses).astype(np.float32, copy=False)

                grad = numerical_grad(
                    sample_poses,
                    lambda x: total_violation(scene, scene.clamp(x)),
                    eps=fd_eps,
                )
                target = (-grad[:, :2]).astype(np.float32, copy=False)
                target[scene.mask] = 0.0

                mask_col = scene.mask.astype(np.float32).reshape(-1, 1)
                node_features = np.concatenate([
                    scene.geoms.astype(np.float32, copy=False),
                    sample_poses,
                    mask_col,
                ], axis=1)

                dataset.append(
                    Data(
                        x=torch.tensor(node_features, dtype=torch.float32),
                        edge_index=torch.tensor(scene.edge_index.T, dtype=torch.long),
                        edge_attr=torch.tensor(scene.edge_attr, dtype=torch.long),
                        target_v=torch.tensor(target, dtype=torch.float32),
                        mask=torch.tensor(scene.mask, dtype=torch.bool),
                    )
                )

            coarse_update = graph_score_plus_step(
                scene,
                poses,
                coarse_bundle,
                step_size=step_size,
                sigma=sigma,
                fd_eps=fd_eps,
            )
            proposed = scene.clamp(poses + coarse_update).astype(np.float32, copy=False)
            poses = selective_project_state_linear(
                scene,
                proposed,
                passes=projection_passes,
            ).astype(np.float32, copy=False)

    return dataset
