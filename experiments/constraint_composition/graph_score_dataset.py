from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.constraint_composition.core import SceneSpec, total_violation
from experiments.constraint_composition.prototypes import numerical_grad


def build_graph_score_dataset(
    scenes: Iterable[SceneSpec],
    num_samples: int = 30000,
    noise_scale: float = 0.2,
    fd_eps: float = 1e-3,
    seed: int = 0,
) -> list[Data]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_graph_score_dataset() requires at least one scene.')

    rng = np.random.default_rng(seed)
    dataset: list[Data] = []

    for sample_idx in range(max(num_samples, 0)):
        scene = scenes[sample_idx % len(scenes)]
        poses = scene.initialize_state(rng).astype(np.float32, copy=False)

        noise = rng.standard_normal(size=poses[:, :2].shape).astype(np.float32) * float(noise_scale)
        noise[scene.mask] = 0.0
        perturbed = poses.copy()
        perturbed[:, :2] += noise
        perturbed = scene.clamp(perturbed).astype(np.float32, copy=False)

        grad = numerical_grad(
            perturbed,
            lambda x: total_violation(scene, scene.clamp(x)),
            eps=fd_eps,
        )
        target = (-grad[:, :2]).astype(np.float32, copy=False)
        target[scene.mask] = 0.0

        mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
        node_features = np.concatenate([
            scene.geoms.astype(np.float32, copy=False),
            perturbed,
            mask_column,
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

    return dataset
