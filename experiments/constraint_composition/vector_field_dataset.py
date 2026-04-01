from __future__ import annotations

from typing import Iterable

import numpy as np

from experiments.constraint_composition.core import SceneSpec, total_violation
from experiments.constraint_composition.global_features import extract_global_features
from experiments.constraint_composition.prototypes import numerical_grad


def build_vector_field_dataset(
    scenes: Iterable[SceneSpec],
    num_samples: int = 5000,
    step_size: float = 0.1,
    fd_eps: float = 1e-3,
    seed: int = 0,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_vector_field_dataset() requires at least one scene.')

    capacity = int(max_nodes) if max_nodes is not None else max(scene.num_nodes for scene in scenes)
    rng = np.random.default_rng(seed)
    x_rows = []
    v_rows = []

    for sample_idx in range(max(num_samples, 0)):
        scene = scenes[sample_idx % len(scenes)]
        poses = scene.initialize_state(rng)

        grad = numerical_grad(
            poses,
            lambda x: total_violation(scene, scene.clamp(x)),
            eps=fd_eps,
        )
        v_target = -step_size * grad[:, :2]
        v_target[scene.mask] = 0.0

        phi = extract_global_features(poses, scene, max_nodes=capacity)
        v_padded = np.zeros((capacity, 2), dtype=np.float32)
        v_padded[:scene.num_nodes] = v_target.astype(np.float32, copy=False)

        x_rows.append(phi.astype(np.float32, copy=False))
        v_rows.append(v_padded.reshape(-1))

    x_arr = np.stack(x_rows, axis=0).astype(np.float32)
    v_arr = np.stack(v_rows, axis=0).astype(np.float32)
    return x_arr, v_arr
