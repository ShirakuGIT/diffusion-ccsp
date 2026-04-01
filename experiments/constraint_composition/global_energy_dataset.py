from __future__ import annotations

from typing import Iterable

import numpy as np

from experiments.constraint_composition.core import SceneSpec, total_violation
from experiments.constraint_composition.global_features import extract_global_features


def build_global_dataset(
    scenes: Iterable[SceneSpec],
    num_samples: int = 5000,
    seed: int = 0,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_global_dataset() requires at least one scene.')

    capacity = int(max_nodes) if max_nodes is not None else max(scene.num_nodes for scene in scenes)
    rng = np.random.default_rng(seed)
    x_rows = []
    y_rows = []

    for sample_idx in range(max(num_samples, 0)):
        scene = scenes[sample_idx % len(scenes)]
        poses = scene.initialize_state(rng)
        phi = extract_global_features(poses, scene, max_nodes=capacity)
        y = total_violation(scene, poses)
        x_rows.append(phi.astype(np.float32, copy=False))
        y_rows.append(float(y))

    x_arr = np.stack(x_rows, axis=0).astype(np.float32)
    y_arr = np.asarray(y_rows, dtype=np.float32)
    return x_arr, y_arr
