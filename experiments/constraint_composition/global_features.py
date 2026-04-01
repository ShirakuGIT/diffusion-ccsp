from __future__ import annotations

from typing import Optional

import numpy as np

from experiments.constraint_composition.core import SceneSpec


def extract_global_features(
    poses: np.ndarray,
    scene: SceneSpec,
    max_nodes: Optional[int] = None,
) -> np.ndarray:
    num_nodes = int(scene.num_nodes)
    capacity = int(max_nodes) if max_nodes is not None else num_nodes
    if capacity < num_nodes:
        raise ValueError(f'max_nodes={capacity} is smaller than scene.num_nodes={num_nodes}.')

    positions = np.zeros((capacity, 2), dtype=np.float32)
    positions[:num_nodes] = poses[:num_nodes, :2].astype(np.float32, copy=False)

    rel_vectors = []
    distances = []
    for i in range(capacity):
        for j in range(i + 1, capacity):
            if i < num_nodes and j < num_nodes:
                rel = positions[j] - positions[i]
                dist = float(np.linalg.norm(rel))
            else:
                rel = np.zeros(2, dtype=np.float32)
                dist = 0.0
            rel_vectors.extend([float(rel[0]), float(rel[1])])
            distances.append(dist)

    return np.concatenate([
        positions.reshape(-1),
        np.asarray(rel_vectors, dtype=np.float32),
        np.asarray(distances, dtype=np.float32),
    ]).astype(np.float32, copy=False)
