from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np

from experiments.constraint_composition.core import SceneSpec, evaluate_constraints
from experiments.constraint_composition.prototypes import extract_invariant_features


def build_constraint_dataset(
    scenes: Iterable[SceneSpec],
    num_samples: int = 5000,
    seed: int = 0,
) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    scenes = list(scenes)
    if not scenes:
        raise ValueError('build_constraint_dataset() requires at least one scene.')

    rng = np.random.default_rng(seed)
    features = defaultdict(list)
    labels = defaultdict(list)

    for sample_idx in range(max(num_samples, 0)):
        scene = scenes[sample_idx % len(scenes)]
        poses = scene.initialize_state(rng)
        for record in evaluate_constraints(scene, poses):
            z = extract_invariant_features(poses, record)
            y = max(0.0, -record.h_value)
            features[record.constraint_type].append(np.asarray(z, dtype=np.float32))
            labels[record.constraint_type].append(float(y))

    datasets: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for constraint_type in sorted(features):
        x_arr = np.stack(features[constraint_type], axis=0).astype(np.float32)
        y_arr = np.asarray(labels[constraint_type], dtype=np.float32)
        datasets[constraint_type] = (x_arr, y_arr)
    return datasets
