from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from experiments.constraint_composition.core import SceneSpec, evaluate_constraints


UNARY_CONSTRAINT_TYPES = {
    'in', 'center-in', 'left-in', 'right-in', 'top-in', 'bottom-in',
}


def extract_invariant_features(poses: np.ndarray, constraint) -> np.ndarray:
    i, j = constraint.nodes
    pose_i_xy = poses[i, :2]
    if constraint.constraint_type in UNARY_CONSTRAINT_TYPES:
        return pose_i_xy.astype(np.float32, copy=True)

    pose_j_xy = poses[j, :2]
    rel = (pose_j_xy - pose_i_xy).astype(np.float32, copy=False)
    ctype = constraint.constraint_type

    if ctype in {'left-of', 'right-of', 'top-of', 'bottom-of'}:
        return rel.copy()
    if ctype in {'close-to', 'away-from'}:
        return np.asarray([np.linalg.norm(rel)], dtype=np.float32)
    if ctype == 'h-aligned':
        return np.asarray([rel[1]], dtype=np.float32)
    if ctype == 'v-aligned':
        return np.asarray([rel[0]], dtype=np.float32)
    return rel.copy()


def _select_diverse_prototypes(candidates: List[np.ndarray], k: int, threshold: float) -> List[np.ndarray]:
    if not candidates or k <= 0:
        return []

    remaining = [np.asarray(candidate, dtype=np.float32).copy() for candidate in candidates]
    selected = [remaining.pop(0)]

    while remaining and len(selected) < k:
        min_dists = np.asarray([
            min(float(np.linalg.norm(candidate - chosen)) for chosen in selected)
            for candidate in remaining
        ], dtype=np.float32)
        best_idx = int(np.argmax(min_dists))
        best_dist = float(min_dists[best_idx])
        if best_dist < threshold and len(selected) > 0:
            break
        selected.append(remaining.pop(best_idx))

    if len(selected) >= k or not remaining:
        return selected[:k]

    while remaining and len(selected) < k:
        min_dists = np.asarray([
            min(float(np.linalg.norm(candidate - chosen)) for chosen in selected)
            for candidate in remaining
        ], dtype=np.float32)
        best_idx = int(np.argmax(min_dists))
        selected.append(remaining.pop(best_idx))

    return selected[:k]


def build_prototypes(
    scenes: Iterable[SceneSpec],
    num_samples: int = 2000,
    k_values: Iterable[int] = (10,),
    diversity_threshold: float = 0.1,
    seed: int = 0,
) -> Tuple[Dict[int, Dict[str, List[np.ndarray]]], Dict[str, object]]:
    scenes = list(scenes)
    k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not scenes:
        raise ValueError('build_prototypes() requires at least one scene.')
    if not k_values:
        raise ValueError('build_prototypes() requires at least one positive K value.')

    rng = np.random.default_rng(seed)
    candidates: Dict[str, List[np.ndarray]] = defaultdict(list)

    for sample_idx in range(max(num_samples, 0)):
        scene = scenes[sample_idx % len(scenes)]
        poses = scene.initialize_state(rng)
        for record in evaluate_constraints(scene, poses):
            if record.h_value > 0.0:
                candidates[record.constraint_type].append(extract_invariant_features(poses, record))

    prototypes_by_k: Dict[int, Dict[str, List[np.ndarray]]] = {}
    stats: Dict[str, object] = {
        'num_samples': int(num_samples),
        'diversity_threshold': float(diversity_threshold),
        'candidate_counts': {ctype: len(values) for ctype, values in sorted(candidates.items())},
        'available_constraint_types': sorted(candidates),
    }

    for k in k_values:
        prototypes: Dict[str, List[np.ndarray]] = {}
        for ctype, values in candidates.items():
            selected = _select_diverse_prototypes(values, k=k, threshold=diversity_threshold)
            if selected:
                prototypes[ctype] = selected
        prototypes_by_k[k] = prototypes

    stats['prototype_counts'] = {
        str(k): {ctype: len(values) for ctype, values in sorted(prototypes.items())}
        for k, prototypes in prototypes_by_k.items()
    }
    return prototypes_by_k, stats


def prototype_energy(scene: SceneSpec,
                     poses: np.ndarray,
                     prototypes: Dict[str, List[np.ndarray]],
                     tau: float = 0.1) -> float:
    total_energy = 0.0
    for record in evaluate_constraints(scene, poses):
        proto_list = prototypes.get(record.constraint_type)
        if not proto_list:
            continue
        local_state = extract_invariant_features(poses, record)
        dists = np.asarray(
            [float(np.sum((local_state - proto) ** 2)) for proto in proto_list],
            dtype=np.float32,
        )
        d_min = float(np.min(dists))
        weights = np.exp(-(dists - d_min) / max(tau, 1e-8))
        z = float(np.sum(weights)) + 1e-8
        soft_energy = d_min - tau * np.log(z)
        total_energy += float(soft_energy)
    return float(total_energy)


def numerical_grad(x: np.ndarray, energy_fn, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float32)
    for flat_idx in range(x.size):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos.flat[flat_idx] += eps
        x_neg.flat[flat_idx] -= eps
        grad.flat[flat_idx] = (energy_fn(x_pos) - energy_fn(x_neg)) / (2.0 * eps)
    return grad
