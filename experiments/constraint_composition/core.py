from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from flow_matching.fix_and_eval import clamp_to_tray, compute_barrier


QUALITATIVE_CONSTRAINT_TYPES = [
    'in', 'center-in', 'left-in', 'right-in', 'top-in', 'bottom-in',
    'cfree', 'left-of', 'top-of',
    'close-to', 'away-from', 'h-aligned', 'v-aligned',
]

QUALITATIVE_DIMS = ((2, 0, 2), (4, 2, 6))


@dataclass
class ConstraintRecord:
    constraint_index: int
    type_index: int
    constraint_type: str
    nodes: tuple[int, int]
    h_value: float
    grad: np.ndarray
    violation: float


@dataclass
class SceneSpec:
    scene_id: int
    geoms: np.ndarray
    target_poses: np.ndarray
    mask: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    constraint_types: List[str]

    @property
    def pose_dim(self) -> int:
        return self.target_poses.shape[1]

    @property
    def num_nodes(self) -> int:
        return self.target_poses.shape[0]

    def initialize_state(self, rng: np.random.Generator) -> np.ndarray:
        poses = rng.normal(size=self.target_poses.shape).astype(np.float32)
        poses[self.mask] = self.target_poses[self.mask]
        return self.clamp(poses)

    def clamp(self, poses: np.ndarray) -> np.ndarray:
        poses_t = torch.tensor(poses, dtype=torch.float32)
        geoms_t = torch.tensor(self.geoms, dtype=torch.float32)
        mask_t = torch.tensor(self.mask, dtype=torch.bool)
        clamped = clamp_to_tray(poses_t, geoms_t, mask_t, pose_dim=self.pose_dim)
        return clamped.cpu().numpy()


def scene_from_data(data, scene_id: int, constraint_types: Optional[List[str]] = None) -> SceneSpec:
    if constraint_types is None:
        constraint_types = QUALITATIVE_CONSTRAINT_TYPES
    pose_begin, pose_end = QUALITATIVE_DIMS[-1][1], QUALITATIVE_DIMS[-1][2]
    geom_end = QUALITATIVE_DIMS[0][2]
    geoms = data.x[:, :geom_end].cpu().numpy().astype(np.float32)
    target_poses = data.x[:, pose_begin:pose_end].cpu().numpy().astype(np.float32)
    mask = data.mask.bool().cpu().numpy() if hasattr(data, 'mask') else np.zeros(data.x.shape[0], dtype=bool)
    edge_index = data.edge_index.T.cpu().numpy().astype(np.int64)
    edge_attr = data.edge_attr.cpu().numpy().astype(np.int64)
    return SceneSpec(
        scene_id=scene_id,
        geoms=geoms,
        target_poses=target_poses,
        mask=mask,
        edge_index=edge_index,
        edge_attr=edge_attr,
        constraint_types=constraint_types,
    )


def evaluate_constraints(scene: SceneSpec, poses: np.ndarray) -> List[ConstraintRecord]:
    records: List[ConstraintRecord] = []
    pose_dim = poses.shape[1]
    for ei in range(scene.edge_index.shape[0]):
        i, j = int(scene.edge_index[ei, 0]), int(scene.edge_index[ei, 1])
        cidx = int(scene.edge_attr[ei])
        if cidx >= len(scene.constraint_types):
            continue
        ctype = scene.constraint_types[cidx]
        h_val, grad_i, grad_j = compute_barrier(
            ctype,
            torch.tensor(poses[i], dtype=torch.float32),
            torch.tensor(poses[j], dtype=torch.float32),
            torch.tensor(scene.geoms[i], dtype=torch.float32),
            torch.tensor(scene.geoms[j], dtype=torch.float32),
        )
        full_grad = np.zeros((scene.num_nodes, pose_dim), dtype=np.float32)
        full_grad[i, :pose_dim] += grad_i[:pose_dim]
        full_grad[j, :pose_dim] += grad_j[:pose_dim]
        records.append(
            ConstraintRecord(
                constraint_index=ei,
                type_index=cidx,
                constraint_type=ctype,
                nodes=(i, j),
                h_value=float(h_val),
                grad=full_grad,
                violation=float(max(0.0, -h_val)),
            )
        )
    return records


def total_violation(scene: SceneSpec, poses: np.ndarray) -> float:
    return float(sum(r.violation for r in evaluate_constraints(scene, poses)))


def total_violation_gradient(scene: SceneSpec, poses: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(poses, dtype=np.float32)
    for record in evaluate_constraints(scene, poses):
        if record.violation > 0:
            grad -= record.grad
    grad[scene.mask] = 0.0
    return grad


def violated_constraint_records(scene: SceneSpec, poses: np.ndarray) -> List[ConstraintRecord]:
    return [r for r in evaluate_constraints(scene, poses) if r.violation > 0]


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < eps:
        return 1.0 if np.linalg.norm(a_flat) < eps and np.linalg.norm(b_flat) < eps else 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def scene_summary(scene: SceneSpec, poses: np.ndarray) -> Dict[str, float]:
    records = evaluate_constraints(scene, poses)
    violations = [r.violation for r in records]
    return {
        'total_violation': float(sum(violations)),
        'num_constraints': len(records),
        'num_violated': int(sum(v > 0.0 for v in violations)),
        'min_barrier': float(min((r.h_value for r in records), default=0.0)),
    }
