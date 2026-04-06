from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch

from experiments.constraint_composition.core import (
    SceneSpec,
    evaluate_constraints,
    total_violation,
    total_violation_gradient,
    violated_constraint_records,
)
from experiments.constraint_composition.global_energy_model import LearnedGlobalEnergy
from experiments.constraint_composition.graph_noise_model import LearnedGraphVectorField
from experiments.constraint_composition.learned_energy_model import LearnedConstraintEnergy
from experiments.constraint_composition.prototypes import numerical_grad, prototype_energy
from experiments.constraint_composition.vector_field_model import LearnedVectorField
from experiments.constraint_composition.vector_field_time_model import LearnedTimeVectorField


@dataclass
class Method:
    name: str
    step_fn: Callable[..., np.ndarray]


def _normalize(update: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(update.reshape(-1))
    if norm < eps:
        return np.zeros_like(update)
    return update / norm


def make_energy_descent(step_size: float, normalized: bool) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        grad = total_violation_gradient(scene, poses)
        update = -_normalize(grad) * step_size if normalized else -grad * step_size
        update[scene.mask] = 0.0
        return update

    name = 'energy_normalized' if normalized else 'energy'
    return Method(name=name, step_fn=step)


def make_consensus(step_size: float, weight_mode: str) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        records = violated_constraint_records(scene, poses)
        if not records:
            return np.zeros_like(poses)

        proposals = []
        weights = []
        for record in records:
            delta = -record.grad
            delta[scene.mask] = 0.0
            proposal = poses + step_size * delta
            proposals.append(proposal)
            if weight_mode == 'uniform':
                weights.append(1.0)
            elif weight_mode == 'violation':
                weights.append(max(record.violation, 1e-6))
            else:
                raise ValueError(f'Unknown consensus weight mode: {weight_mode}')

        weights_arr = np.asarray(weights, dtype=np.float32)
        weights_arr /= weights_arr.sum()
        proposal_stack = np.stack(proposals, axis=0)
        consensus = np.tensordot(weights_arr, proposal_stack, axes=(0, 0))
        update = consensus - poses
        update[scene.mask] = 0.0
        return update

    return Method(name=f'consensus_{weight_mode}', step_fn=step)


def make_mixture(step_size: float, weight_mode: str, alpha: float = 8.0, topk: int = 2) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        records = violated_constraint_records(scene, poses)
        if not records:
            return np.zeros_like(poses)

        severities = np.asarray([r.violation for r in records], dtype=np.float32)
        deltas = np.stack([-r.grad for r in records], axis=0)
        deltas[:, scene.mask, :] = 0.0

        if weight_mode == 'linear':
            weights = severities / max(severities.sum(), 1e-8)
        elif weight_mode == 'softmax':
            logits = alpha * severities
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= max(weights.sum(), 1e-8)
        elif weight_mode == 'topk':
            keep = np.argsort(-severities)[:min(topk, len(severities))]
            weights = np.zeros_like(severities)
            weights[keep] = 1.0 / max(len(keep), 1)
        else:
            raise ValueError(f'Unknown mixture weight mode: {weight_mode}')

        update = step_size * np.tensordot(weights, deltas, axes=(0, 0))
        update[scene.mask] = 0.0
        return update

    return Method(name=f'mixture_{weight_mode}', step_fn=step)


def make_energy_langevin(step_size: float, noise_scale: float, anneal: bool = False) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, t: int = 0, T: int = 1,
             rng: np.random.Generator | None = None, **_: object) -> np.ndarray:
        grad = total_violation_gradient(scene, poses)
        grad[scene.mask] = 0.0

        generator = rng if rng is not None else np.random.default_rng()
        noise_std = noise_scale * (1.0 - float(t) / max(float(T), 1.0)) if anneal else noise_scale
        noise = generator.normal(size=poses.shape).astype(np.float32) * noise_std
        noise[scene.mask] = 0.0

        update = -step_size * grad + noise
        update[scene.mask] = 0.0
        return update

    suffix = 'annealed' if anneal else 'constant'
    return Method(name=f'energy_langevin_sigma{noise_scale:g}_{suffix}', step_fn=step)


def make_projected_energy(step_size: float, alpha: float = 1.0, projection_passes: int = 3) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        grad = total_violation_gradient(scene, poses)
        grad[scene.mask] = 0.0
        v = -grad.reshape(-1).astype(np.float32)

        records = violated_constraint_records(scene, poses)
        if not records:
            return np.zeros_like(poses)

        for _ in range(max(projection_passes, 1)):
            updated = False
            for record in records:
                if record.violation <= 0:
                    continue
                a = record.grad.reshape(-1).astype(np.float32)
                a_norm = float(np.linalg.norm(a))
                if a_norm < 1e-8:
                    continue
                a = a / (a_norm + 1e-8)
                a_norm_sq = float(np.dot(a, a))
                b = alpha * record.violation
                lhs = float(np.dot(a, v))
                if lhs < b:
                    v = v + ((b - lhs) / (a_norm_sq + 1e-8)) * a
                    updated = True
            if not updated:
                break

        update = step_size * v.reshape(poses.shape)
        update[scene.mask] = 0.0
        return update

    return Method(name='projected_energy', step_fn=step)


def make_prototype_energy(step_size: float,
                          prototypes: Dict[str, List[np.ndarray]],
                          k: int,
                          tau: float = 0.1,
                          fd_eps: float = 1e-3) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        grad = numerical_grad(
            poses,
            lambda x: prototype_energy(scene, scene.clamp(x), prototypes, tau=tau),
            eps=fd_eps,
        )
        update = -step_size * grad
        update[scene.mask] = 0.0
        return update

    tau_label = f'{tau:g}'.replace('.', 'p')
    return Method(name=f'prototype_energy_k{k}_tau{tau_label}', step_fn=step)


def learned_energy(scene: SceneSpec,
                   poses: np.ndarray,
                   models: Dict[str, LearnedConstraintEnergy]) -> float:
    from experiments.constraint_composition.core import evaluate_constraints
    from experiments.constraint_composition.prototypes import extract_invariant_features

    total = 0.0
    for record in evaluate_constraints(scene, poses):
        bundle = models.get(record.constraint_type)
        if bundle is None:
            continue
        z = extract_invariant_features(poses, record).astype(np.float32)
        z_norm = (z - bundle.mean) / bundle.std
        z_tensor = torch.tensor(z_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            energy = torch.relu(bundle.model(z_tensor)).squeeze(0)
        total += float(energy.item())
    return float(total)


def make_learned_energy(step_size: float,
                        models: Dict[str, LearnedConstraintEnergy],
                        fd_eps: float = 1e-3) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        grad = numerical_grad(
            poses,
            lambda x: learned_energy(scene, scene.clamp(x), models),
            eps=fd_eps,
        )
        update = -step_size * grad
        update[scene.mask] = 0.0
        return update

    return Method(name='learned_energy', step_fn=step)


def global_energy(scene: SceneSpec, poses: np.ndarray, bundle: LearnedGlobalEnergy) -> float:
    from experiments.constraint_composition.global_features import extract_global_features

    phi = extract_global_features(poses, scene, max_nodes=bundle.max_nodes).astype(np.float32)
    phi_norm = (phi - bundle.mean) / bundle.std
    if np.isnan(phi_norm).any():
        raise ValueError('NaNs detected in normalized global features during inference.')
    x_tensor = torch.tensor(phi_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        energy = torch.relu(bundle.model(x_tensor)).squeeze(0)
    return float(energy.item())


def make_global_energy(step_size: float,
                       bundle: LearnedGlobalEnergy,
                       fd_eps: float = 1e-3) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        grad = numerical_grad(
            poses,
            lambda x: global_energy(scene, scene.clamp(x), bundle),
            eps=fd_eps,
        )
        update = -step_size * grad
        update[scene.mask] = 0.0
        return update

    return Method(name='global_energy', step_fn=step)


def vector_field_step(scene: SceneSpec, poses: np.ndarray, bundle: LearnedVectorField) -> np.ndarray:
    from experiments.constraint_composition.global_features import extract_global_features

    phi = extract_global_features(poses, scene, max_nodes=bundle.max_nodes).astype(np.float32)
    phi_norm = (phi - bundle.mean) / bundle.std
    if np.isnan(phi_norm).any():
        raise ValueError('NaNs detected in normalized vector-field features during inference.')

    x_tensor = torch.tensor(phi_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        v_flat = bundle.model(x_tensor).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    v_padded = v_flat.reshape(bundle.max_nodes, 2)
    update = np.zeros_like(poses, dtype=np.float32)
    update[:scene.num_nodes, :2] = v_padded[:scene.num_nodes]
    update[scene.mask] = 0.0
    return update


def make_vector_field_method(bundle: LearnedVectorField) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        return vector_field_step(scene, poses, bundle)

    return Method(name='vector_field', step_fn=step)


def vector_field_time_step(
    scene: SceneSpec,
    poses: np.ndarray,
    bundle: LearnedTimeVectorField,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    from experiments.constraint_composition.global_features import extract_global_features

    tau = float(step_idx) / float(max(total_steps, 1))
    phi = extract_global_features(poses, scene, max_nodes=bundle.max_nodes).astype(np.float32)
    phi_time = np.concatenate([phi, np.asarray([tau], dtype=np.float32)], axis=0)
    phi_time = (phi_time - bundle.mean) / bundle.std
    if np.isnan(phi_time).any():
        raise ValueError('NaNs detected in normalized time-conditioned features during inference.')

    x_tensor = torch.tensor(phi_time, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        v_flat = bundle.model(x_tensor).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    v_padded = v_flat.reshape(bundle.max_nodes, 2)
    update = np.zeros_like(poses, dtype=np.float32)
    update[:scene.num_nodes, :2] = v_padded[:scene.num_nodes]
    update[scene.mask] = 0.0
    return update


def make_vector_time_method(bundle: LearnedTimeVectorField) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, t: int = 0, T: int = 1, **_: object) -> np.ndarray:
        return vector_field_time_step(scene, poses, bundle, step_idx=t, total_steps=T)

    return Method(name='vector_time', step_fn=step)


def graph_noise_step(
    scene: SceneSpec,
    poses: np.ndarray,
    bundle: LearnedGraphVectorField,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    import torch

    tau = float(step_idx) / float(max(total_steps, 1))
    tau_column = np.full((scene.num_nodes, 1), tau, dtype=np.float32)
    mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
    node_features = np.concatenate([
        scene.geoms.astype(np.float32, copy=False),
        poses.astype(np.float32, copy=False),
        mask_column,
        tau_column,
    ], axis=1)
    node_features = (node_features - bundle.node_mean) / bundle.node_std
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(scene.edge_index.T, dtype=torch.long)
    edge_attr = torch.tensor(scene.edge_attr, dtype=torch.long)
    with torch.no_grad():
        v = bundle.model(x_tensor, edge_index, edge_attr).cpu().numpy().astype(np.float32, copy=False)

    update = np.zeros_like(poses, dtype=np.float32)
    update[:, :2] = v[:scene.num_nodes]
    update[scene.mask] = 0.0
    return update


def make_graph_noise_method(bundle: LearnedGraphVectorField, name: str) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, t: int = 0, T: int = 1, **_: object) -> np.ndarray:
        return graph_noise_step(scene, poses, bundle, step_idx=t, total_steps=T)

    return Method(name=name, step_fn=step)


def graph_score_step(
    scene: SceneSpec,
    poses: np.ndarray,
    bundle: LearnedGraphVectorField,
    step_size: float,
) -> np.ndarray:
    import torch

    mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
    node_features = np.concatenate([
        scene.geoms.astype(np.float32, copy=False),
        poses.astype(np.float32, copy=False),
        mask_column,
    ], axis=1)
    node_features = (node_features - bundle.node_mean) / bundle.node_std
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(scene.edge_index.T, dtype=torch.long)
    edge_attr = torch.tensor(scene.edge_attr, dtype=torch.long)
    with torch.no_grad():
        score = bundle.model(x_tensor, edge_index, edge_attr).cpu().numpy().astype(np.float32, copy=False)

    update = np.zeros_like(poses, dtype=np.float32)
    update[:, :2] = step_size * score[:scene.num_nodes]
    update[scene.mask] = 0.0
    return update


def make_graph_score_method(bundle: LearnedGraphVectorField, step_size: float, name: str = 'graph_score') -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        return graph_score_step(scene, poses, bundle, step_size=step_size)

    return Method(name=name, step_fn=step)


def graph_score_plus_step(
    scene: SceneSpec,
    poses: np.ndarray,
    bundle: LearnedGraphVectorField,
    step_size: float,
    sigma: float,
    fd_eps: float,
) -> np.ndarray:
    import torch

    grad = numerical_grad(
        poses,
        lambda x: total_violation(scene, scene.clamp(x)),
        eps=fd_eps,
    )
    g = (-grad[:, :2]).astype(np.float32, copy=False)
    g[scene.mask] = 0.0
    norm = np.linalg.norm(g, axis=1, keepdims=True).astype(np.float32) + 1e-6
    g_norm = g / norm

    sigma_col = np.full((scene.num_nodes, 1), sigma, dtype=np.float32)
    mask_column = scene.mask.astype(np.float32).reshape(-1, 1)
    node_features = np.concatenate([
        scene.geoms.astype(np.float32, copy=False),
        poses.astype(np.float32, copy=False),
        mask_column,
        sigma_col,
    ], axis=1)
    node_features = (node_features - bundle.node_mean) / bundle.node_std
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(scene.edge_index.T, dtype=torch.long)
    edge_attr = torch.tensor(scene.edge_attr, dtype=torch.long)
    with torch.no_grad():
        residual = bundle.model(x_tensor, edge_index, edge_attr).cpu().numpy().astype(np.float32, copy=False)

    v = g_norm.copy()
    v[:scene.num_nodes] += residual[:scene.num_nodes]
    v[scene.mask] = 0.0

    update = np.zeros_like(poses, dtype=np.float32)
    update[:, :2] = step_size * v
    update[scene.mask] = 0.0
    return update


def make_graph_score_plus_method(
    bundle: LearnedGraphVectorField,
    step_size: float,
    sigma: float,
    fd_eps: float,
    name: str = 'graph_score_plus',
) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        return graph_score_plus_step(
            scene,
            poses,
            bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
        )

    return Method(name=name, step_fn=step)


def selective_project_state(
    scene: SceneSpec,
    poses: np.ndarray,
    passes: int = 1,
    min_step: float = 0.01,
    topk: int = 0,
) -> np.ndarray:
    state = scene.clamp(poses).astype(np.float32, copy=False)
    current_violation = total_violation(scene, state)
    adaptive_passes = max(passes, 0)
    if current_violation < 0.5:
        adaptive_passes = max(adaptive_passes, 3)
    elif current_violation < 1.0:
        adaptive_passes = max(adaptive_passes, 2)

    for _ in range(adaptive_passes):
        records = violated_constraint_records(scene, state)
        if not records:
            break
        records = sorted(records, key=lambda r: r.violation, reverse=True)
        if topk > 0:
            records = records[:topk]

        updated = False
        for record in records:
            if record.violation <= 0.0:
                continue
            grad = record.grad.astype(np.float32, copy=False)
            grad[scene.mask] = 0.0
            grad_norm_sq = float(np.sum(grad * grad))
            if grad_norm_sq < 1e-8:
                continue

            grad_norm = float(np.sqrt(grad_norm_sq + 1e-8))
            direction = grad / (grad_norm + 1e-8)
            step_mag = float(record.violation) / (1.0 + float(record.violation))
            step_mag = max(step_mag, float(min_step))
            correction = step_mag * direction
            candidate = scene.clamp(state + correction).astype(np.float32, copy=False)
            if total_violation(scene, candidate) <= total_violation(scene, state) + 1e-8:
                state = candidate
                updated = True

        if not updated:
            break

    return state


def make_graph_score_plus_projected_method(
    bundle: LearnedGraphVectorField,
    step_size: float,
    sigma: float,
    fd_eps: float,
    projection_passes: int,
    projection_min_step: float,
    projection_topk: int,
    name: str = 'graph_score_plus_projected',
) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        learned_update = graph_score_plus_step(
            scene,
            poses,
            bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
        )
        proposed = scene.clamp(poses + learned_update).astype(np.float32, copy=False)
        projected = selective_project_state(
            scene,
            proposed,
            passes=projection_passes,
            min_step=projection_min_step,
            topk=projection_topk,
        )
        update = (projected - poses).astype(np.float32, copy=False)
        update[scene.mask] = 0.0
        return update

    return Method(name=name, step_fn=step)


def make_graph_score_proj_method(
    bundle: LearnedGraphVectorField,
    step_size: float,
    sigma: float,
    fd_eps: float,
    projection_passes: int,
    name: str = 'graph_score_proj',
) -> Method:
    from experiments.constraint_composition.graph_score_proj_dataset import selective_project_state_linear

    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        learned_update = graph_score_plus_step(
            scene,
            poses,
            bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
        )
        proposed = scene.clamp(poses + learned_update).astype(np.float32, copy=False)
        projected = selective_project_state_linear(scene, proposed, passes=projection_passes)
        update = (projected - poses).astype(np.float32, copy=False)
        update[scene.mask] = 0.0
        return update

    return Method(name=name, step_fn=step)


def selective_project_state_priority(
    scene: SceneSpec,
    poses: np.ndarray,
    passes: int = 1,
    topk: int = 3,
    threshold: float = 0.01,
) -> np.ndarray:
    state = scene.clamp(poses).astype(np.float32, copy=False)

    for _ in range(max(passes, 0)):
        records = violated_constraint_records(scene, state)
        if not records:
            break

        records = sorted(
            records,
            key=lambda r: (float(r.violation), float(np.linalg.norm(r.grad))),
            reverse=True,
        )
        if topk > 0:
            records = records[:topk]

        updated = False
        for record in records:
            if record.violation < threshold:
                break
            grad = record.grad.astype(np.float32, copy=False)
            grad[scene.mask] = 0.0
            grad_norm_sq = float(np.sum(grad * grad))
            if grad_norm_sq < 1e-8:
                continue

            correction = (float(record.violation) / (grad_norm_sq + 1e-8)) * grad
            candidate = scene.clamp(state + correction).astype(np.float32, copy=False)
            if total_violation(scene, candidate) <= total_violation(scene, state) + 1e-8:
                state = candidate
                updated = True

        if not updated:
            break

    return state


def make_graph_score_plus_priority_projected_method(
    bundle: LearnedGraphVectorField,
    step_size: float,
    sigma: float,
    fd_eps: float,
    projection_passes: int,
    projection_topk: int,
    projection_threshold: float,
    name: str = 'graph_score_plus_priority_projected',
) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        learned_update = graph_score_plus_step(
            scene,
            poses,
            bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
        )
        proposed = scene.clamp(poses + learned_update).astype(np.float32, copy=False)
        projected = selective_project_state_priority(
            scene,
            proposed,
            passes=projection_passes,
            topk=projection_topk,
            threshold=projection_threshold,
        )
        update = (projected - poses).astype(np.float32, copy=False)
        update[scene.mask] = 0.0
        return update

    return Method(name=name, step_fn=step)


def make_sequential_projection(step_size: float, ordering: str = 'descending', passes: int = 2) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, rng: np.random.Generator | None = None, **_: object) -> np.ndarray:
        state = poses.copy()
        generator = rng if rng is not None else np.random.default_rng()

        for _ in range(max(passes, 1)):
            records = violated_constraint_records(scene, state)
            if not records:
                break

            if ordering == 'descending':
                records = sorted(records, key=lambda r: r.violation, reverse=True)
            elif ordering == 'random':
                order = generator.permutation(len(records))
                records = [records[idx] for idx in order]
            elif ordering == 'fixed':
                records = sorted(records, key=lambda r: r.constraint_index)
            else:
                raise ValueError(f'Unknown sequential ordering: {ordering}')

            for record in records:
                step_vec = -step_size * record.grad
                step_vec[scene.mask] = 0.0
                state = scene.clamp(state + step_vec)

        update = state - poses
        update[scene.mask] = 0.0
        return update

    name = 'sequential_projection' if ordering == 'descending' else f'sequential_projection_{ordering}'
    return Method(name=name, step_fn=step)


def exploratory_methods(step_size: float = 0.1) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_energy_descent(step_size=step_size, normalized=True),
        make_consensus(step_size=step_size, weight_mode='uniform'),
        make_consensus(step_size=step_size, weight_mode='violation'),
        make_mixture(step_size=step_size, weight_mode='linear'),
        make_mixture(step_size=step_size, weight_mode='softmax'),
        make_mixture(step_size=step_size, weight_mode='topk'),
    ]


def langevin_methods(step_size: float = 0.1,
                     noise_scales: List[float] | None = None,
                     include_annealed: bool = False) -> List[Method]:
    if noise_scales is None:
        noise_scales = [0.0, 0.01, 0.05, 0.1]

    methods = [make_energy_descent(step_size=step_size, normalized=False)]
    for sigma in noise_scales:
        methods.append(make_energy_langevin(step_size=step_size, noise_scale=sigma, anneal=False))
        if include_annealed and sigma > 0:
            methods.append(make_energy_langevin(step_size=step_size, noise_scale=sigma, anneal=True))
    return methods


def projection_methods(step_size: float = 0.1,
                       alpha: float = 1.0,
                       projection_passes: int = 3,
                       sequential_passes: int = 2,
                       include_langevin_reference: bool = True,
                       include_sequential_variants: bool = True) -> List[Method]:
    methods = [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_sequential_projection(step_size=step_size, ordering='descending', passes=sequential_passes),
    ]
    if include_sequential_variants:
        methods.extend([
            make_sequential_projection(step_size=step_size, ordering='fixed', passes=sequential_passes),
            make_sequential_projection(step_size=step_size, ordering='random', passes=sequential_passes),
        ])
    if include_langevin_reference:
        methods.append(make_energy_langevin(step_size=step_size, noise_scale=0.1, anneal=True))
    return methods


def prototype_methods(step_size: float,
                      prototypes_by_k: Dict[int, Dict[str, List[np.ndarray]]],
                      tau_values: List[float],
                      fd_eps: float = 1e-3,
                      alpha: float = 1.0,
                      projection_passes: int = 3) -> List[Method]:
    methods = [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
    ]
    for k in sorted(prototypes_by_k):
        for tau in tau_values:
            methods.append(
                make_prototype_energy(
                    step_size=step_size,
                    prototypes=prototypes_by_k[k],
                    k=k,
                    tau=tau,
                    fd_eps=fd_eps,
                )
            )
    return methods


def learned_energy_methods(step_size: float,
                           models: Dict[str, LearnedConstraintEnergy],
                           fd_eps: float = 1e-3,
                           alpha: float = 1.0,
                           projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_learned_energy(step_size=step_size, models=models, fd_eps=fd_eps),
    ]


def global_energy_methods(step_size: float,
                          bundle: LearnedGlobalEnergy,
                          fd_eps: float = 1e-3,
                          alpha: float = 1.0,
                          projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_global_energy(step_size=step_size, bundle=bundle, fd_eps=fd_eps),
    ]


def vector_field_methods(step_size: float,
                         bundle: LearnedVectorField,
                         alpha: float = 1.0,
                         projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_vector_field_method(bundle=bundle),
    ]


def vector_time_methods(step_size: float,
                        bundle: LearnedTimeVectorField,
                        alpha: float = 1.0,
                        projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_vector_time_method(bundle=bundle),
    ]


def graph_noise_methods(step_size: float,
                        bundle: LearnedGraphVectorField,
                        method_name: str,
                        alpha: float = 1.0,
                        projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_graph_noise_method(bundle=bundle, name=method_name),
    ]


def graph_score_methods(step_size: float,
                        bundle: LearnedGraphVectorField,
                        alpha: float = 1.0,
                        projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_graph_score_method(bundle=bundle, step_size=step_size, name='graph_score'),
    ]


def graph_score_plus_methods(step_size: float,
                             bundle: LearnedGraphVectorField,
                             sigma: float,
                             fd_eps: float = 1e-3,
                             residual_projection_passes: int = 1,
                             residual_projection_min_step: float = 0.05,
                             residual_projection_topk: int = 3,
                             alpha: float = 1.0,
                             projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_graph_score_plus_method(
            bundle=bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
            name='graph_score_plus',
        ),
        make_graph_score_plus_projected_method(
            bundle=bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
            projection_passes=residual_projection_passes,
            projection_min_step=residual_projection_min_step,
            projection_topk=residual_projection_topk,
            name='graph_score_plus_projected',
        ),
    ]


def graph_score_proj_methods(step_size: float,
                             bundle: LearnedGraphVectorField,
                             sigma: float,
                             fd_eps: float = 1e-3,
                             residual_projection_passes: int = 1,
                             alpha: float = 1.0,
                             projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_graph_score_proj_method(
            bundle=bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
            projection_passes=residual_projection_passes,
            name='graph_score_proj',
        ),
    ]


def graph_score_plus_priority_methods(step_size: float,
                                      bundle: LearnedGraphVectorField,
                                      sigma: float,
                                      fd_eps: float = 1e-3,
                                      residual_projection_passes: int = 1,
                                      residual_projection_topk: int = 3,
                                      residual_projection_threshold: float = 0.01,
                                      alpha: float = 1.0,
                                      projection_passes: int = 3) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_graph_score_plus_priority_projected_method(
            bundle=bundle,
            step_size=step_size,
            sigma=sigma,
            fd_eps=fd_eps,
            projection_passes=residual_projection_passes,
            projection_topk=residual_projection_topk,
            projection_threshold=residual_projection_threshold,
            name='graph_score_plus_priority_projected',
        ),
    ]


def make_graph_score_two_phase_method(
    coarse_bundle: LearnedGraphVectorField,
    refine_bundle: LearnedGraphVectorField,
    step_size: float,
    coarse_sigma: float,
    coarse_fd_eps: float,
    switch_threshold: float,
    switch_temperature: float,
    projection_passes: int,
    refine_gain: float = 1.0,
    name: str = 'graph_score_two_phase',
) -> Method:
    from experiments.constraint_composition.graph_score_proj_dataset import selective_project_state_linear

    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        current_violation = total_violation(scene, poses)
        temperature = max(float(switch_temperature), 1e-6)
        weight = 1.0 / (1.0 + np.exp((float(current_violation) - float(switch_threshold)) / temperature))

        coarse_update = graph_score_plus_step(
            scene,
            poses,
            coarse_bundle,
            step_size=step_size,
            sigma=coarse_sigma,
            fd_eps=coarse_fd_eps,
        )
        refine_update = graph_score_step(
            scene,
            poses,
            refine_bundle,
            step_size=step_size,
        )
        learned_update = (1.0 - weight) * coarse_update + weight * (float(refine_gain) * refine_update)

        proposed = scene.clamp(poses + learned_update).astype(np.float32, copy=False)
        projected = selective_project_state_linear(
            scene,
            proposed,
            passes=projection_passes,
        )
        update = (projected - poses).astype(np.float32, copy=False)
        update[scene.mask] = 0.0
        return update

    return Method(name=name, step_fn=step)


def graph_score_two_phase_methods(step_size: float,
                                  coarse_bundle: LearnedGraphVectorField,
                                  refine_bundle: LearnedGraphVectorField,
                                  coarse_sigma: float,
                                  coarse_fd_eps: float = 1e-3,
                                  switch_threshold: float = 1.0,
                                  switch_temperature: float = 0.3,
                                  residual_projection_passes: int = 1,
                                  alpha: float = 1.0,
                                  projection_passes: int = 3) -> List[Method]:
    gain_sweep = [1.0, 2.0, 3.0, 5.0]
    gain_methods = [
        make_graph_score_two_phase_method(
            coarse_bundle=coarse_bundle,
            refine_bundle=refine_bundle,
            step_size=step_size,
            coarse_sigma=coarse_sigma,
            coarse_fd_eps=coarse_fd_eps,
            switch_threshold=switch_threshold,
            switch_temperature=switch_temperature,
            projection_passes=residual_projection_passes,
            refine_gain=g,
            name=f'graph_score_two_phase_gain{int(g)}',
        )
        for g in gain_sweep
    ]
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        *gain_methods,
    ]


# ---------------------------------------------------------------------------
# Analytical Rectified Flow diagnostic methods
# ---------------------------------------------------------------------------

def compose_per_constraint_velocity(
    scene: SceneSpec,
    poses: np.ndarray,
    max_step: float = 0.1,
) -> tuple[np.ndarray, list[float]]:
    """Compose RF velocities from per-constraint individual projections.

    Each violated constraint is projected independently (all seeing the same
    input state), then velocities are summed.  Returns the composed velocity
    and a list of per-constraint velocity norms for scale-dominance analysis.
    """
    records = evaluate_constraints(scene, poses)
    v = np.zeros_like(poses, dtype=np.float32)
    vel_norms: list[float] = []
    for record in records:
        if record.violation <= 0.0:
            continue
        grad = record.grad.copy().astype(np.float32)
        grad[scene.mask] = 0.0
        grad_norm_sq = float(np.sum(grad * grad))
        if grad_norm_sq < 1e-8:
            continue
        correction = (float(record.violation) / (grad_norm_sq + 1e-8)) * grad
        corr_norm = float(np.linalg.norm(correction))
        if corr_norm > max_step:
            correction = (max_step / corr_norm) * correction
        x1_c = scene.clamp(poses + correction).astype(np.float32)
        v_c = x1_c - poses
        v += v_c
        vel_norms.append(float(np.linalg.norm(v_c)))
    v[scene.mask] = 0.0
    return v, vel_norms


def make_rf_composed_reproject(
    max_step: float = 0.1,
    max_vel_norm: float = 5.0,
) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray,
             t: int = 0, T: int = 1, **_: object) -> np.ndarray:
        v, _ = compose_per_constraint_velocity(scene, poses, max_step=max_step)
        dt = 1.0 / max(T, 1)
        t_norm = t / max(T, 1)
        scale = dt / max(1.0 - t_norm, dt)
        dx = v * float(scale)
        dx_norm = float(np.linalg.norm(dx))
        if dx_norm > max_vel_norm:
            dx = dx * (max_vel_norm / dx_norm)
        dx[scene.mask] = 0.0
        return dx

    return Method(name='rf_composed_reproject', step_fn=step)


def make_rf_composed_fixed(max_step: float = 0.1) -> Method:
    cache: Dict[str, np.ndarray | None] = {'x0': None, 'x1': None}

    def step(scene: SceneSpec, poses: np.ndarray,
             t: int = 0, T: int = 1, **_: object) -> np.ndarray:
        if t == 0:
            v, _ = compose_per_constraint_velocity(scene, poses, max_step=max_step)
            v = np.clip(v, -1.0, 1.0)
            cache['x0'] = poses.copy()
            cache['x1'] = scene.clamp(poses + v).astype(np.float32)
        x0 = cache['x0']
        x1 = cache['x1']
        t_next = (t + 1) / max(T, 1)
        target_next = (1.0 - t_next) * x0 + t_next * x1
        dx = (target_next - poses).astype(np.float32)
        dx[scene.mask] = 0.0
        return dx

    return Method(name='rf_composed_fixed', step_fn=step)


def make_rf_composed_raw(max_step: float = 0.1) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray, **_: object) -> np.ndarray:
        v, _ = compose_per_constraint_velocity(scene, poses, max_step=max_step)
        return v

    return Method(name='rf_composed_raw', step_fn=step)


def make_rf_composed_langevin(
    max_step: float = 0.1,
    noise_scale: float = 0.05,
) -> Method:
    def step(scene: SceneSpec, poses: np.ndarray,
             t: int = 0, T: int = 1,
             rng: np.random.Generator | None = None,
             **_: object) -> np.ndarray:
        v, _ = compose_per_constraint_velocity(scene, poses, max_step=max_step)
        dt = 1.0 / max(T, 1)
        t_norm = t / max(T, 1)
        scale = dt / max(1.0 - t_norm, dt)
        sigma = noise_scale * (1.0 - t_norm)
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.standard_normal(size=poses.shape).astype(np.float32)
        dx = v * float(scale) + float(sigma * np.sqrt(dt)) * noise
        dx[scene.mask] = 0.0
        return dx

    return Method(name='rf_composed_langevin', step_fn=step)


def rectified_flow_methods(
    step_size: float = 0.1,
    alpha: float = 1.0,
    projection_passes: int = 3,
    noise_scale: float = 0.05,
    max_step: float = 0.1,
) -> List[Method]:
    return [
        make_energy_descent(step_size=step_size, normalized=False),
        make_projected_energy(step_size=step_size, alpha=alpha, projection_passes=projection_passes),
        make_rf_composed_reproject(max_step=max_step),
        make_rf_composed_fixed(max_step=max_step),
        make_rf_composed_raw(max_step=max_step),
        make_rf_composed_langevin(max_step=max_step, noise_scale=noise_scale),
    ]
