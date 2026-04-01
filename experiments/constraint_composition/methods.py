from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from experiments.constraint_composition.core import (
    SceneSpec,
    total_violation_gradient,
    violated_constraint_records,
)
from experiments.constraint_composition.prototypes import numerical_grad, prototype_energy


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
