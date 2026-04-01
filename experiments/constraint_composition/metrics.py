from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.constraint_composition.core import (
    SceneSpec,
    cosine_similarity,
    total_violation,
    total_violation_gradient,
    violated_constraint_records,
)


def evaluate_trajectory(scene: SceneSpec, trajectory: List[np.ndarray], feasibility_eps: float,
                        plateau_threshold: float = 1e-3) -> Dict[str, object]:
    violations = [total_violation(scene, poses) for poses in trajectory]
    delta_vs = [violations[t + 1] - violations[t] for t in range(len(violations) - 1)]
    violated_counts = [len(violated_constraint_records(scene, poses)) for poses in trajectory]
    step_sizes = []
    cosines = []

    for t in range(len(trajectory) - 1):
        x_t = trajectory[t]
        x_next = trajectory[t + 1]
        dx = x_next - x_t
        grad_v = total_violation_gradient(scene, x_t)
        step_sizes.append(float(np.linalg.norm(dx.reshape(-1))))
        cosines.append(cosine_similarity(dx, -grad_v))

    final_violation = float(violations[-1])
    result = {
        'trajectory_violation': [float(v) for v in violations],
        'trajectory_violated_constraints': [int(v) for v in violated_counts],
        'delta_violation': [float(v) for v in delta_vs],
        'step_cosine': [float(v) for v in cosines],
        'step_size': [float(v) for v in step_sizes],
        'monotonic_fraction': float(np.mean([dv <= 0.0 for dv in delta_vs])) if delta_vs else 1.0,
        'num_plateau_steps': int(sum(abs(dv) < plateau_threshold for dv in delta_vs)),
        'plateau_fraction': float(np.mean([abs(dv) < plateau_threshold for dv in delta_vs])) if delta_vs else 0.0,
        'mean_delta_violation': float(np.mean(delta_vs)) if delta_vs else 0.0,
        'mean_step_cosine': float(np.mean(cosines)) if cosines else 1.0,
        'mean_step_size': float(np.mean(step_sizes)) if step_sizes else 0.0,
        'max_step_size': float(np.max(step_sizes)) if step_sizes else 0.0,
        'final_violation': final_violation,
        'final_num_violated_constraints': int(violated_counts[-1]) if violated_counts else 0,
        'feasible': bool(final_violation < feasibility_eps),
    }
    return result


def aggregate_method_runs(runs: List[Dict[str, object]]) -> Dict[str, object]:
    if not runs:
        return {}

    series_keys = ['trajectory_violation', 'trajectory_violated_constraints', 'delta_violation', 'step_cosine', 'step_size']
    summary: Dict[str, object] = {'num_runs': len(runs)}
    for key in ['monotonic_fraction', 'num_plateau_steps', 'plateau_fraction',
                'mean_delta_violation', 'mean_step_cosine',
                'mean_step_size', 'max_step_size', 'final_violation', 'final_num_violated_constraints']:
        values = [float(run[key]) for run in runs]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))

    summary['feasibility_rate'] = float(np.mean([bool(run['feasible']) for run in runs]))

    for key in series_keys:
        max_len = max(len(run[key]) for run in runs)
        padded = np.full((len(runs), max_len), np.nan, dtype=np.float32)
        for i, run in enumerate(runs):
            vals = np.asarray(run[key], dtype=np.float32)
            padded[i, :len(vals)] = vals
        summary[f'{key}_mean_series'] = np.nanmean(padded, axis=0).tolist()

    return summary
