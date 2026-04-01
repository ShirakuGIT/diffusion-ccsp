from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from experiments.constraint_composition.core import scene_from_data
from experiments.constraint_composition.methods import (
    exploratory_methods,
    global_energy_methods,
    learned_energy_methods,
    langevin_methods,
    projection_methods,
    prototype_methods,
    vector_field_methods,
)
from experiments.constraint_composition.global_energy_dataset import build_global_dataset
from experiments.constraint_composition.global_energy_model import train_global_energy_model
from experiments.constraint_composition.learned_energy_dataset import build_constraint_dataset
from experiments.constraint_composition.learned_energy_model import train_constraint_models
from experiments.constraint_composition.metrics import aggregate_method_runs, evaluate_trajectory
from experiments.constraint_composition.prototypes import build_prototypes
from experiments.constraint_composition.vector_field_dataset import build_vector_field_dataset
from experiments.constraint_composition.vector_field_model import train_vector_field_model


DEFAULT_TASKS = {
    2: 'RandomSplitQualitativeWorld(100)_qualitative_test_2_split',
    3: 'RandomSplitQualitativeWorld(100)_qualitative_test_3_split',
    4: 'RandomSplitQualitativeWorld(100)_qualitative_test_4_split',
    5: 'RandomSplitQualitativeWorld(100)_qualitative_test_5_split',
}


def choose_device(preferred: str) -> str:
    if preferred != 'auto':
        return preferred
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_scenes(task_name: str, max_scenes: int, min_objects: int, max_objects: int):
    dataset = GraphDataset(
        task_name,
        input_mode='qualitative',
        pre_transform=pre_transform,
        visualize=False,
    )
    scenes = []
    for idx, data in enumerate(dataset):
        num_nodes = int(data.x.shape[0])
        if num_nodes < min_objects or num_nodes > max_objects:
            continue
        scenes.append(scene_from_data(data, scene_id=idx, constraint_types=qualitative_constraints))
        if len(scenes) >= max_scenes:
            break
    return scenes


def rollout(scene, method, x0: np.ndarray, steps: int, rng: np.random.Generator) -> List[np.ndarray]:
    trajectory = [scene.clamp(x0)]
    state = trajectory[0]
    for step_idx in range(steps):
        dx = method.step_fn(scene, state, t=step_idx, T=steps, rng=rng)
        state = scene.clamp(state + dx)
        trajectory.append(state.copy())
    return trajectory


def select_methods(args, scenes):
    if args.suite == 'langevin':
        return langevin_methods(
            step_size=args.step_size,
            noise_scales=args.noise_scales,
            include_annealed=args.include_annealed,
        ), None
    if args.suite == 'projection':
        return projection_methods(
            step_size=args.step_size,
            alpha=args.projection_alpha,
            projection_passes=args.projection_passes,
            sequential_passes=args.sequential_passes,
            include_langevin_reference=not args.no_langevin_reference,
            include_sequential_variants=args.include_sequential_variants,
        ), None
    if args.suite == 'prototype':
        prototypes_by_k, prototype_stats = build_prototypes(
            scenes,
            num_samples=args.prototype_samples,
            k_values=args.prototype_k,
            diversity_threshold=args.prototype_diversity_threshold,
            seed=args.seed,
        )
        methods = prototype_methods(
            step_size=args.step_size,
            prototypes_by_k=prototypes_by_k,
            tau_values=args.prototype_tau,
            fd_eps=args.prototype_fd_eps,
            alpha=args.projection_alpha,
            projection_passes=args.projection_passes,
        )
        return methods, prototype_stats
    if args.suite == 'learned':
        constraint_dataset = build_constraint_dataset(
            scenes,
            num_samples=args.learned_dataset_samples,
            seed=args.learned_seed,
        )
        models, train_stats = train_constraint_models(
            constraint_dataset,
            epochs=args.learned_epochs,
            batch_size=args.learned_batch_size,
            lr=args.learned_lr,
            seed=args.learned_seed,
            device='cpu',
        )
        methods = learned_energy_methods(
            step_size=args.step_size,
            models=models,
            fd_eps=args.learned_fd_eps,
            alpha=args.projection_alpha,
            projection_passes=args.projection_passes,
        )
        return methods, {
            'dataset_samples': int(args.learned_dataset_samples),
            'epochs': int(args.learned_epochs),
            'batch_size': int(args.learned_batch_size),
            'lr': float(args.learned_lr),
            'seed': int(args.learned_seed),
            'constraint_stats': train_stats,
        }
    if args.suite == 'global':
        x_arr, y_arr = build_global_dataset(
            scenes,
            num_samples=args.global_dataset_samples,
            seed=args.global_seed,
            max_nodes=args.max_objects,
        )
        bundle, train_stats = train_global_energy_model(
            x_arr,
            y_arr,
            epochs=args.global_epochs,
            batch_size=args.global_batch_size,
            lr=args.global_lr,
            seed=args.global_seed,
            device='cpu',
            max_nodes=args.max_objects,
        )
        methods = global_energy_methods(
            step_size=args.step_size,
            bundle=bundle,
            fd_eps=args.global_fd_eps,
            alpha=args.projection_alpha,
            projection_passes=args.projection_passes,
        )
        return methods, {
            'dataset_samples': int(args.global_dataset_samples),
            'epochs': int(args.global_epochs),
            'batch_size': int(args.global_batch_size),
            'lr': float(args.global_lr),
            'seed': int(args.global_seed),
            'train_stats': train_stats,
        }
    if args.suite == 'vector':
        x_arr, v_arr = build_vector_field_dataset(
            scenes,
            num_samples=args.vector_dataset_samples,
            step_size=args.step_size,
            fd_eps=args.vector_fd_eps,
            seed=args.vector_seed,
            max_nodes=args.max_objects,
        )
        bundle, train_stats = train_vector_field_model(
            x_arr,
            v_arr,
            epochs=args.vector_epochs,
            batch_size=args.vector_batch_size,
            lr=args.vector_lr,
            seed=args.vector_seed,
            device='cpu',
            max_nodes=args.max_objects,
        )
        methods = vector_field_methods(
            step_size=args.step_size,
            bundle=bundle,
            alpha=args.projection_alpha,
            projection_passes=args.projection_passes,
        )
        return methods, {
            'dataset_samples': int(args.vector_dataset_samples),
            'epochs': int(args.vector_epochs),
            'batch_size': int(args.vector_batch_size),
            'lr': float(args.vector_lr),
            'seed': int(args.vector_seed),
            'train_stats': train_stats,
        }
    return exploratory_methods(step_size=args.step_size), None


def build_recommendation(summary: Dict[str, object]) -> Dict[str, object]:
    baseline_name = 'energy'
    baseline = summary.get(baseline_name)
    if baseline is None:
        return {'status': 'no_baseline'}

    projection_like = any(
        name.startswith('projected_energy') or name.startswith('sequential_projection')
        for name in summary
    )
    if projection_like:
        candidates = []
        for method_name, stats in summary.items():
            if method_name == baseline_name or method_name.startswith('energy_langevin_'):
                continue
            improves_feas = stats.get('feasibility_rate', 0.0) > baseline.get('feasibility_rate', 0.0)
            improves_v = stats.get('final_violation_mean', float('inf')) < baseline.get('final_violation_mean', float('inf'))
            valid_direction = stats.get('mean_step_cosine_mean', -1.0) > 0.0
            clears_bar = (
                stats.get('feasibility_rate', 0.0) > 0.2
                and stats.get('final_violation_mean', float('inf')) < 1.29
                and valid_direction
            )
            score = (
                4.0 * float(clears_bar)
                + 3.0 * float(improves_feas)
                + 2.0 * float(improves_v)
                + 1.0 * float(valid_direction)
                + stats.get('feasibility_rate', 0.0)
                - stats.get('final_violation_mean', 0.0)
                - 0.1 * stats.get('final_num_violated_constraints_mean', 0.0)
            )
            candidates.append({
                'method': method_name,
                'score': score,
                'improves_feasibility': improves_feas,
                'improves_final_violation': improves_v,
                'valid_direction': valid_direction,
                'clears_acceptance_bar': clears_bar,
                'feasibility_rate': stats.get('feasibility_rate', 0.0),
                'final_violation_mean': stats.get('final_violation_mean', 0.0),
                'mean_step_cosine_mean': stats.get('mean_step_cosine_mean', 0.0),
                'plateau_fraction_mean': stats.get('plateau_fraction_mean', 0.0),
                'final_num_violated_constraints_mean': stats.get('final_num_violated_constraints_mean', 0.0),
            })
        if not candidates:
            return {'status': 'no_projection_candidates'}
        best = max(candidates, key=lambda item: item['score'])
        return {
            'status': 'ok',
            'baseline_method': baseline_name,
            'best_method': best['method'],
            'acceptance_bar_passed': best['clears_acceptance_bar'],
            'candidates': candidates,
        }

    candidates = []
    for method_name, stats in summary.items():
        if not method_name.startswith('energy_langevin_'):
            continue
        improves_feas = stats.get('feasibility_rate', 0.0) > baseline.get('feasibility_rate', 0.0)
        improves_v = stats.get('final_violation_mean', float('inf')) < baseline.get('final_violation_mean', float('inf'))
        valid_direction = stats.get('mean_step_cosine_mean', -1.0) > 0.0
        score = (
            3.0 * float(improves_feas)
            + 2.0 * float(improves_v)
            + 1.0 * float(valid_direction)
            + stats.get('feasibility_rate', 0.0)
            - stats.get('final_violation_mean', 0.0)
        )
        candidates.append({
            'method': method_name,
            'score': score,
            'improves_feasibility': improves_feas,
            'improves_final_violation': improves_v,
            'valid_direction': valid_direction,
            'feasibility_rate': stats.get('feasibility_rate', 0.0),
            'final_violation_mean': stats.get('final_violation_mean', 0.0),
            'mean_step_cosine_mean': stats.get('mean_step_cosine_mean', 0.0),
            'plateau_fraction_mean': stats.get('plateau_fraction_mean', 0.0),
        })

    if not candidates:
        return {'status': 'no_langevin_candidates'}

    best = max(candidates, key=lambda item: item['score'])
    best_sigma = best['method'].split('sigma', 1)[1].split('_', 1)[0] if 'sigma' in best['method'] else None
    annealed_better = any(
        item['method'].endswith('_annealed') and item['score'] >= best['score']
        for item in candidates
    )
    return {
        'status': 'ok',
        'baseline_method': baseline_name,
        'best_method': best['method'],
        'best_sigma': best_sigma,
        'annealing_helped': annealed_better,
        'candidates': candidates,
    }


def run_experiment(args) -> Dict[str, object]:
    task_name = DEFAULT_TASKS[args.split]
    scenes = load_scenes(task_name, args.max_scenes, args.min_objects, args.max_objects)
    if not scenes:
        raise RuntimeError(
            f'No scenes matched object-count filter {args.min_objects}-{args.max_objects} '
            f'for task {task_name}.'
        )
    methods, aux_stats = select_methods(args, scenes)

    results: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    trials = []

    for scene_idx, scene in enumerate(scenes):
        for trial_idx in range(args.trials_per_scene):
            seed = args.seed + scene_idx * 1000 + trial_idx
            rng = np.random.default_rng(seed)
            x0 = scene.initialize_state(rng)

            trial_record = {
                'scene_id': scene.scene_id,
                'trial_idx': trial_idx,
                'seed': seed,
                'initial_violation': evaluate_trajectory(scene, [x0], args.feasibility_eps)['final_violation'],
                'methods': {},
            }

            for method_idx, method in enumerate(methods):
                method_rng = np.random.default_rng(seed + 10000 * (method_idx + 1))
                trajectory = rollout(scene, method, x0, args.steps, method_rng)
                metrics = evaluate_trajectory(
                    scene,
                    trajectory,
                    args.feasibility_eps,
                    plateau_threshold=args.plateau_threshold,
                )
                results[method.name].append(metrics)
                trial_record['methods'][method.name] = {
                    'final_violation': metrics['final_violation'],
                    'feasible': metrics['feasible'],
                    'mean_step_cosine': metrics['mean_step_cosine'],
                    'monotonic_fraction': metrics['monotonic_fraction'],
                    'num_plateau_steps': metrics['num_plateau_steps'],
                    'final_num_violated_constraints': metrics['final_num_violated_constraints'],
                }
            trials.append(trial_record)

    summary = {name: aggregate_method_runs(runs) for name, runs in results.items()}
    recommendation = build_recommendation(summary)
    return {
        'config': {
            'task_name': task_name,
            'split': args.split,
            'max_scenes': args.max_scenes,
            'min_objects': args.min_objects,
            'max_objects': args.max_objects,
            'trials_per_scene': args.trials_per_scene,
            'steps': args.steps,
            'step_size': args.step_size,
            'suite': args.suite,
            'noise_scales': args.noise_scales,
            'include_annealed': args.include_annealed,
            'projection_alpha': args.projection_alpha,
            'projection_passes': args.projection_passes,
            'sequential_passes': args.sequential_passes,
            'include_sequential_variants': args.include_sequential_variants,
            'langevin_reference': not args.no_langevin_reference,
            'prototype_samples': args.prototype_samples,
            'prototype_k': args.prototype_k,
            'prototype_diversity_threshold': args.prototype_diversity_threshold,
            'prototype_fd_eps': args.prototype_fd_eps,
            'prototype_tau': args.prototype_tau,
            'learned_dataset_samples': args.learned_dataset_samples,
            'learned_epochs': args.learned_epochs,
            'learned_batch_size': args.learned_batch_size,
            'learned_lr': args.learned_lr,
            'learned_fd_eps': args.learned_fd_eps,
            'learned_seed': args.learned_seed,
            'global_dataset_samples': args.global_dataset_samples,
            'global_epochs': args.global_epochs,
            'global_batch_size': args.global_batch_size,
            'global_lr': args.global_lr,
            'global_fd_eps': args.global_fd_eps,
            'global_seed': args.global_seed,
            'vector_dataset_samples': args.vector_dataset_samples,
            'vector_epochs': args.vector_epochs,
            'vector_batch_size': args.vector_batch_size,
            'vector_lr': args.vector_lr,
            'vector_fd_eps': args.vector_fd_eps,
            'vector_seed': args.vector_seed,
            'feasibility_eps': args.feasibility_eps,
            'plateau_threshold': args.plateau_threshold,
            'seed': args.seed,
            'device': args.device,
        },
        'aux_stats': aux_stats,
        'summary': summary,
        'recommendation': recommendation,
        'trials': trials,
    }


def print_summary(summary: Dict[str, object]) -> None:
    print('\nMethod summary\n--------------')
    for method_name, stats in summary.items():
        print(
            f"{method_name:20s} "
            f"feas={stats.get('feasibility_rate', 0.0):.3f}  "
            f"final_V={stats.get('final_violation_mean', 0.0):.4f}  "
            f"viol={stats.get('final_num_violated_constraints_mean', 0.0):.2f}  "
            f"mono={stats.get('monotonic_fraction_mean', 0.0):.3f}  "
            f"cos={stats.get('mean_step_cosine_mean', 0.0):.3f}  "
            f"plateau={stats.get('plateau_fraction_mean', 0.0):.3f}"
        )


def maybe_plot_summary(summary: Dict[str, object], output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.reshape(-1)
    specs = [
        ('trajectory_violation_mean_series', 'Mean V(x_t)', 'Violation'),
        ('trajectory_violated_constraints_mean_series', 'Mean # Violated Constraints', 'Count'),
        ('step_cosine_mean_series', 'Mean cos(Δx, -∇V)', 'Cosine'),
        ('step_size_mean_series', 'Mean |Δx|', 'Step size'),
    ]
    for ax, (key, title, ylabel) in zip(axes, specs):
        for method_name, stats in summary.items():
            series = stats.get(key, [])
            if series:
                ax.plot(series, label=method_name)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    plot_path = output_path.with_suffix('.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def parse_args():
    parser = argparse.ArgumentParser(description='Constraint composition experiment harness')
    parser.add_argument('--suite', type=str, default='projection', choices=['langevin', 'projection', 'prototype', 'learned', 'global', 'vector', 'explore'])
    parser.add_argument('--split', type=int, default=3, choices=sorted(DEFAULT_TASKS))
    parser.add_argument('--max-scenes', type=int, default=20)
    parser.add_argument('--min-objects', type=int, default=3)
    parser.add_argument('--max-objects', type=int, default=6)
    parser.add_argument('--trials-per-scene', type=int, default=1)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--step-size', type=float, default=0.1)
    parser.add_argument('--feasibility-eps', type=float, default=0.05)
    parser.add_argument('--plateau-threshold', type=float, default=1e-3)
    parser.add_argument('--noise-scales', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument('--include-annealed', action='store_true')
    parser.add_argument('--projection-alpha', type=float, default=1.0)
    parser.add_argument('--projection-passes', type=int, default=3)
    parser.add_argument('--sequential-passes', type=int, default=2)
    parser.add_argument('--include-sequential-variants', action='store_true')
    parser.add_argument('--no-langevin-reference', action='store_true')
    parser.add_argument('--prototype-samples', type=int, default=2000)
    parser.add_argument('--prototype-k', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--prototype-tau', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.5])
    parser.add_argument('--prototype-diversity-threshold', type=float, default=0.1)
    parser.add_argument('--prototype-fd-eps', type=float, default=1e-3)
    parser.add_argument('--learned-dataset-samples', type=int, default=5000)
    parser.add_argument('--learned-epochs', type=int, default=10)
    parser.add_argument('--learned-batch-size', type=int, default=128)
    parser.add_argument('--learned-lr', type=float, default=1e-3)
    parser.add_argument('--learned-fd-eps', type=float, default=1e-3)
    parser.add_argument('--learned-seed', type=int, default=0)
    parser.add_argument('--global-dataset-samples', type=int, default=5000)
    parser.add_argument('--global-epochs', type=int, default=10)
    parser.add_argument('--global-batch-size', type=int, default=128)
    parser.add_argument('--global-lr', type=float, default=1e-3)
    parser.add_argument('--global-fd-eps', type=float, default=1e-3)
    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--vector-dataset-samples', type=int, default=5000)
    parser.add_argument('--vector-epochs', type=int, default=10)
    parser.add_argument('--vector-batch-size', type=int, default=128)
    parser.add_argument('--vector-lr', type=float, default=1e-3)
    parser.add_argument('--vector-fd-eps', type=float, default=1e-3)
    parser.add_argument('--vector-seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = choose_device(args.device)
    _ = args.device  # reserved for future device-specific logic

    result = run_experiment(args)
    print_summary(result['summary'])
    if result['recommendation'].get('status') == 'ok':
        print('\nRecommendation')
        print('--------------')
        if args.suite == 'langevin':
            print(
                f"best={result['recommendation']['best_method']}  "
                f"sigma={result['recommendation']['best_sigma']}  "
                f"annealing_helped={result['recommendation']['annealing_helped']}"
            )
        else:
            print(
                f"best={result['recommendation']['best_method']}  "
                f"acceptance_bar_passed={result['recommendation']['acceptance_bar_passed']}"
            )

    output_path = Path(args.output) if args.output else (
        Path('experiments/constraint_composition/results') /
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f'\nSaved results to {output_path}')

    if args.plot:
        plot_path = maybe_plot_summary(result['summary'], output_path)
        if plot_path is not None:
            print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
