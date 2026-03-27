"""
Model Audit & Diagnostic Pipeline
===================================
Evaluates all available flow models to determine: fine-tune / partial-reset / retrain.

Tests (on identical dataset + seeds):
  1. Directional Alignment   — cosine similarity of v_pred vs true velocity u_t
  2. One-step vs Multi-step Gap — shortcut learning detection
  3. Constraint Quality       — violation rate, severity, recovery under perturbation
  4. Stability                — long-horizon integration (40 steps) stability
  5. Robustness               — sensitivity to input noise

Scoring: each test → [0, 1], aggregated into a final verdict.

Usage:
    python audit_models.py
"""

import os, sys, json, time
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'networks'))

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints
from train_flow import FlowMatchingCCSP, get_data_config
from fix_and_eval import clamp_to_tray, compute_barrier


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def build_registry():
    """Discover all candidate models + metadata."""
    import importlib.util

    def load_cls(script, cls_name):
        spec = importlib.util.spec_from_file_location("mod", script)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            return getattr(mod, cls_name, FlowMatchingCCSP)
        except Exception:
            return FlowMatchingCCSP

    candidates = [
        {
            "label": "v1_best (200k steps, OT-CFM)",
            "path": "logs/flow_qualitative_h256/flow_model_best.pt",
            "cls": FlowMatchingCCSP,
            "version": "v1",
            "extra": {},
        },
        {
            "label": "v2_best (10k steps, target-prediction compose)",
            "path": "logs/flow_v2_qualitative_h256/flow_v2_model_best.pt",
            "cls": load_cls("train_flow_v2.py", "FlowMatchingCCSP_v2"),
            "version": "v2",
            "extra": {},
        },
        {
            "label": "v3_best (10k steps, constraint-violation loss)",
            "path": "logs/flow_v3_qualitative_h256/flow_v3_model_best.pt",
            "cls": load_cls("train_flow_v3.py", "FlowMatchingCCSP_v3"),
            "version": "v3",
            "extra": {},
        },
        {
            "label": "v4_best (10k steps, fine-tune + onestep feasibility)",
            "path": "logs/flow_v4_qualitative_h256/flow_model_best.pt",
            "cls": load_cls("train_flow_v4.py", "FlowMatchingCCSP_v4"),
            "version": "v4",
            "extra": {"lambda_onestep": 1.0, "lambda_constraint": 0.0},
        },
    ]

    registry = []
    for c in candidates:
        if not os.path.exists(c["path"]):
            continue
        ck = torch.load(c["path"], map_location="cpu")
        step = ck.get("step", "?")
        loss = ck.get("best_loss", ck.get("loss", "?"))
        registry.append({**c, "step": step, "best_loss": loss, "ckpt": ck})
    return registry


def load_model(entry, dims, constraint_types, device):
    cls = entry["cls"]
    extra = entry.get("extra", {})
    try:
        model = cls(dims=dims, hidden_dim=256, constraint_types=constraint_types,
                    normalize=True, device=device, **extra).to(device)
    except TypeError:
        # fallback: v2/v3 might not accept extra kwargs
        model = cls(dims=dims, hidden_dim=256, constraint_types=constraint_types,
                    normalize=True, device=device).to(device)
    sd = entry["ckpt"].get("model_state_dict", entry["ckpt"])
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Barrier helpers
# ─────────────────────────────────────────────────────────────────────────────

def scene_violation(poses_np, edge_index, edge_attr, geoms_np, constraint_types):
    """Returns (mean_violation, n_violated, n_total)."""
    violations = []
    for ei in range(edge_index.shape[0]):
        i, j = int(edge_index[ei, 0]), int(edge_index[ei, 1])
        ctype_idx = int(edge_attr[ei])
        if ctype_idx >= len(constraint_types):
            continue
        ctype = constraint_types[ctype_idx]
        h, _, _ = compute_barrier(
            ctype,
            torch.tensor(poses_np[i], dtype=torch.float32),
            torch.tensor(poses_np[j], dtype=torch.float32),
            torch.tensor(geoms_np[i], dtype=torch.float32),
            torch.tensor(geoms_np[j], dtype=torch.float32),
        )
        violations.append(min(h, 0.0))
    if not violations:
        return 0.0, 0, 0
    n_viol = sum(1 for v in violations if v < -0.02)
    return -np.mean(violations), n_viol, len(violations)


# ─────────────────────────────────────────────────────────────────────────────
# Individual tests
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_directional_alignment(model, loader, dims, n_scenes=30, device='cuda'):
    """
    Cosine similarity between predicted velocity v_pred and true OT velocity u_t = x1 - x0.
    Measures: does the model know the *direction* to move, not just magnitude?

    Score: mean cosine similarity over free nodes (higher = better).
    Range: [-1, 1], clipped to [0, 1] for scoring.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    cos_sims = []
    n = 0
    for data in loader:
        if n >= n_scenes:
            break
        n += 1
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end]
        mask = batch.mask.bool().to(device)
        free = ~mask

        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            torch.manual_seed(n * 100)
            x_0 = torch.randn_like(x_1)
            x_t = (1.0 - t_val) * x_0 + t_val * x_1
            x_t[mask] = x_1[mask]
            u_t = x_1 - x_0  # true velocity

            v_pred = model(x_t, batch, t_val)

            if free.sum() > 0:
                sim = F.cosine_similarity(
                    v_pred[free].reshape(-1),
                    u_t[free].reshape(-1),
                    dim=0
                ).item()
                cos_sims.append(sim)

    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
    score = float(np.clip((mean_cos + 1.0) / 2.0, 0, 1))
    return {"mean_cosine_similarity": mean_cos, "score": score,
            "interpretation": (
                "HIGH: model knows correct direction" if mean_cos > 0.6
                else "MODERATE: partial geometry" if mean_cos > 0.2
                else "LOW: broken vector field"
            )}


@torch.no_grad()
def test_onestep_vs_rollout_gap(model, loader, dims, n_scenes=30, n_steps=20, device='cuda'):
    """
    One-step endpoint error vs multi-step rollout error.
    Detects shortcut learning.

    one_step_err: ||x_0 + v_pred(x_0, t=0) - x_1||
    rollout_err:  ||rollout(x_0) - x_1||

    gap = rollout_err - one_step_err
    Small gap → consistent global field.
    Large gap → model only learned local/shortcut behavior.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    one_step_errs, rollout_errs = [], []
    n = 0
    for data in loader:
        if n >= n_scenes:
            break
        n += 1
        torch.manual_seed(n * 7)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask

        x_0 = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_0[mask] = x_1[mask]

        # One-step prediction: use velocity at t=0 to jump to x_1
        v0 = model(x_0, batch, 0.0)
        x_pred_one = x_0 + v0  # dt=1 conceptually
        err_one = (x_pred_one[free] - x_1[free]).norm().item() / max(free.sum().item(), 1)
        one_step_errs.append(err_one)

        # Multi-step rollout
        x_t = x_0.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = step * dt
            v = model(x_t, batch, t)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        err_roll = (x_t[free] - x_1[free]).norm().item() / max(free.sum().item(), 1)
        rollout_errs.append(err_roll)

    mean_one = float(np.mean(one_step_errs))
    mean_roll = float(np.mean(rollout_errs))
    gap = mean_roll - mean_one
    # Score: low gap = good. Normalize: gap=0 → 1.0, gap=1.0 → 0.0
    score = float(np.clip(1.0 - gap / 1.0, 0, 1))
    return {
        "one_step_err": mean_one, "rollout_err": mean_roll,
        "gap": gap, "score": score,
        "interpretation": (
            "SMALL GAP: global field consistent" if gap < 0.1
            else "MODERATE GAP: some shortcut learning" if gap < 0.3
            else "LARGE GAP: shortcut/broken field"
        )
    }


@torch.no_grad()
def test_constraint_quality(model, loader, dims, n_scenes=30, n_steps=20, device='cuda'):
    """
    Three subtests:
    (A) Violation rate & severity after rollout
    (B) Recovery: perturb valid pose into invalid, run 5 flow steps, does violation reduce?

    Score: combined.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    viol_severities, viol_rates = [], []
    recoveries = []
    n = 0

    for data in loader:
        if n >= n_scenes:
            break
        n += 1
        torch.manual_seed(n * 13)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)

        edge_index_np = batch.edge_index.T.cpu().numpy()
        edge_attr_np = batch.edge_attr.cpu().numpy()
        geoms_np = geoms.cpu().numpy()

        # (A) Rollout violation
        x_t = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_t[mask] = x_1[mask]
        dt = 1.0 / n_steps
        for step in range(n_steps):
            v = model(x_t, batch, step * dt)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

        sev, n_viol, n_total = scene_violation(
            x_t.cpu().numpy(), edge_index_np, edge_attr_np,
            geoms_np, qualitative_constraints)
        viol_severities.append(sev)
        viol_rates.append(n_viol / max(n_total, 1))

        # (B) Recovery test: start from x_1 + large noise, run 5 flow steps
        # Measure if violation decreases
        free = ~mask
        x_perturbed = x_1.clone()
        x_perturbed[free] += torch.randn_like(x_perturbed[free]) * 0.5
        x_perturbed = clamp_to_tray(x_perturbed, geoms, mask, pose_dim)

        sev_before, _, _ = scene_violation(
            x_perturbed.cpu().numpy(), edge_index_np, edge_attr_np,
            geoms_np, qualitative_constraints)

        x_rec = x_perturbed.clone()
        dt_rec = 1.0 / 5
        for step in range(5):
            t = 0.7 + step * dt_rec * 0.3  # late in trajectory
            v = model(x_rec, batch, t)
            x_rec = x_rec + v * dt_rec
            x_rec[mask] = x_1[mask]
            x_rec = clamp_to_tray(x_rec, geoms, mask, pose_dim)

        sev_after, _, _ = scene_violation(
            x_rec.cpu().numpy(), edge_index_np, edge_attr_np,
            geoms_np, qualitative_constraints)

        # Recovery = reduction in violation (positive = improved)
        recovery = sev_before - sev_after
        recoveries.append(recovery)

    mean_sev = float(np.mean(viol_severities))
    mean_rate = float(np.mean(viol_rates))
    mean_rec = float(np.mean(recoveries))

    # Score: low severity good, high recovery good
    sev_score = float(np.clip(1.0 - mean_sev / 0.3, 0, 1))
    rec_score = float(np.clip((mean_rec + 0.1) / 0.2, 0, 1))
    score = (sev_score + rec_score) / 2.0

    return {
        "violation_severity": mean_sev,
        "violation_rate": mean_rate,
        "recovery_delta": mean_rec,
        "score": score,
        "interpretation": (
            "STRONG: constraints encoded" if mean_sev < 0.05 and mean_rec > 0.02
            else "PARTIAL: learnable via fine-tune" if mean_sev < 0.15 or mean_rec > 0.0
            else "WEAK: no constraint recovery"
        )
    }


@torch.no_grad()
def test_stability(model, loader, dims, n_scenes=30, n_steps_long=40, device='cuda'):
    """
    Run long-horizon integration (40 steps, 2x normal).
    Check: does it explode, collapse, or remain bounded?

    Score: fraction of scenes with final pose norm < 10 (sanity bound).
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    final_norms, mid_norms = [], []
    n = 0
    for data in loader:
        if n >= n_scenes:
            break
        n += 1
        torch.manual_seed(n * 17)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask

        x_t = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_t[mask] = x_1[mask]

        dt = 1.0 / n_steps_long
        for step in range(n_steps_long):
            t = step * dt
            v = model(x_t, batch, t)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

            if step == n_steps_long // 2:
                mid_norms.append(x_t[free].norm().item() / max(free.sum().item(), 1))

        final_norms.append(x_t[free].norm().item() / max(free.sum().item(), 1))

    mean_final = float(np.mean(final_norms))
    mean_mid = float(np.mean(mid_norms))
    # Stability: poses should stay in roughly [0, 2]^2 per component
    # Expected norm for 2D coords in [0,2]: ~sqrt(2)*1 ≈ 1.4 per node
    stable_frac = float(np.mean([n < 5.0 for n in final_norms]))
    score = stable_frac

    return {
        "mean_final_norm": mean_final,
        "mean_mid_norm": mean_mid,
        "stable_fraction": stable_frac,
        "score": score,
        "interpretation": (
            "STABLE: well-bounded trajectories" if stable_frac > 0.9
            else "MODERATE instability" if stable_frac > 0.7
            else "UNSTABLE: exploding trajectories"
        )
    }


@torch.no_grad()
def test_robustness(model, loader, dims, n_scenes=30, device='cuda'):
    """
    Input noise sensitivity: add small epsilon noise to x_t, measure ||Δv||/||Δx||.
    Low sensitivity = smooth, well-structured field (good).
    Very low = might be degenerate.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    pose_dim = dims[-1][0]

    sensitivities = []
    n = 0
    for data in loader:
        if n >= n_scenes:
            break
        n += 1
        torch.manual_seed(n * 23)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask

        x_0 = torch.randn_like(x_1)
        x_0[mask] = x_1[mask]

        for t_val in [0.3, 0.7]:
            x_t = (1.0 - t_val) * x_0 + t_val * x_1
            x_t[mask] = x_1[mask]

            v_clean = model(x_t, batch, t_val)

            epsilon = 0.05
            noise = torch.randn_like(x_t) * epsilon
            noise[mask] = 0.0
            x_noisy = x_t + noise

            v_noisy = model(x_noisy, batch, t_val)

            if free.sum() > 0:
                dv = (v_noisy[free] - v_clean[free]).norm().item()
                dx = noise[free].norm().item()
                if dx > 1e-8:
                    sensitivities.append(dv / dx)

    mean_sens = float(np.mean(sensitivities)) if sensitivities else 0.0
    # Lipschitz estimate. < 2 is smooth, > 10 is very sensitive
    score = float(np.clip(1.0 - (mean_sens - 1.0) / 10.0, 0, 1))

    return {
        "mean_sensitivity": mean_sens,
        "score": score,
        "interpretation": (
            "SMOOTH: Lipschitz well-behaved" if mean_sens < 3.0
            else "MODERATE sensitivity" if mean_sens < 8.0
            else "HIGH sensitivity: noisy field"
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decision logic
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(scores):
    """
    scores: dict with keys 'directional_alignment', 'rollout_gap', 'constraint', 'stability', 'robustness'
    Returns: ('fine-tune' | 'partial-reset' | 'retrain', reason)
    """
    da = scores.get('directional_alignment', {}).get('score', 0)
    rg = scores.get('rollout_gap', {}).get('score', 0)
    cq = scores.get('constraint', {}).get('score', 0)
    st = scores.get('stability', {}).get('score', 0)
    rob = scores.get('robustness', {}).get('score', 0)

    global_field_score = (da + rg + st + rob) / 4.0
    constraint_score = cq

    # Composite
    composite = 0.6 * global_field_score + 0.4 * constraint_score

    if global_field_score >= 0.6 and constraint_score >= 0.3:
        verdict = "FINE-TUNE"
        reason = (f"Good global geometry (score={global_field_score:.2f}). "
                  f"Constraint quality (score={constraint_score:.2f}) is improvable "
                  f"via adding feasibility loss to training.")
    elif global_field_score >= 0.45:
        verdict = "PARTIAL-RESET"
        reason = (f"Moderate global field (score={global_field_score:.2f}). "
                  f"Representation is useful but final layers may be corrupted. "
                  f"Reinitialize pose_decoder + constraint_mlps, retrain with both "
                  f"CFM + feasibility losses.")
    else:
        verdict = "RETRAIN"
        reason = (f"Weak global field (score={global_field_score:.2f}). "
                  f"Fine-tuning will not recover correct transport structure. "
                  f"Retrain from scratch with feasibility-aware objective.")

    return verdict, reason, {
        "global_field": round(global_field_score, 3),
        "constraint": round(constraint_score, 3),
        "composite": round(composite, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main audit loop
# ─────────────────────────────────────────────────────────────────────────────

def run_audit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'#'*70}")
    print(f"# MODEL AUDIT — Flow-CCSP Checkpoint Registry")
    print(f"# Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'#'*70}")

    _, _, dims, constraint_types = get_data_config('qualitative')

    # Load test dataset (fixed 30 scenes, 3-obj, same seeds for all models)
    test_task = "RandomSplitQualitativeWorld(100)_qualitative_test_3_split"
    dataset = GraphDataset(test_task, input_mode='qualitative',
                           pre_transform=pre_transform, visualize=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"\n  Test set: {len(dataset)} scenes (3-obj), using first 30 for audit")

    registry = build_registry()
    print(f"\n  Found {len(registry)} models:")
    for r in registry:
        print(f"    [{r['version']}] {r['label']}")
        print(f"           path={r['path']}")
        print(f"           step={r['step']}, best_loss={r['best_loss']}")

    all_reports = {}

    for entry in registry:
        label = entry['label']
        print(f"\n\n{'='*65}")
        print(f"  AUDITING: {label}")
        print(f"{'='*65}")

        try:
            model = load_model(entry, dims, constraint_types, device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            all_reports[label] = {"error": str(e)}
            continue

        scores = {}

        print(f"  [1/5] Directional Alignment...", end=" ", flush=True)
        scores['directional_alignment'] = test_directional_alignment(
            model, loader, dims, n_scenes=30, device=device)
        r = scores['directional_alignment']
        print(f"cosine={r['mean_cosine_similarity']:.3f}  score={r['score']:.2f}  | {r['interpretation']}")

        print(f"  [2/5] One-step vs Rollout Gap...", end=" ", flush=True)
        scores['rollout_gap'] = test_onestep_vs_rollout_gap(
            model, loader, dims, n_scenes=30, device=device)
        r = scores['rollout_gap']
        print(f"one={r['one_step_err']:.3f} roll={r['rollout_err']:.3f} gap={r['gap']:.3f}  score={r['score']:.2f}  | {r['interpretation']}")

        print(f"  [3/5] Constraint Quality...", end=" ", flush=True)
        scores['constraint'] = test_constraint_quality(
            model, loader, dims, n_scenes=30, device=device)
        r = scores['constraint']
        print(f"sev={r['violation_severity']:.3f} rate={r['violation_rate']:.2f} recovery={r['recovery_delta']:.4f}  score={r['score']:.2f}  | {r['interpretation']}")

        print(f"  [4/5] Stability (40 steps)...", end=" ", flush=True)
        scores['stability'] = test_stability(
            model, loader, dims, n_scenes=30, device=device)
        r = scores['stability']
        print(f"stable_frac={r['stable_fraction']:.2f}  score={r['score']:.2f}  | {r['interpretation']}")

        print(f"  [5/5] Robustness...", end=" ", flush=True)
        scores['robustness'] = test_robustness(
            model, loader, dims, n_scenes=30, device=device)
        r = scores['robustness']
        print(f"sensitivity={r['mean_sensitivity']:.2f}  score={r['score']:.2f}  | {r['interpretation']}")

        verdict, reason, summary_scores = make_decision(scores)

        print(f"\n  ── VERDICT: {verdict} ──")
        print(f"  Global Field: {summary_scores['global_field']:.2f}  |  Constraint: {summary_scores['constraint']:.2f}  |  Composite: {summary_scores['composite']:.2f}")
        print(f"  Reason: {reason}")

        all_reports[label] = {
            "version": entry["version"],
            "step": entry["step"],
            "best_loss": entry["best_loss"],
            "scores": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
                       for k, v in scores.items()},
            "summary_scores": summary_scores,
            "verdict": verdict,
            "reason": reason,
        }

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Final comparison table ──
    print(f"\n\n{'='*90}")
    print(f"  AUDIT SUMMARY")
    print(f"{'='*90}")
    print(f"\n  {'Model':<40} {'Global':>8} {'Constraint':>10} {'Composite':>10}  Verdict")
    print(f"  {'-'*85}")
    best_label, best_score = None, -1
    for label, report in all_reports.items():
        if 'error' in report:
            print(f"  {label[:40]:<40} {'ERROR':>8}")
            continue
        ss = report['summary_scores']
        print(f"  {label[:40]:<40} {ss['global_field']:>8.2f} {ss['constraint']:>10.2f} {ss['composite']:>10.2f}  {report['verdict']}")
        if ss['composite'] > best_score:
            best_score = ss['composite']
            best_label = label

    if best_label:
        best = all_reports[best_label]
        print(f"\n  BEST MODEL: {best_label}")
        print(f"  → Recommended next step: {best['verdict']}")
        print(f"  → {best['reason']}")

    print(f"\n{'='*90}")

    # Save
    path = "./logs/model_audit.json"
    with open(path, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n  Saved: {path}")

    return all_reports


if __name__ == '__main__':
    run_audit()
