"""
Model Audit & Diagnostic Pipeline v2
======================================
Decision-grade audit: fine-tune / partial-reset / retrain.

Tests:
  1. Directional Alignment        — cosine sim of v_pred vs u_t (direction proxy)
  2. Trajectory Consistency       — path error ∫||x_t_pred - x_t_gt||dt + self-consistency
  3. One-step vs Multi-step Gap   — properly scaled, shortcut detection
  4. Constraint Quality           — violation severity, avoidance behavior, recovery
  5. Temporal Coherence           — does velocity evolve smoothly across t?
  6. Monotonic Progress           — d/dt ||x_t - x_1|| < 0 (moving toward goal?)
  7. Stability                    — variance, oscillation, drift over long horizon
  8. Robustness                   — Lipschitz sensitivity estimate

Decision logic (geometry-dominant):
  global_field_score ≥ 0.7 → fine-tune (constraint score decides strategy)
  0.5 ≤ global_field_score < 0.7 → partial-reset
  global_field_score < 0.5 → retrain

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
        {"label": "v1_best (200k OT-CFM)",
         "path": "logs/flow_qualitative_h256/flow_model_best.pt",
         "cls": FlowMatchingCCSP, "version": "v1", "extra": {}},
        {"label": "v2_best (10k target-pred)",
         "path": "logs/flow_v2_qualitative_h256/flow_v2_model_best.pt",
         "cls": load_cls("train_flow_v2.py", "FlowMatchingCCSP_v2"),
         "version": "v2", "extra": {}},
        {"label": "v3_best (10k constraint-viol loss)",
         "path": "logs/flow_v3_qualitative_h256/flow_v3_model_best.pt",
         "cls": load_cls("train_flow_v3.py", "FlowMatchingCCSP_v3"),
         "version": "v3", "extra": {}},
        {"label": "v4_best (10k onestep feasibility)",
         "path": "logs/flow_v4_qualitative_h256/flow_model_best.pt",
         "cls": load_cls("train_flow_v4.py", "FlowMatchingCCSP_v4"),
         "version": "v4", "extra": {"lambda_onestep": 1.0, "lambda_constraint": 0.0}},
    ]

    registry = []
    for c in candidates:
        if not os.path.exists(c["path"]):
            continue
        ck = torch.load(c["path"], map_location="cpu")
        registry.append({**c,
                          "step": ck.get("step", "?"),
                          "best_loss": ck.get("best_loss", "?"),
                          "ckpt": ck})
    return registry


def load_model(entry, dims, constraint_types, device):
    cls = entry["cls"]
    extra = entry.get("extra", {})
    try:
        model = cls(dims=dims, hidden_dim=256, constraint_types=constraint_types,
                    normalize=True, device=device, **extra).to(device)
    except TypeError:
        model = cls(dims=dims, hidden_dim=256, constraint_types=constraint_types,
                    normalize=True, device=device).to(device)
    sd = entry["ckpt"].get("model_state_dict", entry["ckpt"])
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def scene_violation(poses_np, edge_index, edge_attr, geoms_np, constraint_types):
    """Returns (mean_violation_magnitude, n_violated, n_total)."""
    violations = []
    for ei in range(edge_index.shape[0]):
        i, j = int(edge_index[ei, 0]), int(edge_index[ei, 1])
        cidx = int(edge_attr[ei])
        if cidx >= len(constraint_types):
            continue
        h, _, _ = compute_barrier(
            constraint_types[cidx],
            torch.tensor(poses_np[i], dtype=torch.float32),
            torch.tensor(poses_np[j], dtype=torch.float32),
            torch.tensor(geoms_np[i], dtype=torch.float32),
            torch.tensor(geoms_np[j], dtype=torch.float32))
        violations.append(min(h, 0.0))
    if not violations:
        return 0.0, 0, 0
    return (-np.mean(violations),
            sum(1 for v in violations if v < -0.02),
            len(violations))


def min_barrier(poses_np, edge_index, edge_attr, geoms_np, constraint_types):
    """Returns minimum barrier h value across all constraints (most violated)."""
    vals = []
    for ei in range(edge_index.shape[0]):
        i, j = int(edge_index[ei, 0]), int(edge_index[ei, 1])
        cidx = int(edge_attr[ei])
        if cidx >= len(constraint_types):
            continue
        h, _, _ = compute_barrier(
            constraint_types[cidx],
            torch.tensor(poses_np[i], dtype=torch.float32),
            torch.tensor(poses_np[j], dtype=torch.float32),
            torch.tensor(geoms_np[i], dtype=torch.float32),
            torch.tensor(geoms_np[j], dtype=torch.float32))
        vals.append(h)
    return min(vals) if vals else 0.0


@torch.no_grad()
def rollout(model, batch, dims, n_steps, device):
    """Standard Euler rollout. Returns trajectory list [x_0,...,x_1], final x_t."""
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]
    x_1 = batch.x[:, pose_begin:pose_end].to(device)
    geoms = batch.x[:, :geom_end].to(device)
    mask = batch.mask.bool().to(device)

    x_t = torch.randn(x_1.shape[0], pose_dim, device=device)
    x_t[mask] = x_1[mask]

    dt = 1.0 / n_steps
    traj = [x_t.clone()]
    for step in range(n_steps):
        v = model(x_t, batch, step * dt)
        x_t = x_t + v * dt
        x_t[mask] = x_1[mask]
        x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
        traj.append(x_t.clone())
    return traj, x_t


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Directional Alignment
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_directional_alignment(model, loader, dims, n_scenes=30, device='cuda'):
    """Cosine similarity of v_pred vs true OT velocity u_t = x_1 - x_0.
    Note: measures direction proxy only, not full transport. See test_trajectory_consistency."""
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    cos_sims = []
    n = 0
    for data in loader:
        if n >= n_scenes: break
        n += 1
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end]
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            torch.manual_seed(n * 100 + int(t_val * 10))
            x_0 = torch.randn_like(x_1)
            x_0[mask] = x_1[mask]
            x_t = (1.0 - t_val) * x_0 + t_val * x_1
            x_t[mask] = x_1[mask]
            u_t = x_1 - x_0
            v_pred = model(x_t, batch, t_val)
            sim = F.cosine_similarity(v_pred[free].reshape(-1),
                                       u_t[free].reshape(-1), dim=0).item()
            cos_sims.append(sim)

    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
    score = float(np.clip((mean_cos + 1.0) / 2.0, 0, 1))
    return {"mean_cosine": mean_cos, "score": score,
            "note": "Direction proxy only — see trajectory_consistency for full transport test",
            "interpretation": (
                "HIGH directional alignment" if mean_cos > 0.6
                else "MODERATE" if mean_cos > 0.2
                else "LOW — may indicate non-OT transport (not necessarily bad)")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Trajectory Consistency (NEW — fixes Gap A)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_trajectory_consistency(model, loader, dims, n_scenes=30, n_steps=20, device='cuda'):
    """
    Two sub-tests:
    (A) Path error: integrate forward, compare intermediate x_t to GT interpolant x_t = (1-t)*x_0 + t*x_1.
        path_error = mean_t ||x_t_pred - x_t_gt|| (on free nodes)
        Low = follows correct transport path.

    (B) Self-consistency: run forward 20 steps, then reverse 20 steps (negate velocity).
        self_err = ||x_reversed - x_0||
        Low = reversible, coherent field.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    path_errors, self_errors = [], []
    n = 0
    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 31)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        x_0 = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_0[mask] = x_1[mask]

        dt = 1.0 / n_steps
        x_t = x_0.clone()
        step_errs = []

        # (A) Forward integration vs GT interpolant
        for step in range(n_steps):
            t = step * dt
            t_next = (step + 1) * dt
            v = model(x_t, batch, t)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

            # GT at t_next
            x_gt = (1.0 - t_next) * x_0 + t_next * x_1
            err = (x_t[free] - x_gt[free]).norm().item() / free.sum().item()
            step_errs.append(err)

        path_errors.append(np.mean(step_errs))

        # (B) Self-consistency: reverse integration
        x_rev = x_t.clone()
        for step in range(n_steps):
            t = 1.0 - step * dt
            v = model(x_rev, batch, t)
            x_rev = x_rev - v * dt  # reverse
            x_rev[mask] = x_0[mask]
            x_rev = clamp_to_tray(x_rev, geoms, mask, pose_dim)

        self_err = (x_rev[free] - x_0[free]).norm().item() / free.sum().item()
        self_errors.append(self_err)

    mean_path = float(np.mean(path_errors))
    mean_self = float(np.mean(self_errors))

    # Score: path_error=0 → 1.0. Expected ~0.5-1.0 for curved transport.
    # self_error: completely reversible → 0, irreversible → large
    path_score = float(np.clip(1.0 - mean_path / 1.5, 0, 1))
    self_score = float(np.clip(1.0 - mean_self / 2.0, 0, 1))
    score = 0.6 * path_score + 0.4 * self_score

    return {
        "path_error": mean_path, "self_consistency_error": mean_self,
        "path_score": path_score, "self_score": self_score, "score": score,
        "interpretation": (
            "COHERENT transport" if score > 0.65
            else "MODERATE coherence" if score > 0.4
            else "INCOHERENT — non-reversible or wrong path")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: One-step vs Multi-step Gap (fixed scaling — Gap B)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_rollout_gap(model, loader, dims, n_scenes=30, n_steps=20, device='cuda'):
    """
    Properly scaled one-step vs multi-step comparison.

    one_step: integrate with dt=1/n_steps for 1 step at t=0, then scale to t=1
              (equivalent: run n_steps of constant velocity = v(x_0,0))
    rollout:  full n_steps Euler integration

    gap = rollout_err - one_step_err
    Small gap → globally consistent field (velocity field changes correctly with t).
    Large gap → shortcut: model only predicts well at t=0 (local).
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    one_errs, roll_errs = [], []
    n = 0
    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 7)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        x_0 = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_0[mask] = x_1[mask]

        # One-step: use v(x_0, t=0) scaled by dt, repeat n_steps times
        # This is equivalent to constant-velocity integration (no field update)
        v0 = model(x_0, batch, 0.0)
        dt = 1.0 / n_steps
        x_one = x_0.clone()
        for _ in range(n_steps):
            x_one = x_one + v0 * dt  # same velocity, no recompute
            x_one[mask] = x_1[mask]
            x_one = clamp_to_tray(x_one, geoms, mask, pose_dim)
        err_one = (x_one[free] - x_1[free]).norm().item() / free.sum().item()
        one_errs.append(err_one)

        # Multi-step: full Euler with velocity recomputed at each step
        x_t = x_0.clone()
        for step in range(n_steps):
            v = model(x_t, batch, step * dt)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
        err_roll = (x_t[free] - x_1[free]).norm().item() / free.sum().item()
        roll_errs.append(err_roll)

    mean_one = float(np.mean(one_errs))
    mean_roll = float(np.mean(roll_errs))
    gap = mean_roll - mean_one

    # Negative gap = rollout actually better than one-step = great sign
    score = float(np.clip(1.0 - max(gap, 0) / 0.5, 0, 1))
    return {
        "one_step_err": mean_one, "rollout_err": mean_roll, "gap": gap,
        "score": score,
        "interpretation": (
            "CONSISTENT: rollout ≤ one-step (velocity field improves over t)" if gap <= 0
            else "SMALL GAP: minor shortcutting" if gap < 0.1
            else "MODERATE GAP" if gap < 0.3
            else "LARGE GAP: shortcut/broken field")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Constraint Quality (extended — Gap C)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_constraint_quality(model, loader, dims, n_scenes=30, n_steps=20, device='cuda'):
    """
    (A) Rollout violation severity and rate.
    (B) Avoidance: during rollout, track minimum barrier h_min over trajectory.
        If h_min > h_final: field avoided the boundary (good).
        If h_min ≈ h_final: field drifted into violation without course-correction.
    (C) Recovery: perturb valid pose into invalid, run 5 steps, measure improvement.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    viol_sevs, viol_rates = [], []
    avoidance_scores = []
    recovery_deltas = []
    n = 0

    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 13)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask

        ei_np = batch.edge_index.T.cpu().numpy()
        ea_np = batch.edge_attr.cpu().numpy()
        gn_np = geoms.cpu().numpy()

        # (A + B) Rollout + avoidance tracking
        x_t = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_t[mask] = x_1[mask]
        dt = 1.0 / n_steps

        min_h_traj = float('inf')  # most-satisfied barrier minimum during trajectory
        for step in range(n_steps):
            v = model(x_t, batch, step * dt)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)

            h_min = min_barrier(x_t.cpu().numpy(), ei_np, ea_np, gn_np, qualitative_constraints)
            if h_min < min_h_traj:
                min_h_traj = h_min

        sev, n_viol, n_total = scene_violation(
            x_t.cpu().numpy(), ei_np, ea_np, gn_np, qualitative_constraints)
        h_final = min_barrier(x_t.cpu().numpy(), ei_np, ea_np, gn_np, qualitative_constraints)

        viol_sevs.append(sev)
        viol_rates.append(n_viol / max(n_total, 1))

        # Avoidance: did the field ever go deeper into violation than where it ended?
        # If min_h_traj ≈ h_final, trajectory never recovered from its worst state.
        # If min_h_traj < h_final (final is less violated than worst), field recovered.
        avoidance = h_final - min_h_traj  # positive = ended better than worst
        avoidance_scores.append(avoidance)

        # (C) Recovery: perturb into invalid region, run 5 steps at t=0.8
        x_perturbed = x_1.clone()
        if free.sum() > 0:
            x_perturbed[free] += torch.randn_like(x_perturbed[free]) * 0.5
        x_perturbed = clamp_to_tray(x_perturbed, geoms, mask, pose_dim)

        sev_before, _, _ = scene_violation(
            x_perturbed.cpu().numpy(), ei_np, ea_np, gn_np, qualitative_constraints)

        x_rec = x_perturbed.clone()
        dt_rec = 1.0 / 5
        for step in range(5):
            t = 0.7 + step * dt_rec * 0.3
            v = model(x_rec, batch, t)
            x_rec = x_rec + v * dt_rec
            x_rec[mask] = x_1[mask]
            x_rec = clamp_to_tray(x_rec, geoms, mask, pose_dim)

        sev_after, _, _ = scene_violation(
            x_rec.cpu().numpy(), ei_np, ea_np, gn_np, qualitative_constraints)
        recovery_deltas.append(sev_before - sev_after)

    mean_sev = float(np.mean(viol_sevs))
    mean_rate = float(np.mean(viol_rates))
    mean_avoid = float(np.mean(avoidance_scores))
    mean_rec = float(np.mean(recovery_deltas))

    sev_score = float(np.clip(1.0 - mean_sev / 0.3, 0, 1))
    avoid_score = float(np.clip((mean_avoid + 0.1) / 0.3, 0, 1))
    rec_score = float(np.clip((mean_rec + 0.05) / 0.15, 0, 1))
    score = 0.4 * sev_score + 0.3 * avoid_score + 0.3 * rec_score

    behavior = ("AVOIDS violations" if mean_avoid > 0.05 and mean_rec > 0.01
                else "RECOVERS but doesn't avoid" if mean_rec > 0.01
                else "DRIFTS into violation")

    return {
        "violation_severity": mean_sev, "violation_rate": mean_rate,
        "avoidance_delta": mean_avoid, "recovery_delta": mean_rec,
        "score": score,
        "behavior": behavior,
        "interpretation": (
            "STRONG: avoids + recovers" if score > 0.65
            else "PARTIAL: learnable via fine-tune" if score > 0.35
            else "WEAK: structural constraint failure")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Temporal Coherence (NEW — Gap E)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_temporal_coherence(model, loader, dims, n_scenes=30, device='cuda'):
    """
    Does velocity evolve smoothly across t for the same x?
    Incoherent field: v(x, t=0.5) completely different from v(x, t=0.51).

    Measure: ||v(x, t+ε) - v(x, t)|| / ε  (temporal Lipschitz constant)
    Low = temporally smooth, globally consistent.
    High = model treats each t independently (shortcut).
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    pose_dim = dims[-1][0]

    temporal_jumps = []
    n = 0
    eps = 0.05

    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 41)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        x_0 = torch.randn_like(x_1)
        x_0[mask] = x_1[mask]

        for t_val in [0.2, 0.4, 0.6, 0.8]:
            x_t = (1.0 - t_val) * x_0 + t_val * x_1
            x_t[mask] = x_1[mask]

            v_t = model(x_t, batch, t_val)
            v_t_eps = model(x_t, batch, min(t_val + eps, 1.0))

            dv = (v_t_eps[free] - v_t[free]).norm().item()
            norm_v = max(v_t[free].norm().item(), 1e-8)
            temporal_jumps.append(dv / (norm_v * eps))

    mean_jump = float(np.mean(temporal_jumps)) if temporal_jumps else 0.0
    # Low jump (< 2) = temporally smooth; high (> 10) = incoherent
    score = float(np.clip(1.0 - (mean_jump - 1.0) / 15.0, 0, 1))

    return {
        "temporal_lipschitz": mean_jump, "score": score,
        "interpretation": (
            "SMOOTH: temporally coherent field" if mean_jump < 3.0
            else "MODERATE temporal variation" if mean_jump < 8.0
            else "INCOHERENT: velocity jumps across t (shortcut learning)")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Monotonic Progress (NEW — Gap D / Energy)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_monotonic_progress(model, loader, dims, n_scenes=30, n_steps=20, device='cuda'):
    """
    Does ||x_t - x_1|| decrease monotonically over the rollout?

    d_t = ||x_t - x_1||[free]
    monotone_frac: fraction of steps where d_t < d_{t-1}
    mean_slope: average (d_T - d_0) / d_0  (negative = converging, positive = diverging)

    A model that makes genuine progress toward the data manifold should
    have monotone_frac > 0.7 and negative mean_slope.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    monotone_fracs, slopes = [], []
    n = 0
    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 53)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        x_t = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_t[mask] = x_1[mask]
        dt = 1.0 / n_steps

        dists = [(x_t[free] - x_1[free]).norm().item()]
        for step in range(n_steps):
            v = model(x_t, batch, step * dt)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
            dists.append((x_t[free] - x_1[free]).norm().item())

        # Fraction of steps where distance decreased
        decreasing = sum(1 for a, b in zip(dists[:-1], dists[1:]) if b < a)
        monotone_fracs.append(decreasing / n_steps)

        # Slope: relative change from start to end
        if dists[0] > 1e-8:
            slopes.append((dists[-1] - dists[0]) / dists[0])

    mean_mono = float(np.mean(monotone_fracs))
    mean_slope = float(np.mean(slopes)) if slopes else 0.0

    # Score: mono=1 + slope=-1 → 1.0; mono=0.5 + slope=0 → ~0.5
    mono_score = float(np.clip(mean_mono, 0, 1))
    slope_score = float(np.clip((-mean_slope + 1.0) / 2.0, 0, 1))
    score = 0.6 * mono_score + 0.4 * slope_score

    return {
        "monotone_fraction": mean_mono, "mean_slope": mean_slope,
        "score": score,
        "interpretation": (
            "CONVERGING: steady progress toward goal" if mean_mono > 0.7 and mean_slope < 0
            else "PARTIAL progress" if mean_mono > 0.55
            else "NON-MONOTONE: oscillating or diverging")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Stability (enhanced — Gap D)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_stability(model, loader, dims, n_scenes=30, n_steps=40, device='cuda'):
    """
    Long-horizon integration (40 steps, 2x normal).
    Checks: explosion (norm > 10), oscillation (high trajectory variance), drift.
    """
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    geom_end = dims[0][2]
    pose_dim = dims[-1][0]

    stable_fracs, traj_variances, final_norms = [], [], []
    n = 0
    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 17)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        geoms = batch.x[:, :geom_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        x_t = torch.randn(x_1.shape[0], pose_dim, device=device)
        x_t[mask] = x_1[mask]
        dt = 1.0 / n_steps

        norms_over_time = []
        for step in range(n_steps):
            v = model(x_t, batch, step * dt)
            x_t = x_t + v * dt
            x_t[mask] = x_1[mask]
            x_t = clamp_to_tray(x_t, geoms, mask, pose_dim)
            norms_over_time.append(x_t[free].norm().item() / free.sum().item())

        final_norm = norms_over_time[-1]
        final_norms.append(final_norm)
        stable_fracs.append(1.0 if final_norm < 5.0 else 0.0)

        # Variance in the second half (should be low if converged)
        second_half = norms_over_time[n_steps // 2:]
        traj_variances.append(float(np.var(second_half)))

    mean_final = float(np.mean(final_norms))
    stable_frac = float(np.mean(stable_fracs))
    mean_var = float(np.mean(traj_variances))

    var_score = float(np.clip(1.0 - mean_var / 0.1, 0, 1))
    score = 0.5 * stable_frac + 0.5 * var_score

    return {
        "stable_fraction": stable_frac, "trajectory_variance": mean_var,
        "mean_final_norm": mean_final, "score": score,
        "interpretation": (
            "STABLE: converged trajectories" if score > 0.8
            else "OSCILLATING: high variance in tail" if mean_var > 0.05
            else "MODERATE stability")}


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Robustness (Lipschitz)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_robustness(model, loader, dims, n_scenes=30, device='cuda'):
    """Input noise sensitivity: ||Δv|| / ||Δx||."""
    pose_begin, pose_end = dims[-1][1], dims[-1][2]
    pose_dim = dims[-1][0]
    sensitivities = []
    n = 0
    for data in loader:
        if n >= n_scenes: break
        n += 1
        torch.manual_seed(n * 23)
        batch = data.to(device)
        x_1 = batch.x[:, pose_begin:pose_end].to(device)
        mask = batch.mask.bool().to(device)
        free = ~mask
        if free.sum() == 0: continue

        x_0 = torch.randn_like(x_1)
        x_0[mask] = x_1[mask]

        for t_val in [0.3, 0.7]:
            x_t = (1.0 - t_val) * x_0 + t_val * x_1
            x_t[mask] = x_1[mask]
            v_clean = model(x_t, batch, t_val)

            noise = torch.randn_like(x_t) * 0.05
            noise[mask] = 0.0
            v_noisy = model(x_t + noise, batch, t_val)

            dv = (v_noisy[free] - v_clean[free]).norm().item()
            dx = noise[free].norm().item()
            if dx > 1e-8:
                sensitivities.append(dv / dx)

    mean_sens = float(np.mean(sensitivities)) if sensitivities else 0.0
    score = float(np.clip(1.0 - (mean_sens - 1.0) / 10.0, 0, 1))
    return {
        "lipschitz_estimate": mean_sens, "score": score,
        "interpretation": (
            "SMOOTH: well-behaved Lipschitz" if mean_sens < 3.0
            else "MODERATE sensitivity" if mean_sens < 8.0
            else "HIGH sensitivity: noisy/unstable field")}


# ─────────────────────────────────────────────────────────────────────────────
# Decision Logic (geometry-dominant — fixes Gap C in decision logic)
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(scores):
    """
    Geometry dominates: determines retrain vs not.
    Constraints determine fine-tune strategy.
    """
    da   = scores['directional_alignment']['score']
    tc   = scores['trajectory_consistency']['score']
    rg   = scores['rollout_gap']['score']
    temp = scores['temporal_coherence']['score']
    mono = scores['monotonic_progress']['score']
    st   = scores['stability']['score']
    rob  = scores['robustness']['score']
    cq   = scores['constraint_quality']['score']

    # Global field: transport correctness (trajectory + consistency + coherence + mono)
    # Robustness and stability are sanity checks
    global_field_score = (tc * 0.30 + rg * 0.20 + temp * 0.20 +
                          mono * 0.15 + da * 0.10 + st * 0.05)
    constraint_score = cq

    if global_field_score >= 0.7:
        if constraint_score >= 0.5:
            verdict = "FINE-TUNE"
            strategy = "Add more constraint loss weight or feasibility loss."
        else:
            verdict = "FINE-TUNE (constraint-focused)"
            strategy = ("Strong geometry but weak constraints. "
                        "Fine-tune with λ_c * violation_loss + optional λ_onestep.")
    elif global_field_score >= 0.5:
        verdict = "PARTIAL-RESET"
        strategy = ("Moderate geometry — representation has useful structure. "
                    "Reinitialize pose_decoder + constraint_mlps, retrain with "
                    "CFM + feasibility losses from this checkpoint.")
    else:
        verdict = "RETRAIN"
        strategy = ("Weak global field — fine-tuning cannot recover correct transport. "
                    "Retrain from scratch with constraint-aware objective (v3-style).")

    summary = {
        "global_field": round(global_field_score, 3),
        "constraint": round(constraint_score, 3),
        "composite": round(0.6 * global_field_score + 0.4 * constraint_score, 3),
        "component_scores": {
            "directional_alignment": round(da, 3),
            "trajectory_consistency": round(tc, 3),
            "rollout_gap": round(rg, 3),
            "temporal_coherence": round(temp, 3),
            "monotonic_progress": round(mono, 3),
        }
    }
    return verdict, strategy, summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_audit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"\n{'#'*70}")
    print(f"# MODEL AUDIT v2 — Flow-CCSP Checkpoint Registry")
    print(f"# Device: {device} ({gpu_name})")
    print(f"{'#'*70}")

    _, _, dims, constraint_types = get_data_config('qualitative')

    test_task = "RandomSplitQualitativeWorld(100)_qualitative_test_3_split"
    dataset = GraphDataset(test_task, input_mode='qualitative',
                           pre_transform=pre_transform, visualize=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"\n  Test set: {len(dataset)} scenes (3-obj), using first 30")

    registry = build_registry()
    print(f"\n  Found {len(registry)} models:")
    for r in registry:
        print(f"    [{r['version']}] {r['label']}  step={r['step']}  loss={r['best_loss']}")

    all_reports = {}

    TESTS = [
        ("directional_alignment", "Directional Alignment",
         lambda m: test_directional_alignment(m, loader, dims, device=device)),
        ("trajectory_consistency", "Trajectory Consistency",
         lambda m: test_trajectory_consistency(m, loader, dims, device=device)),
        ("rollout_gap", "Rollout Gap",
         lambda m: test_rollout_gap(m, loader, dims, device=device)),
        ("constraint_quality", "Constraint Quality",
         lambda m: test_constraint_quality(m, loader, dims, device=device)),
        ("temporal_coherence", "Temporal Coherence",
         lambda m: test_temporal_coherence(m, loader, dims, device=device)),
        ("monotonic_progress", "Monotonic Progress",
         lambda m: test_monotonic_progress(m, loader, dims, device=device)),
        ("stability", "Stability (40 steps)",
         lambda m: test_stability(m, loader, dims, device=device)),
        ("robustness", "Robustness",
         lambda m: test_robustness(m, loader, dims, device=device)),
    ]

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
        for i, (key, name, fn) in enumerate(TESTS):
            print(f"  [{i+1}/{len(TESTS)}] {name}...", end=" ", flush=True)
            r = fn(model)
            scores[key] = r
            # Print key metric + score
            interp = r.get('interpretation', '')
            sc = r.get('score', 0)
            # Pick the most informative metric to show
            metric_val = (r.get('mean_cosine') or r.get('path_error') or
                          r.get('gap') or r.get('violation_severity') or
                          r.get('temporal_lipschitz') or r.get('monotone_fraction') or
                          r.get('stable_fraction') or r.get('lipschitz_estimate') or 0)
            print(f"score={sc:.2f}  val={metric_val:.3f}  | {interp}")

        verdict, strategy, summary = make_decision(scores)

        print(f"\n  ┌─ VERDICT: {verdict}")
        print(f"  │  Global Field: {summary['global_field']:.3f}  "
              f"Constraint: {summary['constraint']:.3f}  "
              f"Composite: {summary['composite']:.3f}")
        print(f"  │  Components: "
              + "  ".join(f"{k.split('_')[0]}={v:.2f}"
                           for k, v in summary['component_scores'].items()))
        print(f"  └─ Strategy: {strategy}")

        all_reports[label] = {
            "version": entry["version"],
            "step": str(entry["step"]),
            "best_loss": str(entry["best_loss"]),
            "scores": {k: {kk: vv for kk, vv in v.items()
                           if isinstance(vv, (int, float, str))}
                       for k, v in scores.items()},
            "summary": summary,
            "verdict": verdict,
            "strategy": strategy,
        }

        del model
        torch.cuda.empty_cache()

    # ── Final comparison ──
    print(f"\n\n{'='*95}")
    print(f"  FINAL AUDIT SUMMARY")
    print(f"{'='*95}")
    print(f"\n  {'Model':<38} {'Global':>7} {'Const':>6} {'Comp':>6}  {'Verdict'}")
    print(f"  {'-'*88}")

    best_label, best_score = None, -1
    for lbl, rep in all_reports.items():
        if 'error' in rep: continue
        ss = rep['summary']
        print(f"  {lbl[:38]:<38} {ss['global_field']:>7.3f} {ss['constraint']:>6.3f} "
              f"{ss['composite']:>6.3f}  {rep['verdict']}")
        if ss['composite'] > best_score:
            best_score = ss['composite']
            best_label = lbl

    if best_label:
        best = all_reports[best_label]
        print(f"\n  ★ BEST MODEL: {best_label}")
        print(f"    Verdict:  {best['verdict']}")
        print(f"    Strategy: {best['strategy']}")

    print(f"\n{'='*95}")

    path = "./logs/model_audit_v2.json"
    with open(path, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n  Saved: {path}")
    return all_reports


if __name__ == '__main__':
    run_audit()
