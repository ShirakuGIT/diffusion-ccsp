"""
Flow v3: Constraint-Aware Consensus Flow for CCSP
===================================================
Key insight from v2 failure: MSE to ground truth ≈ E[valid solutions],
but E[valid solutions] ∉ valid set (regression bias). The model learns
to predict the *average* feasible pose, which is itself infeasible.

Fix: train with **constraint violation losses** as the primary objective.
The model learns to produce poses that *satisfy constraints*, not just
match ground truth on average.

Loss:
  L = λ_c * Σ_c violation_c(x̂_1) + λ_reg * MSE(x̂_1, x_1)

Architecture: same as v2 (target-prediction composition).
Inference: iterative target refinement x ← x + α*(target - x).

Usage:
    python train_flow_v3.py -input_mode qualitative
    python train_flow_v3.py -input_mode qualitative -lambda_c 10.0 -lambda_reg 1.0
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import (
    qualitative_constraints,
    puzzle_constraints,
    stability_constraints,
    robot_constraints,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_data_config(input_mode):
    if input_mode == 'qualitative':
        train_task = "RandomSplitQualitativeWorld(30000)_qualitative_train"
        test_tasks = {i: f'RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split'
                      for i in range(2, 6)}
        dims = (2, 0, 2), (4, 2, 6)
        constraint_types = qualitative_constraints
    else:
        raise ValueError(f"Only qualitative supported for now: {input_mode}")
    return train_task, test_tasks, dims, constraint_types


def _infer_input_mode(constraint_types):
    ct = set(constraint_types)
    if ct <= set(qualitative_constraints):
        return 'qualitative'
    return 'diffuse_pairwise'


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Constraint Violations
# ─────────────────────────────────────────────────────────────────────────────

def compute_constraint_violations(x_hat, batch, constraint_types, device='cuda'):
    """Compute smooth, differentiable constraint violations for all edges.

    Returns total violation (scalar) and per-type statistics dict.

    All violations are ≥ 0 (0 = satisfied). Uses soft hinge: ReLU(-h + margin).

    Coordinates: x_hat = [x, y, cos, sin] per node.
    Geometry: batch.x[:, 0:2] = [w, h] (half-extents in normalized coords).
    Tray: [0, 2] × [0, 2] in normalized coordinates.
    """
    x = batch.x.to(device)
    edge_index = batch.edge_index.T.to(device)
    edge_attr = batch.edge_attr.to(device)
    geom_end = 2  # qualitative: geom = x[:, 0:2]
    margin = 0.02  # tolerance

    total_violation = torch.tensor(0.0, device=device)
    per_type_violations = {}

    for ei in range(edge_index.shape[0]):
        i_idx = edge_index[ei, 0]
        j_idx = edge_index[ei, 1]
        ctype_idx = int(edge_attr[ei].item())
        if ctype_idx >= len(constraint_types):
            continue
        ctype = constraint_types[ctype_idx]

        # Node poses
        xi, yi = x_hat[i_idx, 0], x_hat[i_idx, 1]
        xj, yj = x_hat[j_idx, 0], x_hat[j_idx, 1]

        # Node geometries (half-extents)
        wi, hi_g = x[i_idx, 0], x[i_idx, 1]
        wj, hj_g = x[j_idx, 0], x[j_idx, 1]

        # Compute violation (≥ 0 means violated, 0 = satisfied)
        violation = torch.tensor(0.0, device=device)

        if ctype == 'in':
            # Object center ± half-extent must be inside [0, 2]
            v_left  = F.relu(wi - xi + margin)
            v_right = F.relu(xi + wi - 2.0 + margin)
            v_bot   = F.relu(hi_g - yi + margin)
            v_top   = F.relu(yi + hi_g - 2.0 + margin)
            violation = v_left + v_right + v_bot + v_top

        elif ctype == 'center-in':
            # Center within a radius of tray center (1,1)
            dist = torch.sqrt((xi - 1.0)**2 + (yi - 1.0)**2 + 1e-8)
            violation = F.relu(dist - 0.5 + margin)

        elif ctype == 'left-in':
            violation = F.relu(xi - 1.0 + margin)  # x < 1.0

        elif ctype == 'right-in':
            violation = F.relu(1.0 - xi + margin)  # x > 1.0

        elif ctype == 'top-in':
            violation = F.relu(1.0 - yi + margin)  # y > 1.0

        elif ctype == 'bottom-in':
            violation = F.relu(yi - 1.0 + margin)  # y < 1.0

        elif ctype == 'cfree':
            # No overlap: max(|dx| - (wi+wj), |dy| - (hi+hj)) > 0
            dx = torch.abs(xi - xj) - (wi + wj)
            dy = torch.abs(yi - yj) - (hi_g + hj_g)
            separation = torch.max(dx, dy)
            violation = F.relu(-separation + margin)

        elif ctype == 'left-of':
            # i is left of j: xj - xi > (wi+wj)/2
            required_gap = (wi + wj) * 0.5
            violation = F.relu(-(xj - xi) + required_gap + margin)

        elif ctype == 'top-of':
            # i is above j: yi - yj > (hi+hj)/2
            required_gap = (hi_g + hj_g) * 0.5
            violation = F.relu(-(yi - yj) + required_gap + margin)

        elif ctype == 'close-to':
            # Distance < threshold
            dist = torch.sqrt((xi - xj)**2 + (yi - yj)**2 + 1e-8)
            threshold = torch.max(wi + wj, hi_g + hj_g) * 1.5
            violation = F.relu(dist - threshold + margin)

        elif ctype == 'away-from':
            # Distance > threshold
            dist = torch.sqrt((xi - xj)**2 + (yi - yj)**2 + 1e-8)
            threshold = torch.max(wi + wj, hi_g + hj_g) * 2.0
            violation = F.relu(-dist + threshold + margin)

        elif ctype == 'h-aligned':
            # Same y: |yi - yj| < threshold
            threshold = (hi_g + hj_g) * 0.3
            violation = F.relu(torch.abs(yi - yj) - threshold)

        elif ctype == 'v-aligned':
            # Same x: |xi - xj| < threshold
            threshold = (wi + wj) * 0.3
            violation = F.relu(torch.abs(xi - xj) - threshold)

        total_violation = total_violation + violation

        # Track per-type stats
        if ctype not in per_type_violations:
            per_type_violations[ctype] = []
        per_type_violations[ctype].append(violation.item())

    return total_violation, per_type_violations


# ─────────────────────────────────────────────────────────────────────────────
# Model (same as v2)
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingCCSP_v3(nn.Module):
    """Constraint-aware consensus flow model.

    Architecture identical to v2. Each per-constraint MLP predicts target
    poses. Composition by averaging. Trained with constraint violation loss.
    """

    def __init__(self, dims, hidden_dim=256, constraint_types=None,
                 normalize=True, device='cuda'):
        super().__init__()

        if constraint_types is None:
            constraint_types = qualitative_constraints

        self.dims             = dims
        self.device           = device
        self.hidden_dim       = hidden_dim
        self.constraint_types = constraint_types
        self.input_mode       = _infer_input_mode(constraint_types)
        self.normalize        = normalize

        geom_dim = dims[0][0]
        pose_dim = dims[-1][0]

        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
        ).to(device)

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
        ).to(device)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        ).to(device)

        self.constraint_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
            ).to(device)
            for _ in constraint_types
        ])

        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, pose_dim),
        ).to(device)

    def _t_tensor(self, t, device):
        t_int = max(0, min(999, int(round(float(t) * 999.0))))
        return torch.tensor([t_int], dtype=torch.long, device=device)

    def _encode(self, x_t, batch):
        device = self.device
        x = batch.x.to(device)
        geom_begin = self.dims[0][1]
        geom_end   = self.dims[0][2]
        geoms_in   = x[:, geom_begin:geom_end]
        geom_emb = self.geom_encoder(geoms_in)
        pose_emb = self.pose_encoder(x_t.to(device))
        edge_index = batch.edge_index.T.to(device)
        return geom_emb, pose_emb, edge_index

    def predict_target(self, x_t, batch, t=1.0):
        """Predict consensus target x̂_1 from current state x_t."""
        import jactorch

        device    = self.device
        t_tensor  = self._t_tensor(t, device)
        n_nodes   = x_t.shape[0]
        pose_dim  = self.dims[-1][0]

        geom_emb, pose_emb, edge_index = self._encode(x_t, batch)

        all_targets = torch.zeros(n_nodes, pose_dim, device=device)
        all_count   = torch.zeros(n_nodes, device=device)

        for i, mlp in enumerate(self.constraint_mlps):
            edge_mask = (batch.edge_attr.to(device) == i)
            if edge_mask.sum() == 0:
                continue
            edges = edge_index[edge_mask]
            src, dst = edges[:, 0], edges[:, 1]

            n_edges = src.shape[0]
            t_emb = self.time_mlp(
                jactorch.add_dim(t_tensor, 0, n_edges)[:, 0]
            )

            inputs = torch.cat([
                geom_emb[src], geom_emb[dst],
                pose_emb[src], pose_emb[dst],
                t_emb,
            ], dim=-1)

            out = mlp(inputs)
            target_src = self.pose_decoder(out[:, :self.hidden_dim])
            target_dst = self.pose_decoder(out[:, self.hidden_dim:])

            all_targets.scatter_add_(0, src.unsqueeze(-1).expand_as(target_src), target_src)
            all_targets.scatter_add_(0, dst.unsqueeze(-1).expand_as(target_dst), target_dst)
            all_count.scatter_add_(0, src, torch.ones(n_edges, device=device))
            all_count.scatter_add_(0, dst, torch.ones(n_edges, device=device))

        denom = all_count.unsqueeze(-1).clamp(min=1)
        if self.normalize:
            x1_hat = all_targets / denom.sqrt()
        else:
            x1_hat = all_targets / denom

        return x1_hat

    def forward(self, x_t, batch, t):
        """Predict velocity via target-prediction composition."""
        device = self.device
        x1_hat = self.predict_target(x_t, batch, t)
        t_float = float(t)
        inv_time = 1.0 / max(1.0 - t_float, 1e-3)
        v = (x1_hat - x_t.to(device)) * inv_time
        mask = batch.mask.bool().to(device)
        v[mask] = 0.0
        return v

    def compute_loss(self, batch, lambda_c=10.0, lambda_reg=1.0, phase='joint'):
        """Constraint-aware loss.

        L = λ_c * constraint_violations(x̂_1)
          + λ_reg * MSE(x̂_1, x_1)

        Args:
            phase: 'warmup' (MSE only), 'joint' (both), 'constraint' (constraint-dominant)
        """
        device = self.device
        batch  = batch.to(device)

        pose_begin = self.dims[-1][1]
        pose_end   = self.dims[-1][2]

        x_1  = batch.x[:, pose_begin:pose_end].clone()
        mask = batch.mask.bool()

        # Sample random noise and timestep
        x_0 = torch.randn_like(x_1)
        t   = torch.rand(1).item()

        x_t = (1.0 - t) * x_0 + t * x_1
        x_t[mask] = x_1[mask]

        # Predict consensus target
        x1_hat = self.predict_target(x_t, batch, t)
        x1_hat[mask] = x_1[mask]  # keep fixed nodes

        free = ~mask
        if free.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # MSE loss (regression anchor)
        mse_loss = F.mse_loss(x1_hat[free], x_1[free])

        if phase == 'warmup':
            return mse_loss

        # Constraint violation loss
        constraint_loss, _ = compute_constraint_violations(
            x1_hat, batch, self.constraint_types, device)

        # Normalize by number of edges
        n_edges = batch.edge_index.shape[1]
        if n_edges > 0:
            constraint_loss = constraint_loss / n_edges

        if phase == 'constraint':
            return lambda_c * constraint_loss + 0.1 * lambda_reg * mse_loss

        # joint
        return lambda_c * constraint_loss + lambda_reg * mse_loss


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_iterative(model, batch, K=20, alpha=0.8, device='cuda'):
    """Iterative target refinement: x ← x + α*(target - x)."""
    model.eval()
    batch = batch.to(device)
    pose_begin = model.dims[-1][1]
    pose_end   = model.dims[-1][2]
    geom_end   = model.dims[0][2]
    x_clean = batch.x[:, pose_begin:pose_end].to(device)
    geoms   = batch.x[:, :geom_end].to(device)
    mask    = batch.mask.bool().to(device)

    x = torch.randn_like(x_clean)
    x[mask] = x_clean[mask]

    for k in range(K):
        target = model.predict_target(x, batch, t=1.0)
        x = x + alpha * (target - x)
        x[mask] = x_clean[mask]
        # Clamp to tray
        for i in range(x.shape[0]):
            if mask[i]: continue
            w = geoms[i, 0].item(); h = geoms[i, 1].item()
            x[i, 0] = x[i, 0].clamp(w + 0.02, 2.0 - w - 0.02)
            x[i, 1] = x[i, 1].clamp(h + 0.02, 2.0 - h - 0.02)

    return x


@torch.no_grad()
def sample_euler(model, batch, n_steps=20, device='cuda'):
    """Standard Euler ODE sampling (for comparison)."""
    model.eval()
    batch = batch.to(device)
    pose_begin = model.dims[-1][1]
    pose_end   = model.dims[-1][2]
    geom_end   = model.dims[0][2]
    x_clean = batch.x[:, pose_begin:pose_end].to(device)
    geoms   = batch.x[:, :geom_end].to(device)
    mask    = batch.mask.bool().to(device)

    x_t = torch.randn_like(x_clean)
    x_t[mask] = x_clean[mask]
    dt = 1.0 / n_steps
    for step in range(n_steps):
        v = model(x_t, batch, step * dt)
        x_t = x_t + v * dt
        x_t[mask] = x_clean[mask]
        for i in range(x_t.shape[0]):
            if mask[i]: continue
            w = geoms[i, 0].item(); h = geoms[i, 1].item()
            x_t[i, 0] = x_t[i, 0].clamp(w + 0.02, 2.0 - w - 0.02)
            x_t[i, 1] = x_t[i, 1].clamp(h + 0.02, 2.0 - h - 0.02)
    return x_t


# ─────────────────────────────────────────────────────────────────────────────
# Fast constraint check
# ─────────────────────────────────────────────────────────────────────────────

def _fast_constraint_check(poses, batch, constraint_types, tol=0.02, device='cuda'):
    x  = batch.x.to(device)
    ei = batch.edge_index.T.to(device)
    ea = batch.edge_attr.to(device)
    for k in range(ei.shape[0]):
        i = ei[k, 0].item(); j = ei[k, 1].item()
        c = int(ea[k].item())
        if c >= len(constraint_types): continue
        ct = constraint_types[c]
        xi, yi = float(poses[i, 0]), float(poses[i, 1])
        xj, yj = float(poses[j, 0]), float(poses[j, 1])
        wi = float(x[i, 0]); hi = float(x[i, 1])
        wj = float(x[j, 0]); hj = float(x[j, 1])
        h = 1.0
        if   ct == 'in':       h = min(xi-wi, 2-(xi+wi), yi-hi, 2-(yi+hi))
        elif ct == 'cfree':    h = max(abs(xi-xj)-(wi+wj), abs(yi-yj)-(hi+hj))
        elif ct == 'left-of':  h = (xj-xi) - (wi+wj)*0.5
        elif ct == 'top-of':   h = (yi-yj) - (hi+hj)*0.5
        elif ct == 'close-to':
            d = ((xi-xj)**2+(yi-yj)**2)**0.5+1e-8
            h = max(wi+wj, hi+hj)*1.5 - d
        elif ct == 'away-from':
            d = ((xi-xj)**2+(yi-yj)**2)**0.5+1e-8
            h = d - max(wi+wj, hi+hj)*2.0
        if h < -tol: return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FlowTrainer_v3:
    def __init__(self, model, train_dataset, test_datasets,
                 lr=5e-4, batch_size=128, train_num_steps=200000,
                 save_every=10000, results_folder='./logs/flow_v3',
                 lambda_c=10.0, lambda_reg=1.0,
                 warmup_steps=5000):
        self.model           = model
        self.device          = model.device
        self.opt             = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.train_num_steps = train_num_steps
        self.save_every      = save_every
        self.results_folder  = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.step            = 0
        self.best_succ       = -1.0
        self.train_losses    = []
        self.lambda_c        = lambda_c
        self.lambda_reg      = lambda_reg
        self.warmup_steps    = warmup_steps

        kw = dict(pin_memory=True, num_workers=0)
        self.train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kw)
        self.test_dls = {
            k: DataLoader(d, batch_size=1, shuffle=False, **kw)
            for k, d in test_datasets.items()
        }

    def _get_phase(self):
        if self.step < self.warmup_steps:
            return 'warmup'
        elif self.step < self.warmup_steps + 50000:
            return 'joint'
        else:
            return 'constraint'

    def save(self, tag):
        path = self.results_folder / f'flow_v3_model_{tag}.pt'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': min(self.train_losses) if self.train_losses else float('inf'),
        }, str(path))
        print(f"  [save] {path}")

    def load(self, tag):
        path = self.results_folder / f'flow_v3_model_{tag}.pt'
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.opt.load_state_dict(ckpt['optimizer_state_dict'])
        self.step = ckpt.get('step', 0)
        self.train_losses = ckpt.get('train_losses', [])
        print(f"  [load] step={self.step}")

    def train(self):
        self.model.train()
        dl_iter  = iter(self.train_dl)
        epoch_losses = []
        t0 = time.time()
        print(f"\n  Training Flow v3 (constraint-aware) for {self.train_num_steps:,} steps …")
        print(f"  λ_c={self.lambda_c}, λ_reg={self.lambda_reg}, warmup={self.warmup_steps}")

        while self.step < self.train_num_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                if epoch_losses:
                    self.train_losses.append(sum(epoch_losses) / len(epoch_losses))
                    epoch_losses = []
                dl_iter = iter(self.train_dl)
                batch   = next(dl_iter)

            phase = self._get_phase()
            self.opt.zero_grad(set_to_none=True)
            loss = self.model.compute_loss(
                batch, lambda_c=self.lambda_c, lambda_reg=self.lambda_reg,
                phase=phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            epoch_losses.append(loss.item())
            self.step += 1

            if self.step % 1000 == 0:
                avg = sum(epoch_losses[-200:]) / min(200, len(epoch_losses))
                print(f"  step {self.step:6d}/{self.train_num_steps} | "
                      f"loss={avg:.5f} | phase={phase} | {(time.time()-t0)/60:.1f}min")

            if self.step % self.save_every == 0:
                epoch_n = self.step // self.save_every
                self.save(epoch_n)
                self._quick_eval()

        if epoch_losses:
            self.train_losses.append(sum(epoch_losses) / len(epoch_losses))
        self.save('final')
        print("\n  Training complete.")

    @torch.no_grad()
    def _quick_eval(self):
        self.model.eval()
        print(f"\n  [eval @ step {self.step}]")
        total_s_iter = total_s_euler = total_n = 0

        for n_obj, dl in sorted(self.test_dls.items()):
            s_iter = s_euler = n = 0
            for batch in dl:
                # Iterative sampling
                poses_iter = sample_iterative(self.model, batch, K=20, alpha=0.8,
                                              device=self.device)
                if _fast_constraint_check(poses_iter, batch, self.model.constraint_types,
                                          device=self.device):
                    s_iter += 1

                # Euler sampling
                poses_euler = sample_euler(self.model, batch, n_steps=20,
                                           device=self.device)
                if _fast_constraint_check(poses_euler, batch, self.model.constraint_types,
                                          device=self.device):
                    s_euler += 1
                n += 1
            total_s_iter += s_iter; total_s_euler += s_euler; total_n += n
            print(f"    {n_obj} obj: iter={s_iter}/{n} ({100*s_iter/n:.1f}%) | "
                  f"euler={s_euler}/{n} ({100*s_euler/n:.1f}%)")

        overall_iter = 100.0 * total_s_iter / total_n if total_n else 0
        overall_euler = 100.0 * total_s_euler / total_n if total_n else 0
        best = max(overall_iter, overall_euler)
        print(f"    Overall: iter={overall_iter:.1f}% | euler={overall_euler:.1f}%")
        if best > self.best_succ:
            self.best_succ = best
            self.save('best')
            print(f"    ↑ New best ({best:.1f}%).")
        self.model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Flow v3 (constraint-aware) for CCSP')
    parser.add_argument('-input_mode',      type=str,   default='qualitative')
    parser.add_argument('-hidden_dim',      type=int,   default=256)
    parser.add_argument('-lr',              type=float, default=5e-4)
    parser.add_argument('-batch_size',      type=int,   default=128)
    parser.add_argument('-train_num_steps', type=int,   default=200000)
    parser.add_argument('-save_every',      type=int,   default=10000)
    parser.add_argument('-results_dir',     type=str,   default=None)
    parser.add_argument('-resume',          type=str,   default=None)
    parser.add_argument('-lambda_c',        type=float, default=10.0)
    parser.add_argument('-lambda_reg',      type=float, default=1.0)
    parser.add_argument('-warmup_steps',    type=int,   default=5000)
    # Optional: warm-start from v2 checkpoint
    parser.add_argument('-init_from_v2',    type=str,   default=None,
                        help='Path to v2 checkpoint for warm-start')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device : {device}")
    print(f"  Mode   : Flow v3 (constraint-aware consensus)")
    print(f"  λ_c={args.lambda_c}, λ_reg={args.lambda_reg}, warmup={args.warmup_steps}")

    train_task, test_tasks, dims, constraint_types = get_data_config(args.input_mode)
    results_dir = args.results_dir or f'./logs/flow_v3_{args.input_mode}_h{args.hidden_dim}'

    ds_kw = dict(input_mode=args.input_mode, pre_transform=pre_transform, visualize=False)
    train_dataset = GraphDataset(train_task, **ds_kw)
    test_datasets = {k: GraphDataset(t, **ds_kw) for k, t in test_tasks.items()
                     if os.path.isdir(f'./data/{t}')}
    print(f"  Train: {len(train_dataset):,}   Tests: {list(test_datasets.keys())}")

    model = FlowMatchingCCSP_v3(dims=dims, hidden_dim=args.hidden_dim,
                                 constraint_types=constraint_types,
                                 normalize=True, device=device).to(device)

    # Optional warm-start from v2
    if args.init_from_v2:
        print(f"  Warm-starting from v2: {args.init_from_v2}")
        v2_ckpt = torch.load(args.init_from_v2, map_location=device)
        model.load_state_dict(v2_ckpt['model_state_dict'])

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_p:,}")

    # Sanity check
    b = next(iter(DataLoader(train_dataset, batch_size=4)))
    loss = model.compute_loss(b, lambda_c=args.lambda_c, lambda_reg=args.lambda_reg)
    print(f"  Sanity loss: {loss.item():.4f}")

    trainer = FlowTrainer_v3(model, train_dataset, test_datasets,
                              lr=args.lr, batch_size=args.batch_size,
                              train_num_steps=args.train_num_steps,
                              save_every=args.save_every,
                              results_folder=results_dir,
                              lambda_c=args.lambda_c,
                              lambda_reg=args.lambda_reg,
                              warmup_steps=args.warmup_steps)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
