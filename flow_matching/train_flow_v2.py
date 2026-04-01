"""
Flow Matching v2: Target-Prediction Composition for CCSP
=========================================================
Key insight: the original flow model composes per-constraint *velocities*
by summing. But velocities from different constraints are not mutually
consistent — each v_c points toward its own target, and their sum does
not generally yield a valid velocity for the joint distribution.

Fix: each per-constraint MLP predicts a *target pose* x̂_1, not a velocity.
We compose by averaging targets (exactly as diffusion composes noise/targets),
then derive velocity from the consensus target:

    v = (x̂_1_avg - x_t) / (1 - t + eps)

This ensures the composed velocity always points toward a single consensus
goal, avoiding the conflicting-velocities failure mode.

Training objective (same as before):
  x_t = (1-t)*x_0 + t*x_1      (OT interpolant)
  u_t = x_1 - x_0               (conditional velocity target)
  v   = (x̂_1_avg - x_t) / (1 - t + eps)
  L   = MSE(v, u_t)             (free nodes only)

Architecture: identical to v1 (same encoders, MLPs, decoder).
Only the forward/loss semantics change.

Usage:
    python train_flow_v2.py -input_mode qualitative
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

from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import (
    qualitative_constraints,
    puzzle_constraints,
    stability_constraints,
    robot_constraints,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data configuration (reuse from v1)
# ─────────────────────────────────────────────────────────────────────────────

def get_data_config(input_mode):
    """Return (train_task, test_tasks, dims, constraint_types)."""
    if input_mode == 'qualitative':
        train_task = "RandomSplitQualitativeWorld(30000)_qualitative_train"
        test_tasks = {i: f'RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split'
                      for i in range(2, 6)}
        dims = (2, 0, 2), (4, 2, 6)
        constraint_types = qualitative_constraints
    elif input_mode == 'diffuse_pairwise':
        train_task = "RandomSplitWorld(30000)_diffuse_pairwise_train"
        test_tasks = {i: f'RandomSplitWorld(100)_diffuse_pairwise_test_{i}_split'
                      for i in range(2, 6)}
        dims = ((2, 0, 2), (2, 2, 4))
        constraint_types = puzzle_constraints
    elif input_mode == 'stability_flat':
        train_task = "RandomSplitWorld(24000)_stability_flat_train"
        test_tasks = {i: f'RandomSplitWorld(10)_stability_flat_test_{i}_object'
                      for i in range(4, 7)}
        dims = ((2, 0, 2), (4, 2, 6))
        constraint_types = stability_constraints
    elif input_mode == 'robot_box':
        train_task = "TableToBoxWorld(10000)_train"
        test_tasks = {i: f"TableToBoxWorld(10)_test_{i}_object" for i in range(2, 7)}
        dims = ((8, 0, 8), (5, 13, 18))
        constraint_types = robot_constraints
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
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
# Flow Matching v2: Target-Prediction Composition
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingCCSP_v2(nn.Module):
    """Flow Matching model that composes TARGET PREDICTIONS, not velocities.

    Each per-constraint MLP predicts x̂_1 (the target pose for each node).
    The composed target is averaged across constraints, then velocity is
    derived as v = (x̂_1_avg - x_t) / (1 - t + eps).

    This mirrors how diffusion composes: averaging denoised targets, not
    combining raw score functions.
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

        # Geometry encoder
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
        ).to(device)

        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
        ).to(device)

        # Time MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        ).to(device)

        # Per-constraint 2-layer MLPs (same architecture as v1)
        self.constraint_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
            ).to(device)
            for _ in constraint_types
        ])

        # Pose decoder: hidden → pose_dim (predicts TARGET POSE, not velocity)
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

    def forward(self, x_t, batch, t):
        """Predict velocity via target-prediction composition.

        1. Each constraint MLP predicts target pose x̂_1 for each node.
        2. Compose by averaging: x̂_1_avg = Σ_c x̂_1_c / sqrt(count)
        3. Derive velocity: v = (x̂_1_avg - x_t) / (1 - t + eps)

        Returns: v [N, pose_dim]
        """
        import jactorch

        device    = self.device
        t_tensor  = self._t_tensor(t, device)
        n_nodes   = x_t.shape[0]
        pose_dim  = self.dims[-1][0]

        geom_emb, pose_emb, edge_index = self._encode(x_t, batch)

        # Accumulate TARGET predictions (not velocities)
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

            # Decode per-node TARGET POSE (not velocity!)
            target_src = self.pose_decoder(out[:, :self.hidden_dim])
            target_dst = self.pose_decoder(out[:, self.hidden_dim:])

            # Scatter-add into target accumulator
            all_targets.scatter_add_(0, src.unsqueeze(-1).expand_as(target_src), target_src)
            all_targets.scatter_add_(0, dst.unsqueeze(-1).expand_as(target_dst), target_dst)
            all_count.scatter_add_(0, src, torch.ones(n_edges, device=device))
            all_count.scatter_add_(0, dst, torch.ones(n_edges, device=device))

        # Average targets (same normalization as diffusion)
        denom = all_count.unsqueeze(-1).clamp(min=1)
        if self.normalize:
            x1_hat = all_targets / denom.sqrt()
        else:
            x1_hat = all_targets / denom

        # Derive velocity from consensus target
        # v = (x̂_1 - x_t) / (1 - t)
        # At t → 1, the velocity would blow up, so clamp (1-t) away from 0
        t_float = float(t)
        inv_time = 1.0 / max(1.0 - t_float, 1e-3)
        v = (x1_hat - x_t.to(device)) * inv_time

        # Fixed nodes: zero velocity
        mask = batch.mask.bool().to(device)
        v[mask] = 0.0

        return v

    def predict_target(self, x_t, batch, t):
        """Return the raw consensus target x̂_1 (useful for debugging)."""
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

    def compute_loss(self, batch):
        """Target-prediction loss: MSE(x̂_1, x_1).

        Instead of training on derived velocity (which blows up as t→1),
        we train directly on the target prediction. This is equivalent to
        how diffusion trains on noise/x_0 prediction.

        During sampling, velocity is derived as v = (x̂_1 - x_t) / (1-t).
        """
        device = self.device
        batch  = batch.to(device)

        pose_begin = self.dims[-1][1]
        pose_end   = self.dims[-1][2]

        x_1  = batch.x[:, pose_begin:pose_end].clone()
        mask = batch.mask.bool()

        x_0  = torch.randn_like(x_1)
        t    = torch.rand(1).item()

        x_t  = (1.0 - t) * x_0 + t * x_1
        x_t[mask] = x_1[mask]

        # Predict consensus target (not velocity)
        x1_hat = self.predict_target(x_t, batch, t)

        free = ~mask
        if free.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return F.mse_loss(x1_hat[free], x_1[free])


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_flow_v2(model, batch, n_steps=20, device='cuda'):
    """Euler ODE integration with target-prediction model."""
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
        # Hard clamp to tray
        for i in range(x_t.shape[0]):
            if mask[i]: continue
            w = geoms[i, 0].item(); h = geoms[i, 1].item()
            x_t[i, 0] = x_t[i, 0].clamp(w + 0.02, 2.0 - w - 0.02)
            x_t[i, 1] = x_t[i, 1].clamp(h + 0.02, 2.0 - h - 0.02)
    return x_t


# ─────────────────────────────────────────────────────────────────────────────
# Fast constraint check (same as v1)
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

class FlowTrainer_v2:
    def __init__(self, model, train_dataset, test_datasets,
                 lr=5e-4, batch_size=128, train_num_steps=200000,
                 save_every=10000, results_folder='./logs/flow_v2'):
        self.model          = model
        self.device         = model.device
        self.opt            = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.train_num_steps = train_num_steps
        self.save_every     = save_every
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.step           = 0
        self.best_succ      = -1.0
        self.train_losses   = []

        kw = dict(pin_memory=True, num_workers=0)
        self.train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kw)
        self.test_dls = {
            k: DataLoader(d, batch_size=1, shuffle=False, **kw)
            for k, d in test_datasets.items()
        }

    def save(self, tag):
        path = self.results_folder / f'flow_v2_model_{tag}.pt'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': min(self.train_losses) if self.train_losses else float('inf'),
        }, str(path))
        print(f"  [save] {path}")

    def load(self, tag):
        path = self.results_folder / f'flow_v2_model_{tag}.pt'
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
        t0       = time.time()
        print(f"\n  Training Flow v2 (target-prediction) for {self.train_num_steps:,} steps …")

        while self.step < self.train_num_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                # End of epoch — record average loss
                if epoch_losses:
                    avg_epoch = sum(epoch_losses) / len(epoch_losses)
                    self.train_losses.append(avg_epoch)
                    epoch_losses = []
                dl_iter = iter(self.train_dl)
                batch   = next(dl_iter)

            self.opt.zero_grad(set_to_none=True)
            loss = self.model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            epoch_losses.append(loss.item())
            self.step += 1

            if self.step % 1000 == 0:
                avg = sum(epoch_losses[-200:]) / min(200, len(epoch_losses))
                print(f"  step {self.step:6d}/{self.train_num_steps} | "
                      f"loss={avg:.5f} | {(time.time()-t0)/60:.1f}min")

            if self.step % self.save_every == 0:
                epoch_n = self.step // self.save_every
                self.save(epoch_n)
                self._quick_eval()

        # Save final epoch losses
        if epoch_losses:
            self.train_losses.append(sum(epoch_losses) / len(epoch_losses))
        self.save('final')
        print("\n  Training complete.")

    @torch.no_grad()
    def _quick_eval(self):
        self.model.eval()
        print(f"\n  [eval @ step {self.step}]")
        total_s = total_n = 0

        for n_obj, dl in sorted(self.test_dls.items()):
            s = n = 0
            for batch in dl:
                poses = sample_flow_v2(self.model, batch, n_steps=20,
                                       device=self.device)
                if _fast_constraint_check(poses, batch, self.model.constraint_types,
                                          device=self.device):
                    s += 1
                n += 1
            rate = 100.0 * s / n if n else 0
            total_s += s; total_n += n
            print(f"    {n_obj} obj: {s}/{n}  ({rate:.1f}%)")

        overall = 100.0 * total_s / total_n if total_n else 0
        print(f"    Overall: {overall:.1f}%")
        if overall > self.best_succ:
            self.best_succ = overall
            self.save('best')
            print(f"    ↑ New best.")
        self.model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Flow v2 (target-prediction) for CCSP')
    parser.add_argument('-input_mode',      type=str,   default='qualitative',
                        choices=['qualitative','diffuse_pairwise','stability_flat','robot_box'])
    parser.add_argument('-hidden_dim',      type=int,   default=256)
    parser.add_argument('-lr',              type=float, default=5e-4)
    parser.add_argument('-batch_size',      type=int,   default=128)
    parser.add_argument('-train_num_steps', type=int,   default=200000)
    parser.add_argument('-save_every',      type=int,   default=10000)
    parser.add_argument('-results_dir',     type=str,   default=None)
    parser.add_argument('-resume',          type=str,   default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device : {device}")
    print(f"  Mode   : Flow v2 (target-prediction composition)")

    train_task, test_tasks, dims, constraint_types = get_data_config(args.input_mode)
    results_dir = args.results_dir or f'./logs/flow_v2_{args.input_mode}_h{args.hidden_dim}'

    ds_kw = dict(input_mode=args.input_mode, pre_transform=pre_transform, visualize=False)
    train_dataset = GraphDataset(train_task, **ds_kw)
    test_datasets = {k: GraphDataset(t, **ds_kw) for k, t in test_tasks.items()
                     if os.path.isdir(f'./data/{t}')}
    print(f"  Train: {len(train_dataset):,}   Tests: {list(test_datasets.keys())}")

    model = FlowMatchingCCSP_v2(dims=dims, hidden_dim=args.hidden_dim,
                                 constraint_types=constraint_types,
                                 normalize=True, device=device).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_p:,}")

    # Sanity check
    b = next(iter(DataLoader(train_dataset, batch_size=4)))
    loss = model.compute_loss(b)
    print(f"  Sanity loss: {loss.item():.4f}")

    trainer = FlowTrainer_v2(model, train_dataset, test_datasets,
                              lr=args.lr, batch_size=args.batch_size,
                              train_num_steps=args.train_num_steps,
                              save_every=args.save_every,
                              results_folder=results_dir)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
