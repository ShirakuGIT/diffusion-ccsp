"""
Flow Matching for Compositional Constraint Satisfaction Problems (CCSP).
=========================================================================
Trains a Conditional Flow Matching (CFM) model as a drop-in replacement for
the diffusion model in Diffusion-CCSP (Yang et al., CoRL 2023).

Key difference from Diffusion-CCSP:
  Diffusion:      predict clean x_0 (or noise ε) from noisy x_t
  Flow matching:  predict velocity v_θ(x_t, t) that transports noise → data

Training objective (OT-CFM, Lipman et al. ICLR 2023):
  x_t = (1-t)*x_0 + t*x_1      (OT interpolant, x_0~N(0,I), x_1=clean)
  u_t = x_1 - x_0               (conditional velocity target)
  L   = E[ ||v_θ(x_t, t) - u_t||² ]   (free nodes only)

Architecture: standalone module (geom_encoder, pose_encoder, time_mlp,
constraint_mlps, pose_decoder) closely mirroring ConstraintDiffuser but
with 2-layer constraint MLPs and flat parameter naming for direct checkpoint
compatibility with pre-trained weights.

Usage:
    python train_flow.py -input_mode qualitative
    python train_flow.py -input_mode qualitative -hidden_dim 256 -train_num_steps 200000
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
    """Return (train_task, test_tasks, dims, constraint_types) for an input_mode."""
    if input_mode == 'qualitative':
        train_task = "RandomSplitQualitativeWorld(30000)_qualitative_train"
        test_tasks = {
            i: f'RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split'
            for i in range(2, 6)
        }
        dims = ((2, 0, 2), (4, 2, 6))
        constraint_types = qualitative_constraints

    elif input_mode == 'diffuse_pairwise':
        train_task = "TriangularRandomSplitWorld[64]_(30000)_diffuse_pairwise_train"
        test_tasks = {
            i: f"TriangularRandomSplitWorld[64]_(10)_diffuse_pairwise_test_{i}_split"
            for i in range(2, 7)
        }
        dims = ((3, 0, 3), (4, 3, 7))
        constraint_types = puzzle_constraints

    elif input_mode == 'stability_flat':
        train_task = "RandomSplitWorld(24000)_stability_flat_train"
        test_tasks = {
            i: f'RandomSplitWorld(10)_stability_flat_test_{i}_object'
            for i in range(4, 7)
        }
        dims = ((2, 0, 2), (4, 2, 6))
        constraint_types = stability_constraints

    elif input_mode == 'robot_box':
        train_task = "TableToBoxWorld(10000)_train"
        test_tasks = {i: f"TableToBoxWorld(10)_test_{i}_object" for i in range(2, 7)}
        dims = ((8, 0, 8), (5, 10, 15), (5, 16, 21))
        constraint_types = robot_constraints

    else:
        raise ValueError(f"Unknown input_mode: {input_mode!r}")

    return train_task, test_tasks, dims, constraint_types


def _infer_input_mode(constraint_types):
    ct = set(constraint_types)
    if ct <= set(robot_constraints):
        return 'robot_box'
    if ct <= set(stability_constraints):
        return 'stability_flat'
    if ct <= set(qualitative_constraints):
        return 'qualitative'
    return 'diffuse_pairwise'


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding (shared with ConstraintDiffuser)
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
# Flow Matching model
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingCCSP(nn.Module):
    """Flow Matching model for CCSP.

    Standalone module (no ConstraintDiffuser backbone) with the same
    per-constraint message-passing structure.  Parameter names match the
    pre-trained checkpoint format:

        geom_encoder.*        – geometry feature encoder
        pose_encoder.*        – pose feature encoder
        time_mlp.*            – sinusoidal time embedding
        constraint_mlps.{i}.* – per-constraint 2-layer MLP
        pose_decoder.*        – pose output head

    Forward semantics: given (x_t, batch, t), returns predicted velocity
    v_θ(x_t, t) ∈ R^{n_nodes × pose_dim}.

    Training: Conditional Flow Matching (OT interpolant).
      x_t = (1-t)*x_0 + t*x_1,   u_t = x_1 - x_0,   L = MSE(v_θ, u_t)
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

        # Geometry encoder: geom_dim → hidden//2 → hidden
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
        ).to(device)

        # Pose encoder: pose_dim → hidden//2 → hidden
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
        ).to(device)

        # Time MLP (same as ConstraintDiffuser):
        # SinusoidalPosEmb → hidden → 4*hidden → hidden
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        ).to(device)

        # Per-constraint 2-layer MLPs:
        # input: [geom_i, geom_j, pose_i, pose_j, time] = 5*hidden
        # hidden: 2*hidden,  output: 2*hidden (split for two poses)
        self.constraint_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
            ).to(device)
            for _ in constraint_types
        ])

        # Pose decoder: hidden → hidden//2 → pose_dim
        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, pose_dim),
        ).to(device)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _t_tensor(self, t, device):
        """Float t ∈ [0,1] → long tensor [0, 999] matching diffusion scale."""
        t_int = max(0, min(999, int(round(float(t) * 999.0))))
        return torch.tensor([t_int], dtype=torch.long, device=device)

    def _encode(self, x_t, batch):
        """Return (geom_emb, pose_emb, edge_index) tensors on self.device."""
        device = self.device
        x = batch.x.to(device)

        geom_begin = self.dims[0][1]
        geom_end   = self.dims[0][2]
        geoms_in   = x[:, geom_begin:geom_end]

        geom_emb = self.geom_encoder(geoms_in)         # [N, hidden]
        pose_emb = self.pose_encoder(x_t.to(device))   # [N, hidden]
        edge_index = batch.edge_index.T.to(device)      # [E, 2]

        return geom_emb, pose_emb, edge_index

    # ── Public forward pass ──────────────────────────────────────────────

    def forward(self, x_t, batch, t):
        """Predict velocity v_θ(x_t, t).

        Args:
            x_t:   [N, pose_dim]   current poses
            batch: PyG DataBatch
            t:     float ∈ [0, 1]

        Returns:
            v:     [N, pose_dim]   predicted velocity
        """
        import jactorch

        device    = self.device
        t_tensor  = self._t_tensor(t, device)
        n_nodes   = x_t.shape[0]
        pose_dim  = self.dims[-1][0]

        geom_emb, pose_emb, edge_index = self._encode(x_t, batch)

        # Accumulate velocity predictions across all constraint types
        all_v     = torch.zeros(n_nodes, pose_dim, device=device)
        all_count = torch.zeros(n_nodes, device=device)

        for i, mlp in enumerate(self.constraint_mlps):
            # Edges of this constraint type
            edge_mask = (batch.edge_attr.to(device) == i)
            if edge_mask.sum() == 0:
                continue
            edges = edge_index[edge_mask]        # [E_i, 2]
            src, dst = edges[:, 0], edges[:, 1]  # source / destination node indices

            # Build per-edge input: [geom_s, geom_d, pose_s, pose_d, time]
            # Each embedding: [E_i, hidden]
            n_edges = src.shape[0]
            t_emb   = self.time_mlp(
                jactorch.add_dim(t_tensor, 0, n_edges)[:, 0]
            )  # [E_i, hidden]

            inputs = torch.cat([
                geom_emb[src],   # [E_i, hidden]
                geom_emb[dst],   # [E_i, hidden]
                pose_emb[src],   # [E_i, hidden]
                pose_emb[dst],   # [E_i, hidden]
                t_emb,           # [E_i, hidden]
            ], dim=-1)           # [E_i, 5*hidden]

            out = mlp(inputs)    # [E_i, 2*hidden]

            # Decode per-node velocity (one half per node in pair)
            v_src = self.pose_decoder(out[:, :self.hidden_dim])   # [E_i, pose_dim]
            v_dst = self.pose_decoder(out[:, self.hidden_dim:])   # [E_i, pose_dim]

            # Scatter-add into accumulator
            all_v.scatter_add_(0, src.unsqueeze(-1).expand_as(v_src), v_src)
            all_v.scatter_add_(0, dst.unsqueeze(-1).expand_as(v_dst), v_dst)
            all_count.scatter_add_(0, src, torch.ones(n_edges, device=device))
            all_count.scatter_add_(0, dst, torch.ones(n_edges, device=device))

        # Average (or sqrt-normalise as in ConstraintDiffuser)
        denom = all_count.unsqueeze(-1).clamp(min=1)
        if self.normalize:
            all_v = all_v / denom.sqrt()
        else:
            all_v = all_v / denom

        # Keep masked (fixed) nodes at zero velocity
        mask = batch.mask.bool().to(device)
        all_v[mask] = 0.0

        return all_v

    # ── Training loss ────────────────────────────────────────────────────

    def compute_loss(self, batch):
        """OT-CFM loss: MSE(v_θ(x_t, t), x_1 - x_0) on free nodes."""
        device = self.device
        batch  = batch.to(device)

        pose_begin = self.dims[-1][1]
        pose_end   = self.dims[-1][2]

        x_1  = batch.x[:, pose_begin:pose_end].clone()
        mask = batch.mask.bool()

        x_0  = torch.randn_like(x_1)
        t    = torch.rand(1).item()

        x_t  = (1.0 - t) * x_0 + t * x_1
        x_t[mask] = x_1[mask]          # fixed nodes stay at clean poses
        u_t  = x_1 - x_0               # OT conditional velocity

        v_pred = self.forward(x_t, batch, t)   # [N, pose_dim]

        free = ~mask
        if free.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return F.mse_loss(v_pred[free], u_t[free])


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_flow_checkpoint(ckpt_path, dims, hidden_dim, constraint_types,
                          device='cuda'):
    """Load a FlowMatchingCCSP checkpoint, handling old/new key formats.

    The pre-trained checkpoint uses flat keys:
        geom_encoder.*, pose_encoder.*, time_mlp.*,
        constraint_mlps.*, pose_decoder.*

    Returns model on `device`, or None if loading fails.
    """
    model = FlowMatchingCCSP(
        dims=dims, hidden_dim=hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt.get('model_state_dict', ckpt.get('model', {}))

    # Try strict load first (flat keys, current format)
    try:
        model.load_state_dict(sd, strict=True)
        print(f"  Loaded (strict): {ckpt_path}")
        return model
    except RuntimeError:
        pass

    # Try with backbone. prefix stripped (wrapping-based old format)
    try:
        stripped = {k.replace('backbone.', '', 1): v for k, v in sd.items()}
        model.load_state_dict(stripped, strict=True)
        print(f"  Loaded (backbone-stripped): {ckpt_path}")
        return model
    except RuntimeError:
        pass

    # Partial load with strict=False (best effort)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) < len(list(model.parameters())):
        print(f"  Loaded (partial): {ckpt_path} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        return model

    print(f"  FAILED to load checkpoint: {ckpt_path}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FlowTrainer:
    def __init__(self, model, train_dataset, test_datasets,
                 lr=5e-4, batch_size=128, train_num_steps=200000,
                 save_every=10000, results_folder='./logs/flow'):
        self.model          = model
        self.device         = model.device
        self.opt            = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.train_num_steps = train_num_steps
        self.save_every     = save_every
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.step           = 0
        self.best_succ      = -1.0

        kw = dict(pin_memory=True, num_workers=0)
        self.train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kw)
        self.test_dls = {
            k: DataLoader(d, batch_size=1, shuffle=False, **kw)
            for k, d in test_datasets.items()
        }

    def save(self, tag):
        path = self.results_folder / f'flow_model_{tag}.pt'
        torch.save({'step': self.step,
                    'model_state_dict': self.model.state_dict()}, str(path))
        print(f"  [save] {path}")

    def load(self, tag):
        path = self.results_folder / f'flow_model_{tag}.pt'
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.step = ckpt.get('step', 0)
        print(f"  [load] step={self.step}")

    def train(self):
        self.model.train()
        dl_iter  = iter(self.train_dl)
        losses   = []
        t0       = time.time()
        print(f"\n  Training for {self.train_num_steps:,} steps …")

        while self.step < self.train_num_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl)
                batch   = next(dl_iter)

            self.opt.zero_grad(set_to_none=True)
            loss = self.model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            losses.append(loss.item())
            self.step += 1

            if self.step % 1000 == 0:
                avg = sum(losses[-200:]) / min(200, len(losses))
                print(f"  step {self.step:6d}/{self.train_num_steps} | "
                      f"loss={avg:.5f} | {(time.time()-t0)/60:.1f}min")

            if self.step % self.save_every == 0:
                self.save(self.step // self.save_every)
                self._quick_eval()

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
                poses = _sample_flow_simple(self.model, batch, n_steps=20,
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
# Inline helpers (no circular import with solve_flow_ccsp)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _sample_flow_simple(model, batch, n_steps=20, device='cuda'):
    """Euler ODE integration (no QP) with tray clamping."""
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
        v   = model(x_t, batch, step * dt)
        x_t = x_t + v * dt
        x_t[mask] = x_clean[mask]
        # Hard clamp to tray
        for i in range(x_t.shape[0]):
            if mask[i]: continue
            w = geoms[i, 0].item(); h = geoms[i, 1].item()
            x_t[i, 0] = x_t[i, 0].clamp(w + 0.02, 2.0 - w - 0.02)
            x_t[i, 1] = x_t[i, 1].clamp(h + 0.02, 2.0 - h - 0.02)
    return x_t


def _fast_constraint_check(poses, batch, constraint_types, tol=0.02, device='cuda'):
    """Quick barrier check for training monitoring."""
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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching model for CCSP')
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
    print(f"  Device : {device}")

    train_task, test_tasks, dims, constraint_types = get_data_config(args.input_mode)
    results_dir = args.results_dir or f'./logs/flow_{args.input_mode}_h{args.hidden_dim}'

    ds_kw = dict(input_mode=args.input_mode, pre_transform=pre_transform, visualize=False)
    train_dataset = GraphDataset(train_task, **ds_kw)
    test_datasets = {k: GraphDataset(t, **ds_kw) for k, t in test_tasks.items()
                     if os.path.isdir(f'./data/{t}')}
    print(f"  Train: {len(train_dataset):,}   Tests: {list(test_datasets.keys())}")

    model = FlowMatchingCCSP(dims=dims, hidden_dim=args.hidden_dim,
                              constraint_types=constraint_types,
                              normalize=True, device=device).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_p:,}")

    # Sanity check
    b = next(iter(DataLoader(train_dataset, batch_size=4)))
    loss = model.compute_loss(b)
    print(f"  Sanity loss: {loss.item():.4f}")

    trainer = FlowTrainer(model, train_dataset, test_datasets,
                          lr=args.lr, batch_size=args.batch_size,
                          train_num_steps=args.train_num_steps,
                          save_every=args.save_every,
                          results_folder=results_dir)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
