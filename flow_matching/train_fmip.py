"""
FMIP-style Flow Matching for CCSP (Pilot)
==========================================
Adds a latent categorical mode variable z that conditions the velocity
predictor. Tests whether multimodal branching helps CCSP solving over
a single global vector field.

Key additions over FlowMatchingCCSP:
  - mode_embedding: Embedding(n_modes, hidden_dim)
  - mode_head: mean-pooled scene repr → mode logits (monitoring/diversity)
  - pose_decoder: conditioned on mode emb — (hidden*2 → pose_dim)
  - forward(x_t, batch, t, z=None): z=None samples from uniform prior
  - compute_loss: optional best-of-K mode routing

Training loss:
  L = L_flow                          (simple, z ~ Uniform)
  L = L_flow via best-of-K z routing  (--best_of_k > 1)

Usage:
    python train_fmip.py
    python train_fmip.py --n_modes 8 --best_of_k 4 --train_num_steps 100000
    python train_fmip.py --resume logs/flow_qualitative_h256/flow_model_best.pt
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader


from flow_matching.datasets import GraphDataset
from networks.data_transforms import pre_transform
from flow_matching.train_flow import (FlowMatchingCCSP, FlowTrainer, get_data_config,
                        _sample_flow_simple, _fast_constraint_check)


# ═══════════════════════════════════════════════════════════════════════════════
# FMIP Model
# ═══════════════════════════════════════════════════════════════════════════════

class FlowMatchingCCSP_FMIP(FlowMatchingCCSP):
    """Flow model with a latent categorical mode variable z.

    z ∈ {0, ..., n_modes-1} indexes a learned mode embedding that is
    injected at the pose decoder, conditioning the velocity prediction
    on a discrete "branch" representing different topological strategies.

    Architecture change (minimal — constraint MLPs unchanged):
      pose_decoder input:  hidden_dim → 2*hidden_dim  (appends mode_emb)
      mode_head:           mean-pooled geom_emb → n_modes logits

    forward(x_t, batch, t, z=None)
      z: LongTensor [n_scenes] or None (samples Uniform if None)
    """

    def __init__(self, *args, n_modes=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_modes = n_modes
        device = self.device

        # Mode embedding
        self.mode_embedding = nn.Embedding(n_modes, self.hidden_dim).to(device)

        # Mode head — pools per-scene geometry features → mode logits
        # Used for diversity monitoring; not trained with an explicit loss by default
        self.mode_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, n_modes),
        ).to(device)

        # Override pose_decoder: now takes [hidden (constraint) + hidden (mode)]
        pose_dim = self.dims[-1][0]
        self.pose_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, pose_dim),
        ).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x_t, batch, t, z=None):
        """Predict velocity v_θ(x_t, t, z).

        Args:
            x_t:   [N, pose_dim]  current poses
            batch: PyG DataBatch
            t:     float ∈ [0,1]
            z:     LongTensor [n_scenes] or None

        Returns:
            v:     [N, pose_dim]  predicted velocity
        """
        import jactorch

        device = self.device
        t_tensor = self._t_tensor(t, device)
        n_nodes = x_t.shape[0]
        pose_dim = self.dims[-1][0]

        geom_emb, pose_emb, edge_index = self._encode(x_t, batch)

        batch_vec = batch.batch.to(device)           # [N] — scene index per node
        n_scenes = int(batch_vec.max().item()) + 1

        # Resolve mode variable
        if z is None:
            z = torch.randint(0, self.n_modes, (n_scenes,), device=device)
        else:
            z = z.to(device)

        mode_emb = self.mode_embedding(z)            # [n_scenes, hidden]
        node_mode_emb = mode_emb[batch_vec]          # [N, hidden]

        # Message passing with mode injected at decoder
        all_v = torch.zeros(n_nodes, pose_dim, device=device)
        all_count = torch.zeros(n_nodes, device=device)

        for i, mlp in enumerate(self.constraint_mlps):
            edge_mask = (batch.edge_attr.to(device) == i)
            if edge_mask.sum() == 0:
                continue
            edges = edge_index[edge_mask]
            src, dst = edges[:, 0], edges[:, 1]
            n_edges = src.shape[0]

            t_emb = self.time_mlp(
                jactorch.add_dim(t_tensor, 0, n_edges)[:, 0])  # [E, hidden]

            inp = torch.cat([
                geom_emb[src], geom_emb[dst],
                pose_emb[src], pose_emb[dst],
                t_emb], dim=-1)                                 # [E, 5*hidden]
            out = mlp(inp)                                      # [E, 2*hidden]

            # Inject mode embedding before decoder
            h_src = torch.cat([out[:, :self.hidden_dim],
                                node_mode_emb[src]], dim=-1)    # [E, 2*hidden]
            h_dst = torch.cat([out[:, self.hidden_dim:],
                                node_mode_emb[dst]], dim=-1)

            v_src = self.pose_decoder(h_src)                    # [E, pose_dim]
            v_dst = self.pose_decoder(h_dst)

            all_v.scatter_add_(0, src.unsqueeze(-1).expand_as(v_src), v_src)
            all_v.scatter_add_(0, dst.unsqueeze(-1).expand_as(v_dst), v_dst)
            all_count.scatter_add_(0, src, torch.ones(n_edges, device=device))
            all_count.scatter_add_(0, dst, torch.ones(n_edges, device=device))

        denom = all_count.unsqueeze(-1).clamp(min=1)
        all_v = all_v / denom.sqrt() if self.normalize else all_v / denom

        mask = batch.mask.bool().to(device)
        all_v[mask] = 0.0
        return all_v

    def predict_mode_logits(self, batch):
        """Predict mode distribution for a batch (for diversity monitoring).

        Returns:
            logits: [n_scenes, n_modes]
        """
        device = self.device
        batch = batch.to(device)
        x = batch.x
        geom_begin, geom_end = self.dims[0][1], self.dims[0][2]
        geom_emb = self.geom_encoder(x[:, geom_begin:geom_end])  # [N, hidden]

        batch_vec = batch.batch.to(device)
        n_scenes = int(batch_vec.max().item()) + 1

        # Mean-pool over nodes per scene
        scene_emb = torch.zeros(n_scenes, self.hidden_dim, device=device)
        n_nodes_per = torch.zeros(n_scenes, device=device)
        scene_emb.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(geom_emb), geom_emb)
        n_nodes_per.scatter_add_(0, batch_vec, torch.ones(batch_vec.shape[0], device=device))
        scene_emb = scene_emb / n_nodes_per.unsqueeze(-1).clamp(min=1)

        return self.mode_head(scene_emb)  # [n_scenes, n_modes]

    # ── Training loss ─────────────────────────────────────────────────────────

    def compute_loss(self, batch, best_of_k=1):
        """OT-CFM loss with mode conditioning.

        Args:
            batch:      PyG DataBatch
            best_of_k:  1 → uniform random z;  K>1 → best-of-K routing

        Returns:
            (loss tensor, loss_dict)
        """
        device = self.device
        batch = batch.to(device)

        pose_begin = self.dims[-1][1]
        pose_end   = self.dims[-1][2]

        x_1  = batch.x[:, pose_begin:pose_end].clone()
        mask = batch.mask.bool()
        free = ~mask

        if free.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        x_0 = torch.randn_like(x_1)
        t   = torch.rand(1).item()
        x_t = (1.0 - t) * x_0 + t * x_1
        x_t[mask] = x_1[mask]
        u_t = x_1 - x_0

        batch_vec = batch.batch.to(device)
        n_scenes  = int(batch_vec.max().item()) + 1

        if best_of_k <= 1:
            # Simple: uniform random z across all scenes
            z = torch.randint(0, self.n_modes, (n_scenes,), device=device)
            v_pred = self.forward(x_t, batch, t, z=z)
            L_flow = F.mse_loss(v_pred[free], u_t[free])
            return L_flow, {'flow': L_flow.item(), 'z_mean': float(z.float().mean())}

        else:
            # Best-of-K: find best mode (no grad), then backprop through it
            best_loss_val = float('inf')
            best_k = 0
            K = min(best_of_k, self.n_modes)

            with torch.no_grad():
                for k in range(K):
                    z_k = torch.full((n_scenes,), k, dtype=torch.long, device=device)
                    v_k = self.forward(x_t, batch, t, z=z_k)
                    loss_k = F.mse_loss(v_k[free], u_t[free]).item()
                    if loss_k < best_loss_val:
                        best_loss_val = loss_k
                        best_k = k

            z_best = torch.full((n_scenes,), best_k, dtype=torch.long, device=device)
            v_pred = self.forward(x_t, batch, t, z=z_best)
            L_flow = F.mse_loss(v_pred[free], u_t[free])
            return L_flow, {'flow': L_flow.item(), 'best_k': best_k}


# ═══════════════════════════════════════════════════════════════════════════════
# FMIP Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class FlowTrainer_FMIP(FlowTrainer):
    """Trainer for FlowMatchingCCSP_FMIP.

    Extends FlowTrainer with:
      - best_of_k routing during training
      - mode usage logging (which z values are being selected)
    """

    def __init__(self, *args, best_of_k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_of_k = best_of_k

    def train(self):
        self.model.train()
        dl_iter = iter(self.train_dl)
        losses = []
        best_k_counts = {k: 0 for k in range(self.model.n_modes)}
        t0 = time.time()
        n_modes = self.model.n_modes
        K = self.best_of_k

        print(f"\n  Training FMIP for {self.train_num_steps:,} steps …")
        print(f"  n_modes={n_modes}, best_of_k={K}")

        while self.step < self.train_num_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl)
                batch = next(dl_iter)

            self.opt.zero_grad(set_to_none=True)
            loss, loss_dict = self.model.compute_loss(batch, best_of_k=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            losses.append(loss_dict.get('flow', loss.item()))
            if 'best_k' in loss_dict:
                best_k_counts[loss_dict['best_k']] += 1
            self.step += 1

            if self.step % 1000 == 0:
                avg = sum(losses[-200:]) / min(200, len(losses))
                elapsed = (time.time() - t0) / 60
                mode_str = ''
                if K > 1:
                    total = sum(best_k_counts.values()) or 1
                    mode_str = '  z:[' + ','.join(
                        f'{best_k_counts[k]/total:.2f}' for k in range(n_modes)) + ']'
                print(f"  step {self.step:6d}/{self.train_num_steps} | "
                      f"loss={avg:.5f} | {elapsed:.1f}min{mode_str}")

            if self.step % self.save_every == 0:
                self.save(self.step // self.save_every)
                self._quick_eval()

        self.save('final')
        print("\n  Training complete.")
        if K > 1:
            total = sum(best_k_counts.values()) or 1
            print("  Mode usage distribution:")
            for k in range(n_modes):
                print(f"    z={k}: {100*best_k_counts[k]/total:.1f}%")

    @torch.no_grad()
    def _quick_eval(self):
        """Eval with z sampled randomly (mode-averaged performance)."""
        self.model.eval()
        print(f"\n  [eval @ step {self.step}]")
        total_s = total_n = 0

        for n_obj, dl in sorted(self.test_dls.items()):
            s = n = 0
            for batch in dl:
                poses = _sample_fmip_simple(self.model, batch,
                                            n_steps=20, device=self.device)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _sample_fmip_simple(model, batch, n_steps=20, z=None, device='cuda'):
    """Euler ODE for FMIP model (z=None → sample uniformly each step)."""
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
        v = model(x_t, batch, step * dt, z=z)
        x_t = x_t + v * dt
        x_t[mask] = x_clean[mask]
        for i in range(x_t.shape[0]):
            if mask[i]: continue
            w = geoms[i, 0].item(); h = geoms[i, 1].item()
            x_t[i, 0] = x_t[i, 0].clamp(w + 0.02, 2.0 - w - 0.02)
            x_t[i, 1] = x_t[i, 1].clamp(h + 0.02, 2.0 - h - 0.02)

    return x_t


def load_fmip_checkpoint(ckpt_path, dims, hidden_dim, constraint_types,
                          n_modes=4, device='cuda'):
    """Load a FlowMatchingCCSP_FMIP checkpoint."""
    model = FlowMatchingCCSP_FMIP(
        dims=dims, hidden_dim=hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device,
        n_modes=n_modes,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [FMIP load] missing keys: {len(missing)}")
    print(f"  Loaded FMIP checkpoint: {ckpt_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train FMIP-style flow model for CCSP')
    parser.add_argument('--hidden_dim',      type=int,   default=256)
    parser.add_argument('--n_modes',         type=int,   default=4,
                        help='Number of latent modes z')
    parser.add_argument('--best_of_k',       type=int,   default=1,
                        help='K for best-of-K mode routing during training (1=uniform)')
    parser.add_argument('--train_num_steps', type=int,   default=100000)
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--save_every',      type=int,   default=10000)
    parser.add_argument('--resume',          type=str,   default=None,
                        help='Path to checkpoint to resume/fine-tune from')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_mode = 'qualitative'

    print(f"\n{'#'*65}")
    print(f"# FMIP-style Flow Matching (Pilot)")
    print(f"# n_modes={args.n_modes}  best_of_k={args.best_of_k}")
    print(f"# steps={args.train_num_steps}  hidden_dim={args.hidden_dim}")
    print(f"{'#'*65}")

    _, _, dims, constraint_types = get_data_config(input_mode)

    model = FlowMatchingCCSP_FMIP(
        dims=dims, hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device,
        n_modes=args.n_modes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sd   = ckpt.get('model_state_dict', ckpt)
        missing, _ = model.load_state_dict(sd, strict=False)
        print(f"  Resumed from {args.resume}  (missing: {len(missing)} keys)")

    train_task, test_tasks, _, _ = get_data_config(input_mode)
    ds_kw = dict(input_mode=input_mode, pre_transform=pre_transform, visualize=False)
    train_ds = GraphDataset(train_task, **ds_kw)
    test_datasets = {
        k: GraphDataset(v, **ds_kw)
        for k, v in test_tasks.items()
        if k <= 3 and os.path.isdir(f'./data/{v}')
    }
    print(f"  Train: {len(train_ds):,}   Tests: {list(test_datasets.keys())}")

    save_dir = f'./logs/fmip_qualitative_h{args.hidden_dim}_m{args.n_modes}'
    trainer = FlowTrainer_FMIP(
        model, train_ds, test_datasets,
        lr=args.lr, batch_size=args.batch_size,
        train_num_steps=args.train_num_steps,
        save_every=args.save_every,
        results_folder=save_dir,
        best_of_k=args.best_of_k,
    )
    trainer.train()


if __name__ == '__main__':
    main()
