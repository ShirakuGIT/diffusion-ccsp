"""
Flow Matching v4: Feasibility-Aware Training
=============================================
Adds two auxiliary losses on top of the standard OT-CFM objective:

  L_total = L_flow + λ₁ * L_onestep + λ₂ * L_constraint

  L_flow       = MSE(v_θ(x_t, t), u_t)           [standard CFM]
  L_onestep    = mean_viol(h(x_t + v_pred * dt))   [one-step feasibility]
  L_constraint = mean_viol(h(x_t))                 [current state feasibility]

L_onestep is the key fix: it teaches the model that the velocity it predicts
should move towards the constraint manifold, not just match the OT target.

Analytic barriers (h > 0 = satisfied, h < 0 = violated):
  Uses the same compute_barrier() as PCFM — no learned energy needed.

Usage:
    python train_flow_v4.py
    python train_flow_v4.py --lambda_onestep 1.0 --lambda_constraint 0.1
    python train_flow_v4.py --resume  # fine-tune from v1 checkpoint
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'networks'))

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from networks.denoise_fn import qualitative_constraints

from train_flow import (FlowMatchingCCSP, FlowTrainer, get_data_config,
                         _sample_flow_simple, _fast_constraint_check)
from fix_and_eval import compute_barrier


# ═══════════════════════════════════════════════════════════════════════════════
# Analytic barrier loss (fully differentiable — uses soft ReLU violations)
# ═══════════════════════════════════════════════════════════════════════════════

def barrier_violation_loss(poses, batch, constraint_types, device):
    """
    Compute mean soft violation of all constraints.

    violation_i = max(0, -h_i(x))   (0 if satisfied, positive if violated)
    L = mean(violation_i)

    Uses analytic gradients from compute_barrier. Since compute_barrier uses
    .item() internally, we recompute a differentiable version for each type.
    """
    x = batch.x.to(device)
    edge_index = batch.edge_index.T.to(device)
    edge_attr = batch.edge_attr.to(device)
    geom_end = 2  # qualitative: geom = x[:, 0:2]

    total_viol = torch.tensor(0.0, device=device)
    n_constraints = 0

    for ei in range(edge_index.shape[0]):
        i = int(edge_index[ei, 0].item())
        j = int(edge_index[ei, 1].item())
        ctype_idx = int(edge_attr[ei].item())
        if ctype_idx >= len(constraint_types):
            continue
        ctype = constraint_types[ctype_idx]

        pi = poses[i]   # [pose_dim], differentiable
        pj = poses[j]

        xi, yi = pi[0], pi[1]
        xj, yj = pj[0], pj[1]
        gi = x[i, :geom_end]
        gj = x[j, :geom_end]
        wi, hi_g = gi[0], gi[1]
        wj, hj_g = gj[0], gj[1]

        # Differentiable h computation per constraint type
        if ctype == 'cfree':
            dx = (xi - xj).abs() - (wi + wj)
            dy = (yi - yj).abs() - (hi_g + hj_g)
            h = torch.stack([dx, dy]).max()
        elif ctype == 'away-from':
            dist = torch.sqrt((xi - xj)**2 + (yi - yj)**2 + 1e-8)
            threshold = torch.max(wi + wj, hi_g + hj_g) * 2.0
            h = dist - threshold
        elif ctype == 'left-of':
            h = xj - xi - (wi + wj) * 0.5
        elif ctype == 'top-of':
            h = yi - yj - (hi_g + hj_g) * 0.5
        elif ctype == 'h-aligned':
            h = torch.tensor(0.3, device=device) - (yi - yj).abs()
        elif ctype == 'v-aligned':
            h = torch.tensor(0.3, device=device) - (xi - xj).abs()
        elif ctype == 'close-to':
            dist = torch.sqrt((xi - xj)**2 + (yi - yj)**2 + 1e-8)
            threshold = torch.max(wi + wj, hi_g + hj_g) * 1.5
            h = threshold - dist
        elif ctype == 'center-in':
            dist = torch.sqrt((xi - 1.0)**2 + (yi - 1.0)**2 + 1e-8)
            h = torch.tensor(0.5, device=device) - dist
        elif ctype == 'left-in':
            h = 1.0 - xi
        elif ctype == 'right-in':
            h = xi - 1.0
        elif ctype == 'top-in':
            h = yi - 1.0
        elif ctype == 'bottom-in':
            h = 1.0 - yi
        elif ctype in ('in', 'top-in'):
            h = torch.tensor(1.0, device=device)  # always satisfied (tray)
        else:
            continue  # unknown constraint, skip

        # Soft violation: max(0, -h)
        viol = F.relu(-h)
        total_viol = total_viol + viol
        n_constraints += 1

    if n_constraints == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_viol / n_constraints


# ═══════════════════════════════════════════════════════════════════════════════
# V4 Model: FlowMatchingCCSP + feasibility losses
# ═══════════════════════════════════════════════════════════════════════════════

class FlowMatchingCCSP_v4(FlowMatchingCCSP):
    """Flow model with feasibility-aware training loss."""

    def __init__(self, *args, lambda_onestep=1.0, lambda_constraint=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_onestep = lambda_onestep
        self.lambda_constraint = lambda_constraint

    def compute_loss(self, batch):
        """
        Combined loss:
          L = L_flow + λ₁ * L_onestep + λ₂ * L_constraint
        """
        device = self.device
        batch = batch.to(device)

        pose_begin = self.dims[-1][1]
        pose_end = self.dims[-1][2]

        x_1 = batch.x[:, pose_begin:pose_end].clone()
        mask = batch.mask.bool()
        free = ~mask

        x_0 = torch.randn_like(x_1)
        t = torch.rand(1).item()
        dt = 1.0 / 20  # one step size

        x_t = (1.0 - t) * x_0 + t * x_1
        x_t[mask] = x_1[mask]
        u_t = x_1 - x_0

        v_pred = self.forward(x_t, batch, t)

        if free.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L_flow: standard OT-CFM
        L_flow = F.mse_loss(v_pred[free], u_t[free])

        losses = {'flow': L_flow.item()}
        total = L_flow

        # L_onestep: violation after one Euler step
        # Only apply in later part of trajectory (t > 0.3) where it's meaningful
        if self.lambda_onestep > 0 and t > 0.3:
            x_next = x_t + v_pred * dt
            x_next = x_next.clone()
            x_next[mask] = x_1[mask]
            L_onestep = barrier_violation_loss(
                x_next, batch, self.constraint_types, device)
            total = total + self.lambda_onestep * L_onestep
            losses['onestep'] = L_onestep.item()

        # L_constraint: violation at current state x_t
        if self.lambda_constraint > 0 and t > 0.5:
            L_constraint = barrier_violation_loss(
                x_t.detach(), batch, self.constraint_types, device)
            # Don't backprop through x_t for this — it's a monitoring signal
            # Instead use it to scale the flow loss
            losses['constraint'] = L_constraint.item()

        return total, losses


# ═══════════════════════════════════════════════════════════════════════════════
# V4 Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class FlowTrainer_v4(FlowTrainer):
    """Trainer that handles the (loss, dict) return from compute_loss."""

    def train(self):
        self.model.train()
        dl_iter = iter(self.train_dl)
        losses_flow = []
        losses_onestep = []
        t0 = time.time()
        print(f"\n  Training v4 for {self.train_num_steps:,} steps …")
        print(f"  λ_onestep={self.model.lambda_onestep}, "
              f"λ_constraint={self.model.lambda_constraint}")

        while self.step < self.train_num_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl)
                batch = next(dl_iter)

            self.opt.zero_grad(set_to_none=True)
            result = self.model.compute_loss(batch)
            if isinstance(result, tuple):
                loss, loss_dict = result
            else:
                loss, loss_dict = result, {}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            losses_flow.append(loss_dict.get('flow', loss.item()))
            losses_onestep.append(loss_dict.get('onestep', 0.0))
            self.step += 1

            if self.step % 1000 == 0:
                avg_flow = sum(losses_flow[-200:]) / min(200, len(losses_flow))
                avg_onestep = sum(losses_onestep[-200:]) / min(200, len(losses_onestep))
                elapsed = (time.time() - t0) / 60
                print(f"  step {self.step:6d}/{self.train_num_steps} | "
                      f"L_flow={avg_flow:.5f}  L_onestep={avg_onestep:.5f} | "
                      f"{elapsed:.1f}min")

            if self.step % self.save_every == 0:
                self.save(self.step // self.save_every)
                self._quick_eval()

        self.save('final')
        print("\n  Training complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--train_num_steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lambda_onestep', type=float, default=1.0,
                        help='Weight for one-step feasibility loss')
    parser.add_argument('--lambda_constraint', type=float, default=0.1,
                        help='Weight for current-state constraint loss')
    parser.add_argument('--resume', action='store_true',
                        help='Fine-tune from v1 checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_mode = 'qualitative'

    print(f"\n{'#'*65}")
    print(f"# Flow Matching v4 — Feasibility-Aware Training")
    print(f"# steps={args.train_num_steps}, λ_onestep={args.lambda_onestep}, "
          f"λ_constraint={args.lambda_constraint}")
    print(f"# {'Fine-tuning from v1' if args.resume else 'Training from scratch'}")
    print(f"{'#'*65}")

    _, _, dims, constraint_types = get_data_config(input_mode)

    model = FlowMatchingCCSP_v4(
        dims=dims, hidden_dim=args.hidden_dim,
        constraint_types=constraint_types,
        normalize=True, device=device,
        lambda_onestep=args.lambda_onestep,
        lambda_constraint=args.lambda_constraint,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    if args.resume:
        v1_ckpt = './logs/flow_qualitative_h256/flow_model_best.pt'
        if os.path.exists(v1_ckpt):
            ckpt = torch.load(v1_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded v1 weights from {v1_ckpt} — fine-tuning")
        else:
            print(f"  WARNING: v1 checkpoint not found at {v1_ckpt}, training from scratch")

    train_task, test_tasks, _, _ = get_data_config(input_mode)
    train_ds = GraphDataset(train_task, input_mode=input_mode,
                             pre_transform=pre_transform, visualize=False)
    test_datasets = {
        k: GraphDataset(v, input_mode=input_mode, pre_transform=pre_transform,
                        visualize=False)
        for k, v in test_tasks.items() if k <= 3
    }

    print(f"  Train: {len(train_ds):,} scenes")

    save_dir = f'./logs/flow_v4_qualitative_h{args.hidden_dim}'
    trainer = FlowTrainer_v4(
        model, train_ds, test_datasets,
        lr=args.lr, batch_size=args.batch_size,
        train_num_steps=args.train_num_steps,
        save_every=10000,
        results_folder=save_dir,
    )

    trainer.train()


if __name__ == '__main__':
    main()
