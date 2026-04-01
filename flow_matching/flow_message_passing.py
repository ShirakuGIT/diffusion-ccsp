import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'envs'))
sys.path.insert(0, str(ROOT / 'networks'))
sys.path.insert(0, str(ROOT.parent / 'Jacinle'))

from train_flow import FlowMatchingCCSP


class MessagePassingFlowMatchingCCSP(FlowMatchingCCSP):
    """Flow model with intra-step constraint interaction via K message rounds."""

    def __init__(self, *args, n_rounds=3, residual=True, **kwargs):
        super().__init__(*args, aggregator='sum', **kwargs)
        self.n_rounds = n_rounds
        self.residual = residual

        hidden_dim = self.hidden_dim

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        ).to(self.device)
        self.node_norm = nn.LayerNorm(hidden_dim).to(self.device)

    def _aggregate_messages(self, node_h, geom_emb, edge_index, edge_attr, t_emb):
        device = self.device
        n_nodes = node_h.shape[0]

        all_msg = torch.zeros(n_nodes, self.hidden_dim, device=device)
        all_count = torch.zeros(n_nodes, device=device)

        for i, mlp in enumerate(self.constraint_mlps):
            edge_mask = (edge_attr == i)
            if edge_mask.sum() == 0:
                continue

            edges = edge_index[edge_mask]
            src, dst = edges[:, 0], edges[:, 1]
            n_edges = src.shape[0]

            edge_t = t_emb.expand(n_edges, -1)
            inputs = torch.cat([
                geom_emb[src], geom_emb[dst],
                node_h[src], node_h[dst],
                edge_t,
            ], dim=-1)
            out = mlp(inputs)
            msg_src = out[:, :self.hidden_dim]
            msg_dst = out[:, self.hidden_dim:]

            all_msg.scatter_add_(0, src.unsqueeze(-1).expand_as(msg_src), msg_src)
            all_msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg_dst), msg_dst)
            all_count.scatter_add_(0, src, torch.ones(n_edges, device=device))
            all_count.scatter_add_(0, dst, torch.ones(n_edges, device=device))

        denom = all_count.unsqueeze(-1).clamp(min=1)
        if self.normalize:
            all_msg = all_msg / denom.sqrt()
        else:
            all_msg = all_msg / denom
        return all_msg

    def forward(self, x_t, batch, t):
        device = self.device
        n_nodes = x_t.shape[0]

        geom_emb, pose_emb, edge_index = self._encode(x_t, batch)
        edge_attr = batch.edge_attr.to(device)
        t_tensor = self._t_tensor(t, device)
        t_emb = self.time_mlp(t_tensor)
        node_t = t_emb.expand(n_nodes, -1)

        node_h = pose_emb
        for _ in range(self.n_rounds):
            agg_msg = self._aggregate_messages(node_h, geom_emb, edge_index, edge_attr, t_emb)
            update = self.node_update(torch.cat([node_h, agg_msg, node_t], dim=-1))
            if self.residual:
                node_h = self.node_norm(node_h + update)
            else:
                node_h = self.node_norm(update)

        all_v = self.pose_decoder(node_h)
        mask = batch.mask.bool().to(device)
        all_v[mask] = 0.0
        return all_v


def load_message_passing_checkpoint(ckpt_path, dims, hidden_dim, constraint_types,
                                    n_rounds=3, device='cuda'):
    model = MessagePassingFlowMatchingCCSP(
        dims=dims,
        hidden_dim=hidden_dim,
        constraint_types=constraint_types,
        normalize=True,
        device=device,
        n_rounds=n_rounds,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt.get('model', {}))
    model.load_state_dict(sd, strict=True)
    return model
