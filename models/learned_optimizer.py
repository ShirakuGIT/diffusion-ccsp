import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_edge_index_2_by_e(edge_index):
    if edge_index.dim() != 2:
        raise ValueError(f'edge_index must be rank-2, got shape={tuple(edge_index.shape)}')
    if edge_index.shape[0] == 2:
        return edge_index
    if edge_index.shape[1] == 2:
        return edge_index.t()
    raise ValueError(f'edge_index must have shape [2, E] or [E, 2], got {tuple(edge_index.shape)}')


def _fixed_pose_values_from_batch(batch, pose_dim, x):
    if not hasattr(batch, 'mask'):
        return None, None

    fixed_mask = batch.mask.bool().to(x.device)
    fixed_values = x

    if hasattr(batch, 'x') and batch.x.size(-1) >= pose_dim:
        fixed_values = batch.x[:, -pose_dim:].to(x.device)

    return fixed_mask, fixed_values


def apply_fixed_poses(x, fixed_mask, fixed_values):
    if fixed_mask is None or fixed_values is None:
        return x
    return torch.where(fixed_mask.unsqueeze(-1), fixed_values, x)


class LearnedOptimizer(nn.Module):
    """Minimal edge-MLP message passing optimizer."""

    def __init__(self, pose_dim, num_edge_types, hidden_dim=256, step_scale=0.1):
        super().__init__()
        self.pose_dim = pose_dim
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim
        self.step_scale = step_scale

        input_dim = 2 * pose_dim + num_edge_types
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index = _as_edge_index_2_by_e(edge_index).to(x.device)
        edge_attr = edge_attr.to(x.device).long()

        src = edge_index[0].long()
        dst = edge_index[1].long()

        edge_type_oh = F.one_hot(edge_attr, num_classes=self.num_edge_types).to(dtype=x.dtype)

        msg_dst = self.edge_mlp(torch.cat([x[src], x[dst], edge_type_oh], dim=-1))
        msg_src = self.edge_mlp(torch.cat([x[dst], x[src], edge_type_oh], dim=-1))

        delta_x = torch.zeros_like(x)
        delta_x.index_add_(0, dst, msg_dst)
        delta_x.index_add_(0, src, msg_src)
        return self.step_scale * delta_x


@torch.no_grad()
def solve(model, x_init, batch, steps=5, step_size=0.1):
    x = x_init.clone()
    fixed_mask, fixed_values = _fixed_pose_values_from_batch(batch, x.shape[-1], x)

    for _ in range(steps):
        delta = model(x, batch.edge_index, batch.edge_attr)
        delta = torch.tanh(delta)
        x = x + step_size * delta
        x = apply_fixed_poses(x, fixed_mask, fixed_values)

    return x
