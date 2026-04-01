from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@dataclass
class LearnedGraphVectorField:
    model: nn.Module
    node_mean: np.ndarray
    node_std: np.ndarray
    node_dim: int
    num_edge_types: int


class NoiseConditionedGraphVectorField(nn.Module):
    def __init__(self, node_dim: int, num_edge_types: int, hidden_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.self_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_edge_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_index = edge_index.long()
        edge_attr = edge_attr.long()

        h = self.node_encoder(x)
        out = self.self_head(h)

        src = edge_index[0]
        dst = edge_index[1]
        edge_oh = F.one_hot(edge_attr, num_classes=self.num_edge_types).to(dtype=x.dtype)

        msg_dst = self.edge_mlp(torch.cat([h[src], h[dst], edge_oh], dim=-1))
        msg_src = self.edge_mlp(torch.cat([h[dst], h[src], edge_oh], dim=-1))

        counts = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        counts.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device))
        counts.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))

        out.index_add_(0, dst, msg_dst)
        out.index_add_(0, src, msg_src)
        out = out / counts.clamp(min=1.0)
        return out


def train_graph_vector_field(
    dataset,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = 'cpu',
) -> tuple[LearnedGraphVectorField, Dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    node_features = torch.cat([data.x for data in dataset], dim=0).cpu().numpy().astype(np.float32)
    node_mean = node_features.mean(axis=0).astype(np.float32)
    node_std = node_features.std(axis=0).astype(np.float32)
    node_std = np.where(node_std < 1e-6, 1.0, node_std).astype(np.float32)

    normalized_graphs = []
    for data in dataset:
        x_norm = ((data.x.cpu().numpy().astype(np.float32) - node_mean) / node_std).astype(np.float32)
        normalized_graphs.append(
            Data(
                x=torch.tensor(x_norm, dtype=torch.float32),
                edge_index=data.edge_index.clone(),
                edge_attr=data.edge_attr.clone(),
                target_v=data.target_v.clone(),
                mask=data.mask.clone(),
            )
        )

    num_edge_types = int(max(int(data.edge_attr.max().item()) for data in dataset)) + 1
    node_dim = int(dataset[0].x.shape[1])
    model = NoiseConditionedGraphVectorField(
        node_dim=node_dim,
        num_edge_types=num_edge_types,
        hidden_dim=64,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(normalized_graphs, batch_size=batch_size, shuffle=True)
    epoch_losses: List[float] = []

    for _ in range(epochs):
        batch_losses = []
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            active = ~batch.mask
            sq_error = (pred - batch.target_v) ** 2
            if bool(active.any().item()):
                loss = sq_error[active].mean()
            else:
                loss = sq_error.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

    model.eval()
    bundle = LearnedGraphVectorField(
        model=model,
        node_mean=node_mean,
        node_std=node_std,
        node_dim=node_dim,
        num_edge_types=num_edge_types,
    )
    stats = {
        'num_graphs': int(len(dataset)),
        'num_nodes_total': int(node_features.shape[0]),
        'node_dim': int(node_dim),
        'num_edge_types': int(num_edge_types),
        'final_train_loss': float(epoch_losses[-1]) if epoch_losses else 0.0,
        'epoch_losses': epoch_losses,
    }
    return bundle, stats
