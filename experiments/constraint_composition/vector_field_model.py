from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VectorFieldMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class LearnedVectorField:
    model: VectorFieldMLP
    mean: np.ndarray
    std: np.ndarray
    max_nodes: int


def train_vector_field_model(
    x_np: np.ndarray,
    v_np: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = 'cpu',
    max_nodes: int = 0,
) -> tuple[LearnedVectorField, Dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_np = np.asarray(x_np, dtype=np.float32)
    v_np = np.asarray(v_np, dtype=np.float32)
    mean = x_np.mean(axis=0).astype(np.float32)
    std = x_np.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_norm = ((x_np - mean) / std).astype(np.float32)

    if np.isnan(x_norm).any():
        raise ValueError('NaNs detected after vector-field feature normalization.')

    dataset = TensorDataset(
        torch.tensor(x_norm, dtype=torch.float32),
        torch.tensor(v_np, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VectorFieldMLP(input_dim=x_np.shape[1], output_dim=v_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses: List[float] = []

    for _ in range(epochs):
        batch_losses = []
        for x_batch, v_batch in loader:
            x_batch = x_batch.to(device)
            v_batch = v_batch.to(device)
            pred = model(x_batch)
            loss = ((pred - v_batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

    model.eval()
    bundle = LearnedVectorField(
        model=model,
        mean=mean,
        std=std,
        max_nodes=int(max_nodes),
    )
    stats = {
        'num_samples': int(x_np.shape[0]),
        'input_dim': int(x_np.shape[1]),
        'output_dim': int(v_np.shape[1]),
        'target_mean_abs': float(np.abs(v_np).mean()),
        'target_max_abs': float(np.abs(v_np).max()),
        'final_train_loss': float(epoch_losses[-1]) if epoch_losses else 0.0,
        'epoch_losses': epoch_losses,
        'max_nodes': int(max_nodes),
        'has_nan_after_norm': bool(np.isnan(x_norm).any()),
    }
    return bundle, stats
