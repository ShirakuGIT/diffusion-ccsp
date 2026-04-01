from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConstraintMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class LearnedConstraintEnergy:
    model: ConstraintMLP
    mean: np.ndarray
    std: np.ndarray


def train_constraint_models(
    datasets: Dict[str, tuple[np.ndarray, np.ndarray]],
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = 'cpu',
) -> tuple[Dict[str, LearnedConstraintEnergy], Dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    models: Dict[str, LearnedConstraintEnergy] = {}
    stats: Dict[str, object] = {}
    loss_fn = nn.MSELoss()

    for constraint_type, (x_np, y_np) in sorted(datasets.items()):
        x_np = np.asarray(x_np, dtype=np.float32)
        y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
        mean = x_np.mean(axis=0).astype(np.float32)
        std = x_np.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        x_norm = ((x_np - mean) / std).astype(np.float32)

        dataset = TensorDataset(
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = ConstraintMLP(input_dim=x_np.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epoch_losses: List[float] = []

        for _ in range(epochs):
            batch_losses = []
            for z_batch, y_batch in loader:
                z_batch = z_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(z_batch)
                loss = loss_fn(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

        model.eval()
        models[constraint_type] = LearnedConstraintEnergy(
            model=model,
            mean=mean,
            std=std,
        )
        stats[constraint_type] = {
            'num_samples': int(x_np.shape[0]),
            'input_dim': int(x_np.shape[1]),
            'target_mean': float(y_np.mean()),
            'target_std': float(y_np.std()),
            'final_train_loss': float(epoch_losses[-1]) if epoch_losses else 0.0,
            'epoch_losses': epoch_losses,
        }

    return models, stats
