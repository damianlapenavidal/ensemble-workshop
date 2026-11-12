"""The two finalist models from the ocean-forecasting competition, plus helpers to load them.

Copied from the competition repository (src/model.py, src/predict.py) so this workshop folder runs on its own.
Both models predict a *correction* that is added to persistence (the last observed SST map).
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn


class CausalConv1d(nn.Module):
    """1D convolution padded on the past side only, so day t never sees days after t."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.left_pad = dilation * (kernel_size - 1)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.main_path = nn.Sequential(CausalConv1d(in_channels, out_channels, kernel_size, dilation), nn.ReLU(), nn.Dropout(dropout))
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.main_path(x) + self.shortcut(x)


class TCNModel(nn.Module):
    """Flattened TCN: each daily map becomes a 168-value vector; a causal TCN runs over the 14 days."""
    def __init__(self, grid_shape=(12, 14), hidden_channels=32, forecast_days=3, kernel_size=3, dilations=(1, 2, 4),
                 dropout=0.1, input_mean=0.0, input_std=1.0):
        super().__init__()
        self.grid_shape, self.forecast_days = grid_shape, forecast_days
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        cells = grid_shape[0] * grid_shape[1]
        blocks, in_channels = [], cells
        for dilation in dilations:
            blocks.append(TemporalBlock(in_channels, hidden_channels, kernel_size, dilation, dropout))
            in_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.readout = nn.Linear(hidden_channels, forecast_days * cells)

    def forward(self, x):
        sequence = ((x - self.input_mean) / self.input_std).flatten(start_dim=2).permute(0, 2, 1)   # (batch, 168, 14)
        summary = self.network(sequence)[:, :, -1]                                                  # last day's features
        correction = self.readout(summary).reshape(x.shape[0], self.forecast_days, *self.grid_shape) * self.input_std
        return x[:, -1:].repeat(1, self.forecast_days, 1, 1) + correction


class SpatialTCN(nn.Module):
    """(2+1)D model: Conv2d per day -> causal TCN shared by every grid cell -> Conv2d head."""
    def __init__(self, grid_shape=(12, 14), hidden_channels=32, forecast_days=3, kernel_size=2, dilations=(1, 2, 4, 8),
                 dropout=0.1, mask=None, input_mean=0.0, input_std=1.0):
        super().__init__()
        self.forecast_days = forecast_days
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        self.register_buffer("land_mask", torch.ones(grid_shape) if mask is None else torch.as_tensor(mask, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Conv2d(2, hidden_channels, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU())
        self.temporal = nn.Sequential(*[TemporalBlock(hidden_channels, hidden_channels, kernel_size, d, dropout) for d in dilations])
        self.head = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(hidden_channels, forecast_days, 1))

    def forward(self, x):
        b, t, h, w = x.shape
        z = ((x - self.input_mean) / self.input_std).reshape(b * t, 1, h, w)
        z = torch.cat([z, self.land_mask.expand(b * t, 1, h, w)], dim=1)
        z = self.encoder(z)
        c = z.shape[1]
        z = z.reshape(b, t, c, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, c, t)   # every cell is its own sequence
        z = self.temporal(z)[:, :, -1].reshape(b, h, w, c).permute(0, 3, 1, 2)
        correction = self.head(z) * self.input_std
        return x[:, -1:].repeat(1, self.forecast_days, 1, 1) + correction


MODELS = {"tcn": TCNModel, "spatial_tcn": SpatialTCN}


def load_checkpoint(path):
    """Rebuild a trained model from one of the competition checkpoints. Returns (model_type, model)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = MODELS[ckpt["model_type"]](hidden_channels=cfg["hidden_channels"], kernel_size=cfg["kernel_size"],
                                       dilations=tuple(cfg["dilations"]), dropout=cfg["dropout"],
                                       input_mean=float(ckpt["input_mean"]), input_std=float(ckpt["input_std"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt["model_type"], model.eval()


@torch.no_grad()
def predict(model, X, batch_size=64):
    """Run a model over X with shape (examples, 14, 12, 14); returns (examples, 3, 12, 14) in degrees Celsius."""
    X = torch.as_tensor(X, dtype=torch.float32)
    return np.concatenate([model(X[i:i + batch_size]).numpy() for i in range(0, len(X), batch_size)])


def rmse_per_day(pred, y, mask):
    """Official metric pieces: RMSE over ocean cells for each lead day. The score is their mean."""
    return np.sqrt(((pred - y) ** 2)[:, :, mask].mean(axis=(0, 2)))


def score(pred, y, mask):
    return float(rmse_per_day(pred, y, mask).mean())
