"""Shared temporal encoder with case-level MIL supervision; optional PyTorch."""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

CHANNELS = ["err", "peer", "cabin_ambient", "delta", "automatic", "full", "half", "observed", "eligible"]


class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 32, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalMIL(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        channels = len(CHANNELS) if channels is None else channels
        self.encoder = nn.Sequential(nn.Conv1d(channels, 32, 1), *[ResidualBlock(d) for d in [1, 2, 4, 8]])
        self.window_projection = nn.Linear(64, 32)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, mask):
        # cars, windows, channels, time; mask is cars, windows, time.
        cars, windows, channels, steps = x.shape
        z = self.encoder(x.reshape(cars * windows, channels, steps))
        m = mask.reshape(cars * windows, 1, steps)
        avg = (z * m).sum(-1) / m.sum(-1).clamp_min(1)
        maximum = z.masked_fill(~m.bool(), -1e4).amax(-1)
        valid = m.sum(-1) > 0
        maximum = torch.where(valid, maximum, torch.zeros_like(maximum))
        z = self.window_projection(torch.cat([avg, maximum], -1)).reshape(cars, windows, 32)
        wm = mask.sum(-1).gt(0).unsqueeze(-1)
        mean = (z * wm).sum(1) / wm.sum(1).clamp_min(1)
        maximum = z.masked_fill(~wm, -1e4).amax(1)
        maximum = torch.where(wm.any(1), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat([mean, maximum], -1)).squeeze(-1)


def windows(features):
    case = features.case
    timeline = case.frame.index.unique().sort_values()
    start = timeline.min().floor("h")
    end = timeline.max().ceil("h")
    grid = pd.date_range(start, end, freq="30s", inclusive="left")
    if len(grid) < 120:
        grid = pd.date_range(start, periods=120, freq="30s")
    all_x, all_m = [], []
    for car in case.cars:
        f = case.frame.loc[case.frame.car.eq(car)].reindex(grid)
        delta = f.cabin.diff().where(f.episode.eq(f.episode.shift()))
        x = pd.DataFrame(
            {
                "err": f.err,
                "peer": f.peer,
                "cabin_ambient": f.cabin - f.ambient_shared,
                "delta": delta,
                "automatic": f["mode"].eq("Automatic Cooling").astype(float),
                "full": f["mode"].eq("Full Cooling").astype(float),
                "half": f["mode"].eq("Half Cooling").astype(float),
                "observed": f.observed.fillna(False).astype(float),
                "eligible": f.eligible.fillna(False).astype(float),
            }
        )
        a = x.to_numpy(dtype=np.float32)
        m = f.eligible.fillna(False).to_numpy(dtype=np.float32)
        pad = (-len(a)) % 120
        if pad:
            a = np.pad(a, ((0, pad), (0, 0)), constant_values=np.nan)
            m = np.pad(m, (0, pad))
        a = a.reshape(-1, 120, len(CHANNELS)).transpose(0, 2, 1)
        m = m.reshape(-1, 120)
        m[m.mean(1) < 0.7] = 0
        all_x.append(a)
        all_m.append(m)
    xx = np.stack(all_x)
    mm = np.stack(all_m)
    keep = mm.sum(axis=(0, 2)) > 0
    if not keep.any():
        keep[0] = True
    return xx[:, keep], mm[:, keep]


def choose_device(value):
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    # CPU is faster for these very small grouped batches and fully reproducible.
    return "cpu"


def train_neural(data, labels, params, device="auto", deadline=None):
    device = choose_device(device)
    torch.set_num_threads(2)
    batches = {name: windows(f) for name, f in sorted(data.items())}
    # Fit scaling only on observed positions of training cases, equal sampling per case.
    samples = []
    for x, m in batches.values():
        flat = x.transpose(0, 1, 3, 2)[m.astype(bool)]
        if len(flat):
            samples.append(flat[:: max(1, len(flat) // 2000)])
    if not samples:
        raise ValueError("No eligible neural windows")
    joined = np.concatenate(samples)
    median = np.nanmedian(joined, axis=0)
    scale = np.nanpercentile(joined, 75, axis=0) - np.nanpercentile(joined, 25, axis=0)
    median = np.nan_to_num(median)
    scale = np.where(np.isfinite(scale) & (scale > 0.01), scale, 1.0)
    median[4:] = 0
    scale[4:] = 1
    tensors = {
        n: (torch.tensor(np.nan_to_num((x - median[None, None, :, None]) / scale[None, None, :, None]), device=device), torch.tensor(m, device=device))
        for n, (x, m) in batches.items()
    }
    targets = {n: data[n].case.cars.index(labels[n]) for n in data}
    states = []
    epoch_choices = []
    max_epochs = params.get("max_epochs", 100)
    seeds = params.get("seeds", [17, 42, 2026])

    def fit_pass(names, seed, epochs, validation=None):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TemporalMIL().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        best_loss = float("inf")
        best_epoch = 1
        stale = 0
        for epoch in range(epochs):
            if deadline and time.monotonic() > deadline:
                raise TimeoutError("Neural experiment time budget exhausted")
            model.train()
            optim.zero_grad()
            for name in names:
                x, m = tensors[name]
                # Same selected time windows for all cars; fixed cap keeps cases equally weighted.
                idx = np.sort(rng.choice(x.shape[1], min(8, x.shape[1]), replace=False))
                scores = model(x[:, idx], m[:, idx])
                loss = nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([targets[name]], device=device)) / len(names)
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if validation:
                model.eval()
                with torch.no_grad():
                    x, m = tensors[validation]
                    idx = np.linspace(0, x.shape[1] - 1, min(16, x.shape[1]), dtype=int)
                    score = model(x[:, idx], m[:, idx])
                    vl = float(nn.functional.cross_entropy(score.unsqueeze(0), torch.tensor([targets[validation]], device=device)))
                if vl < best_loss - 1e-4:
                    best_loss = vl
                    best_epoch = epoch + 1
                    stale = 0
                else:
                    stale += 1
                if stale >= 10:
                    break
        return model, best_epoch if validation else epochs

    names = sorted(data)
    for seed in seeds:
        # This pilot split is entirely inside the training partition; outer labels are inaccessible.
        if len(names) > 2:
            _, epochs = fit_pass(names[:-1], seed, max_epochs, names[-1])
        else:
            epochs = min(20, max_epochs)
        model, _ = fit_pass(names, seed, epochs)
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        epoch_choices.append(epochs)
    return {
        "states": states,
        "median": median,
        "scale": scale,
        "channels": CHANNELS,
        "epochs": epoch_choices,
        "device_trained": device,
        "early_stop_training_files": names[:-1],
        "early_stop_validation_file": names[-1] if len(names) > 2 else None,
    }


def predict_neural(bundle, features):
    x, m = windows(features)
    x = np.nan_to_num((x - bundle["median"][None, None, :, None]) / bundle["scale"][None, None, :, None])
    # Deterministic representative windows; same selection as inner validation.
    idx = np.linspace(0, x.shape[1] - 1, min(16, x.shape[1]), dtype=int)
    xt = torch.tensor(x[:, idx], dtype=torch.float32)
    mt = torch.tensor(m[:, idx], dtype=torch.float32)
    result = []
    with torch.no_grad():
        for state in bundle["states"]:
            model = TemporalMIL()
            model.load_state_dict(state)
            model.eval()
            result.append(model(xt, mt).numpy())
    return pd.Series(np.mean(result, axis=0), index=features.case.cars).where(features.table.available)
