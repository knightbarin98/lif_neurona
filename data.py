from __future__ import annotations

import csv

import torch

from config import CLASS_NAMES, INPUT_FEATURE

def load_iris_raw(path: str) -> tuple[list[list[float]], list[str]]:
    """Lee iris.csv y devuelve (feature: list[list[float]], labels: list[str])"""
    features, labels = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) < 5:
                continue
            try:
                float(row[0])
            except ValueError:
                continue
            feats = [float(row[i]) for i in range(INPUT_FEATURE)]
            label = row[INPUT_FEATURE]
            features.append(feats)
            labels.append(label)
    return features, labels

def build_tensors(features: list[list[float]], labels: list[str],
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve x (N×n): una fila por vector x̄_j ∈ ℝⁿ, y_idx (N,): clase k de cada vector."""
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    x = torch.tensor(features, dtype=torch.float32, device=device) #Nx4
    y_idx = torch.tensor([name_to_idx[l] for l in labels], dtype=torch.long, device=device) #N
    return x, y_idx
