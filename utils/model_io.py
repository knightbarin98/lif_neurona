from __future__ import annotations

import os
from dataclasses import asdict

import torch

from config import LIFConfig
from lif import LIFNeuron
from .logging_utils import write_json


def save_model(directory: str, neuron: LIFNeuron, metadata: dict, extra_tensors: dict | None = None) -> str:
    os.makedirs(directory, exist_ok=True)
    payload = {
        "w": neuron.w.detach().cpu(),
        "AFR": neuron.AFR.detach().cpu(),
        "SDFR": neuron.SDFR.detach().cpu(),
        "lif_config": asdict(neuron.cfg),
        "metadata": metadata,
    }
    if extra_tensors:
        payload.update({k: v.detach().cpu() for k, v in extra_tensors.items()})
    path = os.path.join(directory, "model.pt")
    torch.save(payload, path)
    # Copia legible por humanos (sin los tensores grandes)
    write_json(os.path.join(directory, "model.json"), {
        "w": payload["w"], "AFR": payload["AFR"], "SDFR": payload["SDFR"],
        "lif_config": payload["lif_config"],
        "metadata": metadata,
    })
    return path


def load_model(path: str, device: torch.device | str = "cpu",
               dtype: torch.dtype = torch.float32) -> tuple[LIFNeuron, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    neuron = LIFNeuron(
        w=payload["w"].to(device=device, dtype=dtype),
        AFR=payload["AFR"].to(device=device, dtype=dtype),
        SDFR=payload["SDFR"].to(device=device, dtype=dtype),
        cfg=LIFConfig(**payload["lif_config"]),
    )
    return neuron, payload
