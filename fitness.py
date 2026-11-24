from __future__ import annotations

import torch

from config import LIF_CONFIG, LIFConfig
from lif import class_firing_rates, firing_rate, input_current


def distancia_AFR(AFR: torch.Tensor) -> torch.Tensor:
    """Suma de las distancias euclidianas entre las AFR de cada par de clases. (…×K) -> (…)."""
    K = AFR.shape[-1]
    k, l = torch.triu_indices(K, K, offset=1, device=AFR.device)   # pares k < l   ej. (0,1), (0,2), (1,2)
    # AFR_k es un escalar: la distancia euclidiana en ℝ¹ es |AFR_k − AFR_l|
    return (AFR[..., k] - AFR[..., l]).abs().sum(dim=-1)


def fitness(AFR: torch.Tensor, SDFR: torch.Tensor) -> torch.Tensor:
    """[ec. 5] F_i = 1 / distancia(AFR) + Σ_k SDFR_k (se minimiza). (…×K), (…×K) -> (…)."""
    return 1.0 / distancia_AFR(AFR) + SDFR.sum(dim=-1)


def fitness_population(x: torch.Tensor, y_idx: torch.Tensor, w: torch.Tensor,
                       cfg: LIFConfig = LIF_CONFIG) -> torch.Tensor:
    """
        Función objetivo del DE: F_i de cada vector de pesos w̄_i de la población, con los vectores x̄_j.
        x (N×n), y_idx (N,), w (NP×n) -> F_i (NP,)
    """
    I = input_current(x, w, cfg.theta)           # (NP×N)   ej. (60×75)
    FR = firing_rate(I, cfg)                     # (NP×N)   FR_j de cada x̄_j con cada w̄_i
    AFR, SDFR = class_firing_rates(FR, y_idx)    # (NP×K), (NP×K)
    return fitness(AFR, SDFR)                    # (NP,)
