from __future__ import annotations

import torch

from config import FITNESS, LAMBDA_FITNESS, LIF_CONFIG, LIFConfig
from lif import class_firing_rates, firing_rate, input_current

# Funciones fitness (todas se minimizan). config.FITNESS elige cuál usa el DE:
#   "ec5":    [ec. 5 del artículo]  F_i = 1/distancia(AFR) + Σ_k SDFR_k
#   "fisher": F_i = 1 / min_vecinas J_kl,   J_kl = |AFR_k − AFR_l| / (SDFR_k + SDFR_l + ε)
#   "error":  F_i = error de entrenamiento con cl = argmin_k |AFR_k − F_r|  +  λ · 1/min_vecinas J_kl
FITNESS_FORMULAS: dict[str, str] = {
    "ec5": "F = 1/distancia(AFR) + Σ SDFR_k",
    "fisher": "F = 1 / min J_kl",
    "error": "F = error + λ/min J_kl",
}


def distancia_AFR(AFR: torch.Tensor) -> torch.Tensor:
    """Suma de las distancias euclidianas entre las AFR de cada par de clases. (…×K) -> (…)."""
    K = AFR.shape[-1]
    k, l = torch.triu_indices(K, K, offset=1, device=AFR.device)   # pares k < l   ej. (0,1), (0,2), (1,2)
    # AFR_k es un escalar: la distancia euclidiana en ℝ¹ es |AFR_k − AFR_l|
    return (AFR[..., k] - AFR[..., l]).abs().sum(dim=-1)


def fitness(AFR: torch.Tensor, SDFR: torch.Tensor) -> torch.Tensor:
    """[ec. 5] F_i = 1 / distancia(AFR) + Σ_k SDFR_k (se minimiza). (…×K), (…×K) -> (…)."""
    return 1.0 / distancia_AFR(AFR) + SDFR.sum(dim=-1)


def separacion_J(AFR: torch.Tensor, SDFR: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
        J_kl = |AFR_k − AFR_l| / (SDFR_k + SDFR_l + ε) de cada par de clases VECINAS en la recta de las AFR.
        Con una sola neurona las clases quedan en fila; el orden depende de w̄, por eso se ordenan primero.
        No tiene unidades: si FR se mide en Hz, numerador y denominador cambian igual.
        (…×K), (…×K) -> (…×(K−1))      ej. (60×3) -> (60×2)
    """
    orden = torch.argsort(AFR, dim=-1)                          # (…×K) clases de menor a mayor AFR
    AFR_o = torch.gather(AFR, -1, orden)
    SDFR_o = torch.gather(SDFR, -1, orden)
    # ε evita dividir entre cero cuando todas las flores de dos clases vecinas dan el mismo conteo de disparos
    return (AFR_o[..., 1:] - AFR_o[..., :-1]) / (SDFR_o[..., 1:] + SDFR_o[..., :-1] + epsilon)


def fitness_fisher(AFR: torch.Tensor, SDFR: torch.Tensor, epsilon: float) -> torch.Tensor:
    """F_i = 1 / min J_kl: el par de clases vecinas peor separado (cuello de botella). (…×K) -> (…)."""
    return 1.0 / separacion_J(AFR, SDFR, epsilon).min(dim=-1).values


def error_entrenamiento(FR: torch.Tensor, AFR: torch.Tensor, y_idx: torch.Tensor) -> torch.Tensor:
    """Fracción de vectores con argmin_k |AFR_k − FR_j| ≠ clase real. FR (…×N), AFR (…×K) -> (…)."""
    distances = (FR[..., :, None] - AFR[..., None, :]).abs()    # (…×N×1) − (…×1×K) -> (…×N×K)
    cl = torch.argmin(distances, dim=-1)                        # (…×N)
    return (cl != y_idx).to(FR.dtype).mean(dim=-1)


def fitness_from_FR(FR: torch.Tensor, y_idx: torch.Tensor, cfg: LIFConfig = LIF_CONFIG,
                    name: str = FITNESS) -> torch.Tensor:
    """F_i a partir de las razones de disparo FR (…×N) de los vectores x̄_j con clase y_idx (N,). -> (…)."""
    AFR, SDFR = class_firing_rates(FR, y_idx)    # (…×K), (…×K)
    epsilon = 1.0 / cfg.t                        # resolución de FR: un disparo en t ms
    if name == "ec5":
        return fitness(AFR, SDFR)
    if name == "fisher":
        return fitness_fisher(AFR, SDFR, epsilon)
    if name == "error":
        return error_entrenamiento(FR, AFR, y_idx) + LAMBDA_FITNESS * fitness_fisher(AFR, SDFR, epsilon)
    raise ValueError(f"FITNESS desconocida: {name!r} (usa 'ec5', 'fisher' o 'error')")


def fitness_population(x: torch.Tensor, y_idx: torch.Tensor, w: torch.Tensor,
                       cfg: LIFConfig = LIF_CONFIG) -> torch.Tensor:
    """
        Función objetivo del DE: F_i de cada vector de pesos w̄_i de la población, con los vectores x̄_j.
        x (N×n), y_idx (N,), w (NP×n) -> F_i (NP,)
    """
    I = input_current(x, w, cfg.theta)           # (NP×N)   ej. (60×75)
    FR = firing_rate(I, cfg)                     # (NP×N)   FR_j de cada x̄_j con cada w̄_i
    return fitness_from_FR(FR, y_idx, cfg)       # (NP,)
