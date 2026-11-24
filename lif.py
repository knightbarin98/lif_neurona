from __future__ import annotations

from dataclasses import dataclass

import torch

from config import LIF_CONFIG, NUM_CLASSES, LIFConfig

# Modelo LIF del artículo [ec. 2]:  dV/dt = I + a − bV = F(t, V);   si V ≥ V_th  ->  V = c
# resuelto con el método de Euler:   V_{i+1} = V_i + h F(t_i, V_i) = V_i + h (I + a − b V_i)
# Todas las funciones aceptan cualquier forma de I: un escalar por neurona simulada, todas a la vez.


def input_current(x: torch.Tensor, w: torch.Tensor, theta: float = LIF_CONFIG.theta) -> torch.Tensor:
    """[ec. 4] I = x̄·w̄·θ: corriente de entrada de cada vector x̄_j (un escalar por par x̄_j, w̄_i)."""
    # w (n,):    (1×n) * (N×n) -> suma sobre n -> (N,)          ej. (4,) y (75×4) -> (75,)
    # w (NP×n):  (NP×1×n) * (N×n) -> (NP×N×n) -> suma -> (NP×N) ej. (60×4) y (75×4) -> (60×75)
    # producto punto elemento a elemento: cada I_j no depende de cuántos w̄_i se evalúan juntos
    return (w[..., None, :] * x).sum(dim=-1) * theta


def F(t_i: float, V_i: torch.Tensor, I: torch.Tensor, a: float = LIF_CONFIG.a,
      b: float = LIF_CONFIG.b) -> torch.Tensor:
    """F(t, V) = I + a − bV. El modelo es autónomo: t_i no interviene, se deja por fidelidad a la ecuación."""
    return I + a - b * V_i


def euler_step(t_i: float, V_i: torch.Tensor, I: torch.Tensor, cfg: LIFConfig = LIF_CONFIG) -> torch.Tensor:
    """V_{i+1} = V_i + h F(t_i, V_i)."""
    return V_i + cfg.h * F(t_i, V_i, I, cfg.a, cfg.b)


def simulation_steps(cfg: LIFConfig = LIF_CONFIG) -> int:
    return int(round(cfg.t / cfg.h))     # t / h = 1000 ms / 1 ms = 1000 pasos de Euler


def simulate(I: torch.Tensor, cfg: LIFConfig = LIF_CONFIG,
             record: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
        Estimula la neurona con I durante t ms. Devuelve el total de disparos de cada neurona (misma forma que I)
        y, si record=True, el voltaje V en t = 0, h, ..., t (forma de I + (pasos+1,)), ya reiniciado tras cada disparo.
    """
    V_i = torch.full_like(I, cfg.V_0)            # V_{i=0} = −60
    disparos = torch.zeros_like(I)
    history = [V_i.clone()] if record else None
    for i in range(simulation_steps(cfg)):
        t_i = i * cfg.h
        V_next = euler_step(t_i, V_i, I, cfg)    # V_{i+1}
        disparo = V_next >= cfg.V_th             # V_{i+1} ≥ V_th  ->  disparo
        disparos += disparo
        V_i = V_next.masked_fill_(disparo, cfg.c)    # V_{i+1} = c
        if record:
            history.append(V_i.clone())
    V = torch.stack(history, dim=-1) if record else None
    return disparos, V


def firing_rate(I: torch.Tensor, cfg: LIFConfig = LIF_CONFIG) -> torch.Tensor:
    """FR = total de disparos / tiempo de simulación (disparos por ms)."""
    disparos, _ = simulate(I, cfg)
    return disparos / cfg.t


def class_firing_rates(FR: torch.Tensor, y_idx: torch.Tensor,
                       num_classes: int = NUM_CLASSES) -> tuple[torch.Tensor, torch.Tensor]:
    """
        AFR_k = Σ_j FR_j / p  y  SDFR_k (desviación estándar poblacional) de los p vectores de cada clase k.
        FR (…×N) -> AFR (…×K), SDFR (…×K)      ej. (60×75) -> (60×3), (60×3)
    """
    onehot = (y_idx[:, None] == torch.arange(num_classes, device=y_idx.device)).to(FR.dtype)   # (N×K)
    p = onehot.sum(dim=0)                                       # (K,) vectores por clase   ej. [25, 25, 25]
    AFR = (FR @ onehot) / p                                     # (…×N) @ (N×K) -> (…×K)
    desv = FR - AFR[..., y_idx]                                 # FR_j − AFR de su clase: (…×N)
    SDFR = torch.sqrt(((desv * desv) @ onehot) / p)             # (…×N) @ (N×K) -> (…×K)
    return AFR, SDFR


def classify(F_r: torch.Tensor, AFR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[ec. 3] cl = argmin_k |AFR_k − F_r|. F_r (N,), AFR (K,) -> cl (N,), |AFR_k − F_r| (N×K)."""
    distances = (AFR[None, :] - F_r[:, None]).abs()             # (1×K) − (N×1) -> (N×K)
    return torch.argmin(distances, dim=1), distances


@dataclass
class LIFNeuron:
    w: torch.Tensor              # (n,)  w̄: pesos sinápticos calibrados con DE
    AFR: torch.Tensor            # (K,)  razón de disparo promedio de cada clase (entrenamiento)
    SDFR: torch.Tensor           # (K,)  desviación estándar de la razón de disparo de cada clase
    cfg: LIFConfig = LIF_CONFIG

    @classmethod
    def fit(cls, w: torch.Tensor, x: torch.Tensor, y_idx: torch.Tensor,
            cfg: LIFConfig = LIF_CONFIG) -> LIFNeuron:
        # AFR y SDFR del conjunto de entrenamiento con los pesos w̄ ya optimizados
        AFR, SDFR = class_firing_rates(firing_rate(input_current(x, w, cfg.theta), cfg), y_idx)
        return cls(w=w, AFR=AFR, SDFR=SDFR, cfg=cfg)

    def current(self, x: torch.Tensor) -> torch.Tensor:
        return input_current(x, self.w, self.cfg.theta)          # (N×n) -> (N,)

    def firing_rate(self, x: torch.Tensor) -> torch.Tensor:
        return firing_rate(self.current(x), self.cfg)            # (N,) F_r de cada vector

    def membrane_potential(self, x: torch.Tensor) -> torch.Tensor:
        _, V = simulate(self.current(x), self.cfg, record=True)
        return V                                                 # (N×(pasos+1))   ej. (75×1001)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        cl, _ = classify(self.firing_rate(x), self.AFR)
        return cl                                                # (N,)
