from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


# Conteo rápido: con I constante no hace falta recorrer los t/h pasos por neurona.
# Antes del primer disparo, la recurrencia de Euler V_{n+1} = q V_n + h (I + a), con q = 1 − h b, da
#     V_n = A_n I + B_n,   A_{n+1} = q A_n + h,   B_{n+1} = q B_n + h a,   A_0 = 0,   B_0 = V de arranque
# y la neurona ya disparó en el paso n si V_n ≥ V_th  <=>  I ≥ I_min(n) = (V_th − B_n) / A_n.
# A_n y B_n no dependen de x̄ ni de w̄: las tablas I_min se calculan una sola vez para todas las neuronas.
# Tras cada disparo V = c y, como I no cambia, todos los intervalos siguientes duran lo mismo.


@lru_cache(maxsize=None)
def umbrales_de_corriente(V_inicio: float, cfg: LIFConfig = LIF_CONFIG,
                          device: torch.device | None = None) -> torch.Tensor:
    """I_min(n) para n = 1, ..., t/h: corriente mínima para alcanzar V_th en el paso n arrancando en V_inicio."""
    q = 1.0 - cfg.h * cfg.b
    A_n, B_n = 0.0, float(V_inicio)
    I_min = []
    for _ in range(simulation_steps(cfg)):
        A_n = q * A_n + cfg.h
        B_n = q * B_n + cfg.h * cfg.a
        I_min.append((cfg.V_th - B_n) / A_n)
    I_min = torch.tensor(I_min, dtype=torch.float64, device=device)      # (pasos,)   ej. (3000,)
    if not bool((I_min[1:] < I_min[:-1]).all()):
        raise ValueError("El conteo rápido requiere I_min(n) estrictamente decreciente con estos parámetros LIF")
    return I_min


def contar_disparos(I: torch.Tensor, cfg: LIFConfig = LIF_CONFIG) -> torch.Tensor:
    """
        Total de disparos en t ms (misma forma que I), igual que simulate pero sin el ciclo de Euler.
        t_1: paso del primer disparo (arranca en V_0); T_p: pasos entre disparos (arranca en c).
        disparos = 1 + ⌊(N − t_1) / T_p⌋ si t_1 ≤ N, si no 0, con N = t/h.
    """
    N = simulation_steps(cfg)
    I_min_inicio = umbrales_de_corriente(cfg.V_0, cfg, I.device)          # desde V_0 = −60
    I_min_reinicio = umbrales_de_corriente(cfg.c, cfg, I.device)          # desde c = −50
    menos_I = -I.to(torch.float64)
    # primer n con I ≥ I_min(n): búsqueda binaria en −I_min, que es creciente
    t_1 = torch.searchsorted(-I_min_inicio, menos_I) + 1                   # (forma de I), N + 1 = nunca dispara
    T_p = torch.searchsorted(-I_min_reinicio, menos_I) + 1
    disparos = torch.where(t_1 <= N, 1 + (N - t_1.clamp(max=N)) // T_p, torch.zeros_like(t_1))
    return disparos.to(I.dtype)


def firing_rate(I: torch.Tensor, cfg: LIFConfig = LIF_CONFIG) -> torch.Tensor:
    """FR = total de disparos / tiempo de simulación (disparos por ms)."""
    return contar_disparos(I, cfg) / cfg.t


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
