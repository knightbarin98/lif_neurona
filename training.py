from __future__ import annotations

import math

import torch

from algorithm import DifferentialEvolution, OptimizationResult
from config import DE_CONFIG, DE_LOWER_BOUND, DE_UPPER_BOUND, LIF_CONFIG, DEConfig, LIFConfig
from fitness import fitness_population
from lif import LIFNeuron

# Etapa de entrenamiento: une el modelo LIF (lif.py, fitness.py) con la evolución diferencial (algorithm/).
# El DE solo ve una función objetivo (NP×n) -> (NP,); no sabe nada de la neurona.


def train_subexperiment(x_train: torch.Tensor, y_train: torch.Tensor, seed: int, device: torch.device,
                        de_config: DEConfig = DE_CONFIG,
                        lif_config: LIFConfig = LIF_CONFIG) -> tuple[LIFNeuron, list[float], OptimizationResult]:
    """
        Calibra w̄ con DE/rand/1/bin desde una población nueva (`seed`) de NP vectores en [0, 1].
        Devuelve la neurona LIF (W̄ optimizado + AFR, SDFR de entrenamiento), el mejor F_i por generación
        (generación 0 = población inicial ... generaciones) y el OptimizationResult del DE.
    """
    rng = torch.Generator(device=device).manual_seed(seed)

    def objective(w: torch.Tensor) -> torch.Tensor:
        return fitness_population(x_train, y_train, w, lif_config)     # (NP×n) -> F_i (NP,)

    de = DifferentialEvolution(objective, DE_LOWER_BOUND, DE_UPPER_BOUND, de_config, rng, dtype=x_train.dtype)
    result = de.run()
    W = result.best_position                                            # W̄: pesos optimizados (n,)
    neuron = LIFNeuron.fit(W, x_train, y_train, lif_config)

    # el fitness del mejor vector, recalculado con W̄ solo, debe coincidir con el que registró el DE
    F_i = float(fitness_population(x_train, y_train, W[None, :], lif_config)[0])
    assert math.isclose(F_i, result.best_fitness, rel_tol=1e-5), (F_i, result.best_fitness)
    return neuron, result.fitness_history, result


def test_fitness_history(x_test: torch.Tensor, y_test: torch.Tensor, best_position_history: list[torch.Tensor],
                         lif_config: LIFConfig = LIF_CONFIG, chunk_size: int = 500) -> list[float]:
    """
        F_i sobre el conjunto de prueba del mejor w̄ de cada generación, calculado al terminar:
        el DE nunca ve el conjunto de prueba. Solo se simula cada vector distinto una vez.
    """
    W_history = torch.stack(best_position_history)                      # (G+1×n) en CPU
    unique, inverse = torch.unique(W_history, dim=0, return_inverse=True)
    F_unique = torch.cat([fitness_population(x_test, y_test, chunk.to(x_test.device), lif_config).cpu()
                          for chunk in torch.split(unique, chunk_size)])
    return F_unique[inverse].tolist()
