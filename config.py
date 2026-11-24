from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

import torch

NO_EXPERIMENTS: int = 35
K_FOLDS: int = 2
TRAIN_SIZE: int = 75            # 50% entrenamiento
TEST_SIZE: int = 75             # 50% prueba
TOTAL_SAMPLE: int = 150

INPUT_FEATURE: int = 4          # n: número de características de Iris Plant
NUM_CLASSES: int = 3            # total de clases k = 1, ..., 3

MODE_DECIMALS: int = 6
STD_POPULATION: bool = True

BASE_SEED: int = 42

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parámetros del modelo LIF (notas; artículo de Vazquez y Cachón, 2010)
@dataclass(frozen=True)
class LIFConfig:
    theta: float = 0.1          # θ: factor de ganancia de I = x̄·w̄·θ
    a: float = 0.5              # constante del modelo dV/dt = I + a − bV
    b: float = -0.001          # constante del modelo
    c: float = -50.0            # voltaje de estado de reposo (reinicio tras un disparo)
    h: float = 1.0              # incremento de tiempo del método de Euler (ms)
    V_th: float = 50.0          # voltaje de umbral
    V_0: float = -60.0          # V_{i=0}: voltaje inicial en t = 0
    t: float = 1000.0           # tiempo de simulación (ms)


LIF_CONFIG: LIFConfig = LIFConfig()

# los vectores de pesos w̄ se inicializan y se buscan en [-1, 1] (distribución uniforme)
DE_LOWER_BOUND: float = -1.0
DE_UPPER_BOUND: float = 1.0


# evolución diferencial DE/rand/1/bin que calibra w̄
@dataclass(frozen=True)
class DEConfig:
    NP: int = 60                        # tamaño de población: 60 vectores de pesos
    dimension: int = INPUT_FEATURE      # 4 características de Iris Plant
    F: float = 0.9
    CR: float = 0.8
    generaciones: int = 1000            # notas: 10000; el fitness se estanca hacia la generación 20 (artículo: 1000)


DE_CONFIG: DEConfig = DEConfig()

# función fitness que minimiza el DE (fitness.py):
#   "ec5"    ec. 5 del artículo: 1/distancia(AFR) + Σ SDFR_k
#   "fisher" 1 / min J_kl, J_kl = |AFR_k − AFR_l| / (SDFR_k + SDFR_l + ε) entre clases vecinas
#   "error"  error de entrenamiento + LAMBDA_FITNESS · (1 / min J_kl) para desempatar
FITNESS: str = "fisher"
LAMBDA_FITNESS: float = 0.001

DATA_CSV: str = "iris.csv"
RESULTS_DIR: str = f"resultados_fitness_{FITNESS}"     # main usa "resultados" (ec. 5)
SAVE_SUBEXPERIMENT_PLOTS: bool = True

CLASS_NAMES: list[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
FEATURE_NAMES: list[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def to_dict() -> dict:
    # todas las constantes en MAYÚSCULAS de este módulo (para guardarlas junto a los resultados)
    def plain(v):
        if isinstance(v, torch.device):
            return str(v)
        return asdict(v) if is_dataclass(v) else v
    return {k: plain(v) for k, v in globals().items() if k.isupper()}
