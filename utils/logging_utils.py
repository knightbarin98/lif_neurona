from __future__ import annotations

import csv
import gzip
import json
import math
import os

import torch

FLOAT_FMT = "{:.10g}"
GENERATION_COLUMNS = ["generation", "train_fitness", "test_fitness"]


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else FLOAT_FMT.format(v)
    return v


def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_csv(path: str, header: list, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([_fmt(v) for v in row])


def write_generations_csv(path: str, fitness_history: list, test_fitness_history: list) -> None:
    # la generación 0 es el mejor vector de pesos de la población inicial
    write_csv(path, GENERATION_COLUMNS, [[g, F_i, test_F_i] for g, (F_i, test_F_i)
                                         in enumerate(zip(fitness_history, test_fitness_history))])


def write_confusion_csv(path: str, cm: torch.Tensor, class_names: list) -> None:
    rows = [[f"real_{class_names[i]}"] + [int(v) for v in cm[i].tolist()] for i in range(cm.shape[0])]
    write_csv(path, ["real \\ predicha"] + [f"pred_{c}" for c in class_names], rows)


class GlobalGenerationsWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = gzip.open(path, "wt", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["experiment_id", "subexperiment_id", "train_set", "test_set"] + GENERATION_COLUMNS)
        self.rows_written = 0

    def append(self, experiment_id: int, subexperiment_id: int, train_set: str, test_set: str,
               fitness_history: list, test_fitness_history: list) -> None:
        for g, (F_i, test_F_i) in enumerate(zip(fitness_history, test_fitness_history)):
            self._w.writerow([experiment_id, subexperiment_id, train_set, test_set, g, _fmt(F_i), _fmt(test_F_i)])
        self.rows_written += len(fitness_history)

    def close(self) -> None:
        self._f.close()
