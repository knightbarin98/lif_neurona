from __future__ import annotations

import torch

from lif import LIFNeuron, classify
from training import test_fitness_history, train_subexperiment
from utils.metrics import classification_metrics
from utils.statistics import describe

def evaluate(neuron: LIFNeuron, x_tilde: torch.Tensor, y_idx_test: torch.Tensor) -> dict:
    """
    Etapa de prueba: I y F_r de cada vector x̃, cl = argmin_k |AFR_k − F_r|. Devuelve dict con predicciones.
    """
    I = neuron.current(x_tilde)                       # 75
    F_r = neuron.firing_rate(x_tilde)                 # 75
    cl, distances = classify(F_r, neuron.AFR)         # 75, 75×3
    accuracy = (cl == y_idx_test).float().mean().item()
    return {
        "y_true": y_idx_test.cpu(),
        "y_pred": cl.cpu(),
        "I": I.cpu(),
        "F_r": F_r.cpu(),
        "distances": distances.cpu(),                 # 75×3, |AFR_k − F_r| para guardar dist_AFR_0..2
        "accuracy": accuracy,
    }

def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        confusion_matrix[t, p] += 1
    return confusion_matrix

def create_two_folds(y_idx: torch.Tensor, seed: int, device: torch.device, per_class: int = 25,
                     num_classes: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    y_cpu = y_idx.cpu()
    part_A, part_B = [], []
    for c in range(num_classes):
        cls_idx = (y_cpu == c).nonzero(as_tuple=True)[0] #indices de la clase c
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=gen)] #number of elements
        part_A.append(perm[:per_class])
        part_B.append(perm[per_class: 2*per_class])
    idx_A = torch.cat(part_A).to(device)
    idx_B = torch.cat(part_B).to(device)
    return idx_A, idx_B

def check_folds(idx_A: torch.Tensor, idx_B: torch.Tensor, y_idx: torch.Tensor, N: int = 150,
                per_class: int = 25, num_classes: int = 3) -> None:
    a, b = set(idx_A.tolist()), set(idx_B.tolist())
    assert len(a) == N // 2 and len(b) == N // 2, (len(a), len(b))
    assert a.isdisjoint(b), "A ∩ B != vacío"
    assert a | b == set(range(N)), "A ∪ B != todas"
    # estratificación: 25 de cada clase en cada fold
    y_cpu = y_idx.cpu()
    for name, idx in (("A", idx_A), ("B", idx_B)):
        counts = torch.bincount(y_cpu[idx.cpu()], minlength=num_classes)
        assert torch.all(counts == per_class), f"fold {name} no estratificado: {counts.tolist()}"

def run_subexperiment(x: torch.Tensor, y_idx: torch.Tensor, train_idx: torch.Tensor,
                      test_idx: torch.Tensor, seed: int, device: torch.device) -> dict:
    x_train, x_tilde = x[train_idx], x[test_idx]
    y_train, y_test = y_idx[train_idx], y_idx[test_idx]

    neuron, fitness_history, de_result = train_subexperiment(x_train, y_train, seed, device)
    test_fitness = test_fitness_history(x_tilde, y_test, de_result.best_position_history)

    ev = evaluate(neuron, x_tilde, y_test)
    train_accuracy = evaluate(neuron, x_train, y_train)["accuracy"]
    confussion_matrix = compute_confusion_matrix(ev["y_true"], ev["y_pred"])
    metrics = classification_metrics(confussion_matrix)

    return {
        "seed": seed,
        "fitness_history": fitness_history,
        "test_fitness_history": test_fitness,
        "train_fitness_statistics": describe(fitness_history),
        "test_fitness_statistics": describe(test_fitness),
        "generations_executed": len(fitness_history),
        "final_training_fitness": fitness_history[-1],
        "final_test_fitness": test_fitness[-1],
        "cumulative_training_fitness": float(sum(fitness_history)),
        "mean_training_fitness": float(sum(fitness_history) / len(fitness_history)),
        "train_accuracy": train_accuracy,
        "eval": ev,
        "confusion_matrix": confussion_matrix,
        "metrics": metrics,
        "train_idx": train_idx.cpu(), "test_idx": test_idx.cpu(),
        # mejor-modelo / reproducción: W̄ y las AFR, SDFR de entrenamiento que usa la etapa de prueba
        "w": neuron.w.detach().cpu(), "AFR": neuron.AFR.detach().cpu(), "SDFR": neuron.SDFR.detach().cpu(),
        "de_result": de_result,                         # OptimizationResult del DE
    }

def run_experiment(x: torch.Tensor, y_idx: torch.Tensor, experiment_id: int, base_seed: int,
                   device: torch.device) -> dict:
    idx_A, idx_B = create_two_folds(y_idx, base_seed, device)
    check_folds(idx_A, idx_B, y_idx, x.shape[0])

    subexperiment_1 = run_subexperiment(x, y_idx, idx_A, idx_B, base_seed, device)
    subexperiment_2 = run_subexperiment(x, y_idx, idx_B, idx_A, base_seed, device)

    confussion_matrix_experimento = subexperiment_1["confusion_matrix"] + subexperiment_2["confusion_matrix"]
    assert confussion_matrix_experimento.sum().item() == 150, "predicciones OOF != 150"

    return {"experiment_id": experiment_id, "seed": base_seed,
            "idx_A": idx_A.cpu(), "idx_B": idx_B.cpu(),
            "sub1": subexperiment_1, "sub2": subexperiment_2, "confusion_matrix": confussion_matrix_experimento,
            # todas las generaciones de los dos subexperimentos
            "train_fitness_statistics": describe(subexperiment_1["fitness_history"]
                                                 + subexperiment_2["fitness_history"]),
            "test_fitness_statistics": describe(subexperiment_1["test_fitness_history"]
                                                + subexperiment_2["test_fitness_history"])}
