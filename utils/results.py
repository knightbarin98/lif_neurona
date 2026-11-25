from __future__ import annotations

import os
from dataclasses import asdict

import torch

from config import CLASS_NAMES, DE_CONFIG, LIF_CONFIG, RESULTS_DIR, SAVE_SUBEXPERIMENT_PLOTS
from fitness import FITNESS_FORMULA
from .logging_utils import write_confusion_csv, write_csv, write_generations_csv, write_json
from .metrics import class_counts, classification_metrics
from .statistics import describe
from .visualization import plot_subexperiment

# sub1 entrena con A y prueba con B; sub2 al revés
SUBEXPERIMENTS = (("sub1", "A_to_B"), ("sub2", "B_to_A"))
N_CLASSES = len(CLASS_NAMES)


def subexperiment_title(experiment_id: int, name: str) -> str:
    return f"Exp {experiment_id:02d} · {name.replace('_to_', '→')}"


def subexperiment_summary(sub: dict, y_idx: torch.Tensor) -> dict:
    fitness_history = sub["fitness_history"]
    return {
        "seed": sub["seed"],
        "train_class_counts": class_counts(y_idx[sub["train_idx"]], N_CLASSES),
        "test_class_counts": class_counts(y_idx[sub["test_idx"]], N_CLASSES),
        "lif": asdict(LIF_CONFIG),
        "training": {
            "de_strategy": "DE/rand/1/bin",
            "fitness": FITNESS_FORMULA,
            "de_config": asdict(DE_CONFIG),
            "generations_executed": sub["generations_executed"],     # generación 0 ... generaciones
            "initial_training_fitness": fitness_history[0],
            "final_training_fitness": sub["final_training_fitness"],
            "cumulative_training_fitness": sub["cumulative_training_fitness"],
            "mean_training_fitness": sub["mean_training_fitness"],
            "train_accuracy": sub["train_accuracy"],
            "w_final": sub["w"],
            "AFR": sub["AFR"],
            "SDFR": sub["SDFR"],
        },
        "final": {"test_accuracy": sub["eval"]["accuracy"], "test_fitness": sub["final_test_fitness"]},
        "confusion_matrix_test": sub["confusion_matrix"],
        "metrics_test": sub["metrics"],
        "train_fitness_statistics": sub["train_fitness_statistics"],
        "test_fitness_statistics": sub["test_fitness_statistics"],
    }


def save_subexperiment(sub: dict, title: str, x: torch.Tensor, y_idx: torch.Tensor, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    write_generations_csv(os.path.join(out_dir, "generations.csv"), sub["fitness_history"], sub["test_fitness_history"])
    write_json(os.path.join(out_dir, "summary.json"), subexperiment_summary(sub, y_idx))

    torch.save({"w": sub["w"], "AFR": sub["AFR"], "SDFR": sub["SDFR"]}, os.path.join(out_dir, "weights.pt"))
    write_confusion_csv(os.path.join(out_dir, "confusion_matrix_test.csv"), sub["confusion_matrix"], CLASS_NAMES)

    # corriente, razón de disparo y distancia a la AFR de cada clase de cada flor de prueba
    test_idx = sub["test_idx"]
    ev = sub["eval"]
    rows = [[int(test_idx[n]), int(ev["y_true"][n]), int(ev["y_pred"][n]), float(ev["I"][n]), float(ev["F_r"][n])]
            + ev["distances"][n].tolist()
            for n in range(test_idx.numel())]
    write_csv(os.path.join(out_dir, "test_outputs.csv"),
              ["original_index", "true_class", "predicted_class", "I", "F_r"]
              + [f"dist_AFR_{k}" for k in range(N_CLASSES)], rows)

    if SAVE_SUBEXPERIMENT_PLOTS:
        plot_subexperiment(sub, title, x, y_idx, os.path.join(out_dir, "plots"))


def experiment_aggregate(experiment: dict) -> dict:
    cm = experiment["confusion_matrix"]
    metrics = classification_metrics(cm)
    return {"accuracy": metrics["accuracy"], "metrics": metrics, "confusion_matrix": cm}


def experiment_summary(experiment: dict, aggregate: dict) -> dict:
    s1, s2 = experiment["sub1"], experiment["sub2"]
    acc = [s1["metrics"]["accuracy"], s2["metrics"]["accuracy"]]
    return {
        "experiment_id": experiment["experiment_id"],
        "experiment_seed": experiment["seed"],
        "test_accuracy_A_to_B": acc[0],
        "test_accuracy_B_to_A": acc[1],
        "test_accuracy_mean": sum(acc) / 2,
        "test_accuracy_aggregated": aggregate["accuracy"],
        "test_accuracy_statistics": describe(acc),
        "train_accuracy": {"A_to_B": s1["train_accuracy"], "B_to_A": s2["train_accuracy"]},
        "train_fitness_statistics": experiment["train_fitness_statistics"],
        "test_fitness_statistics": experiment["test_fitness_statistics"],
        "aggregated_confusion_matrix_test": aggregate["confusion_matrix"],
        "aggregated_metrics_test": aggregate["metrics"],
        "generations": {"A_to_B": s1["generations_executed"], "B_to_A": s2["generations_executed"]},
    }


def save_experiment(experiment: dict, x: torch.Tensor, y_idx: torch.Tensor) -> dict:
    exp_id, seed = experiment["experiment_id"], experiment["seed"]
    exp_dir = os.path.join(RESULTS_DIR, f"experiment_{exp_id:02d}")
    os.makedirs(exp_dir, exist_ok=True)

    write_json(os.path.join(exp_dir, "experiment.json"), {
        "experiment_id": exp_id,
        "seed": seed,
        "A_indices": experiment["idx_A"],
        "B_indices": experiment["idx_B"],
        "A_class_counts": class_counts(y_idx[experiment["idx_A"]], N_CLASSES),
        "B_class_counts": class_counts(y_idx[experiment["idx_B"]], N_CLASSES),
    })
    in_A = set(experiment["idx_A"].tolist())
    write_csv(os.path.join(exp_dir, "split_indices.csv"), ["original_index", "set", "class"],
              [[i, "A" if i in in_A else "B", int(y_idx[i])] for i in range(x.shape[0])])

    aggregate = experiment_aggregate(experiment)
    write_json(os.path.join(exp_dir, "experiment_summary.json"), experiment_summary(experiment, aggregate))
    write_confusion_csv(os.path.join(exp_dir, "confusion_matrix_test_aggregated.csv"),
                        aggregate["confusion_matrix"], CLASS_NAMES)

    for key, name in SUBEXPERIMENTS:
        save_subexperiment(experiment[key], subexperiment_title(exp_id, name), x, y_idx,
                           os.path.join(exp_dir, f"subexperiment_{name}"))
    return aggregate


def fitness_statistics_columns(train_stats: dict, test_stats: dict) -> dict:
    return {"train_fitness_mean": train_stats["mean"], "train_fitness_median": train_stats["median"],
            "train_fitness_std": train_stats["std"], "test_fitness_mean": test_stats["mean"],
            "test_fitness_median": test_stats["median"], "test_fitness_std": test_stats["std"]}


def sub_summary_row(experiment_id: int, subexperiment_id: int, sub: dict, train_set: str, test_set: str) -> dict:
    m = sub["metrics"]
    row = {
        "experiment_id": experiment_id, "subexperiment_id": subexperiment_id,
        "subexperiment": f"{train_set}_to_{test_set}", "seed": sub["seed"],
        "generations_executed": sub["generations_executed"],
        "final_training_fitness": sub["final_training_fitness"],
        "cumulative_training_fitness": sub["cumulative_training_fitness"],
        "mean_training_fitness": sub["mean_training_fitness"],
        "final_test_fitness": sub["final_test_fitness"],
        **fitness_statistics_columns(sub["train_fitness_statistics"], sub["test_fitness_statistics"]),
        "train_accuracy": sub["train_accuracy"],
        "test_accuracy": m["accuracy"], "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
    }
    for k, c in enumerate(CLASS_NAMES):
        row[f"precision_{c}"] = m["precision_per_class"][k]
    for k, c in enumerate(CLASS_NAMES):
        row[f"AFR_{c}"] = float(sub["AFR"][k])
    for k, c in enumerate(CLASS_NAMES):
        row[f"SDFR_{c}"] = float(sub["SDFR"][k])
    return row


def exp_summary_row(experiment: dict, aggregate: dict) -> dict:
    acc_ab = experiment["sub1"]["metrics"]["accuracy"]
    acc_ba = experiment["sub2"]["metrics"]["accuracy"]
    return {
        "experiment_id": experiment["experiment_id"], "seed": experiment["seed"],
        "test_accuracy_A_to_B": acc_ab, "test_accuracy_B_to_A": acc_ba,
        "test_accuracy_mean": (acc_ab + acc_ba) / 2, "test_accuracy_aggregated": aggregate["accuracy"],
        **fitness_statistics_columns(experiment["train_fitness_statistics"], experiment["test_fitness_statistics"]),
    }
