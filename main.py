from __future__ import annotations

import os
import logging
import torch

from config import DEVICE, DATA_CSV, NO_EXPERIMENTS as NUM_EXPERIMENTS, BASE_SEED, RESULTS_DIR, NUM_CLASSES
from data import load_iris_raw, build_tensors
from experiment import run_experiment
from utils.results import save_experiment, sub_summary_row, exp_summary_row
from utils.global_results import save_global, save_best_model
from utils.statistics import fitness_statistics_line

def setup_logger() -> logging.Logger:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = logging.getLogger("iris")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"), mode="w")
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def better(a: dict, b: dict | None) -> bool:
    """True si el subexperimento `a` es mejor que `b` (acc alta, luego fitness baja)."""
    if b is None:
        return True
    ka = (a["metrics"]["accuracy"], -a["final_training_fitness"])
    kb = (b["metrics"]["accuracy"], -b["final_training_fitness"])
    return ka > kb

def run_all_experiments(verbose: bool = True) -> torch.Tensor:
    log = setup_logger()
    feats, labels = load_iris_raw(DATA_CSV)
    x, y_idx = build_tensors(feats, labels, DEVICE)
    x_cpu, y_idx_cpu = x.cpu(), y_idx.cpu()

    log.info("Iris cargado: %s muestras, device=%s", x.shape[0], DEVICE)

    confussion_matrix_global = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    finals, cumulatives, all_generation_fitness, all_generation_test_fitness = [], [], [], []
    sub_rows, exp_rows = [], []
    generations_list, accuracies = [], []
    best = None

    for e in range(1, NUM_EXPERIMENTS + 1):
        experiment = run_experiment(x, y_idx, e, BASE_SEED + e, DEVICE)
        confussion_matrix_global += experiment["confusion_matrix"]

        for sid, (sub, tr, te) in enumerate([(experiment["sub1"], "A", "B"), (experiment["sub2"], "B", "A")], start=1):
            finals.append(sub["final_training_fitness"])
            cumulatives.append(sub["cumulative_training_fitness"])
            all_generation_fitness.extend(sub["fitness_history"])
            all_generation_test_fitness.extend(sub["test_fitness_history"])
            generations_list.append(sub["generations_executed"])
            sub_rows.append(sub_summary_row(e, sid, sub, tr, te))

            # log de estadísticos + métricas por subexperimento
            m = sub["metrics"]
            log.info("exp %02d sub %d | generaciones=%d | final_F=%.6f cum=%.2f mean=%.6f | "
                     "AFR=[%.4f,%.4f,%.4f] SDFR=[%.4f,%.4f,%.4f] | train_acc=%.4f acc=%.4f macroP=%.4f | "
                     "P[set,ver,vir]=[%.3f,%.3f,%.3f]",
                     e, sid, sub["generations_executed"], sub["final_training_fitness"],
                     sub["cumulative_training_fitness"], sub["mean_training_fitness"],
                     *sub["AFR"].tolist(), *sub["SDFR"].tolist(), sub["train_accuracy"],
                     m["accuracy"], m["macro_precision"], *m["precision_per_class"])
            log.info("exp %02d sub %d | %s", e, sid,
                     fitness_statistics_line(sub["train_fitness_statistics"], sub["test_fitness_statistics"]))
            log.info("exp %02d sub %d | resultado DE: %s", e, sid, sub["de_result"])

            # seguimiento del mejor modelo
            if better(sub, best):
                best = dict(sub)
                best["experiment_id"] = e
                best["subexperiment_id"] = sid

        aggregate = save_experiment(experiment, x_cpu, y_idx_cpu)
        exp_rows.append(exp_summary_row(experiment, aggregate))
        accuracies.append(aggregate["accuracy"])
        if verbose:
            log.info("== experimento %02d | accuracy OOF=%.4f | %s", e, aggregate["accuracy"],
                     fitness_statistics_line(experiment["train_fitness_statistics"],
                                             experiment["test_fitness_statistics"]))

    # validaciones globales
    assert len(sub_rows) == 2 * NUM_EXPERIMENTS
    total = confussion_matrix_global.sum().item()
    assert total == 150 * NUM_EXPERIMENTS, total

    stats = save_global(confussion_matrix_global, finals, cumulatives, all_generation_fitness,
                        all_generation_test_fitness, sub_rows, exp_rows)
    best_dir = save_best_model(best, x_cpu, y_idx_cpu)

    log.info("===== RESUMEN GLOBAL =====")
    log.info("todos los experimentos | %s",
             fitness_statistics_line(stats["all_generations_training_fitness"], stats["all_generations_test_fitness"]))
    log.info("CM global (filas=real, cols=pred):\n%s", confussion_matrix_global)
    log.info("total predicciones=%d | accuracy global=%.4f",
             total, confussion_matrix_global.diag().sum().item() / total)
    log.info("accuracy media por experimento=%.4f", sum(accuracies) / len(accuracies))
    log.info("generaciones: min=%d max=%d media=%.0f",
             min(generations_list), max(generations_list), sum(generations_list) / len(generations_list))
    log.info("MEJOR modelo: exp %02d sub %d | acc=%.4f macroP=%.4f -> %s",
             best["experiment_id"], best["subexperiment_id"],
             best["metrics"]["accuracy"], best["metrics"]["macro_precision"], best_dir)
    return confussion_matrix_global




if __name__ == "__main__":
    run_all_experiments()
