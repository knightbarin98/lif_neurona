from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo archivos PNG
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import CLASS_NAMES, LIF_CONFIG, LIFConfig  # noqa: E402
from lif import LIFNeuron  # noqa: E402

# Paleta categórica (primeros 3 espacios validados para daltonismo entre sí)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # azul, naranja, aqua
TRAIN_COLOR, TEST_COLOR = SERIES[0], SERIES[1]
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
CLASS_BAND = ["#eef4fc", "#fdf0ea", "#eaf7f1"]       # fondo muy claro por clase verdadera
N_CLASSES = 3
ITERATION_LABEL = "Generación"
ITERATIONS_LABEL = "Generaciones"
TRAINER_LABEL = "DE"
FITNESS_LABEL = "F = 1/distancia(AFR) + Σ SDFR_k"
SAMPLE_LABEL = "Muestra del conjunto de prueba (en el orden del fold: agrupadas por clase)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.6, "figure.dpi": 110,
})


def display_names(class_names: list) -> list:
    return [c.replace("Iris-", "") for c in class_names]    # "Iris-setosa" -> "setosa" en las gráficas


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _np(t):
    return t.detach().cpu().double().numpy() if isinstance(t, torch.Tensor) else t


def plot_error_curve(generations, errors, title: str, color: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(_np(generations), _np(errors), color=color)
    ax.set_xlabel(ITERATION_LABEL)
    ax.set_ylabel(f"{FITNESS_LABEL}  (log)")
    ax.set_title(title)
    _save(fig, path)


def plot_confusion_matrix(cm, class_names: list, title: str, path: str) -> None:
    cm = _np(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=max(cm.max(), 1))
    ax.grid(False)
    n = len(class_names)
    for i in range(n):
        row_total = cm[i].sum()
        for j in range(n):
            v = cm[i, j]
            pct = f"\n{100 * v / row_total:.0f}%" if row_total else ""
            ax.text(j, i, f"{int(v)}{pct}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.55 * cm.max() else TEXT)
    ax.set_xticks(range(n), class_names)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def _class_bands(ax, y_true, class_names: list) -> None:
    y = list(_np(y_true).astype(int))
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            ax.axvspan(start - 0.5, i - 0.5, color=CLASS_BAND[y[start]], zorder=0, linewidth=0)
            ax.text((start + i - 1) / 2, 1.0, class_names[y[start]], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXT_2)
            start = i


def _misclassified(ax, values_at_prediction, y_true, y_pred) -> None:
    # marca con una x cada muestra mal clasificada, a la altura del valor que decidió la predicción
    yt, yp = _np(y_true).astype(int), _np(y_pred).astype(int)
    wrong = [n for n in range(len(yt)) if yt[n] != yp[n]]
    if wrong:
        ax.scatter(wrong, [values_at_prediction(n, yp[n]) for n in wrong], marker="x", s=55, color=TEXT, zorder=5,
                   label=f"mal clasificada ({len(wrong)})")


def plot_neuron_responses(values, y_true, y_pred, class_names: list, title: str, ylabel: str,
                          label_format: str, path: str, target_lines: bool = True) -> None:
    v = _np(values)
    N = v.shape[0]
    x = list(range(N))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _class_bands(ax, y_true, class_names)
    for k in range(N_CLASSES):
        ax.plot(x, v[:, k], color=SERIES[k], marker="o", markersize=3, linewidth=1.2,
                label=f"{label_format.format(k=k)}  ({class_names[k]})")
    if y_pred is not None:
        _misclassified(ax, lambda n, k: v[n, k], y_true, y_pred)
    if target_lines:
        for t in (0.0, 1.0):
            ax.axhline(t, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xlabel(SAMPLE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, path)


def plot_membrane_potential(V, y_true, class_names: list, title: str, path: str,
                            cfg: LIFConfig = LIF_CONFIG) -> None:
    Vn = _np(V)                                       # (N×(pasos+1)): V en t = 0, h, ..., t
    N = Vn.shape[0]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(Vn.T, cmap=SEQUENTIAL, vmin=min(cfg.V_0, cfg.c), vmax=cfg.V_th, aspect="auto",
                   interpolation="nearest", origin="lower", extent=(-0.5, N - 0.5, 0, cfg.t))
    ax.grid(False)
    y = list(_np(y_true).astype(int))
    start = 0
    for i in range(1, N + 1):
        if i == N or y[i] != y[start]:
            if i < N:
                ax.axvline(i - 0.5, color=SURFACE, linewidth=2)
            ax.text((start + i - 1) / 2, 1.0, class_names[y[start]], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXT_2)
            start = i
    ax.set_ylabel("t (ms)")
    ax.set_xlabel(SAMPLE_LABEL)
    ax.set_title(title, pad=16)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="V")
    _save(fig, path)


def plot_firing_rates(F_r, y_true, y_pred, AFR, SDFR, class_names: list, title: str, path: str) -> None:
    fr, afr, sdfr = _np(F_r), _np(AFR), _np(SDFR)
    yt = _np(y_true).astype(int)
    N = fr.shape[0]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _class_bands(ax, y_true, class_names)
    for k in range(N_CLASSES):
        # AFR_k ± SDFR_k del conjunto de entrenamiento
        ax.axhspan(afr[k] - sdfr[k], afr[k] + sdfr[k], color=SERIES[k], alpha=0.12, linewidth=0, zorder=1)
        ax.axhline(afr[k], color=SERIES[k], linestyle="--", linewidth=1.1, zorder=2,
                   label=f"AFR_{k} ± SDFR_{k}  ({class_names[k]})")
        idx = [n for n in range(N) if yt[n] == k]
        ax.scatter(idx, fr[idx], s=14, color=SERIES[k], zorder=3)
    _misclassified(ax, lambda n, k: fr[n], y_true, y_pred)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xlabel(SAMPLE_LABEL)
    ax.set_ylabel("F_r (disparos / ms)")
    ax.set_title(title, pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, path)


def plot_accuracy_distribution(values, title: str, path: str) -> None:
    v = [round(float(a), 6) for a in _np(values)]
    uniq = sorted(set(v))
    counts = [v.count(u) for u in uniq]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar([100 * u for u in uniq], counts, width=0.9, color=SERIES[0])
    ax.set_xlabel("Accuracy de prueba (%)")
    ax.set_ylabel("Número de subexperimentos")
    ax.set_title(title)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _save(fig, path)


def plot_experiment_accuracies(acc_ab, acc_ba, acc_mean, title: str, path: str) -> None:
    ids = list(range(1, len(acc_ab) + 1))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.vlines(ids, [min(a, b) * 100 for a, b in zip(acc_ab, acc_ba)],
              [max(a, b) * 100 for a, b in zip(acc_ab, acc_ba)], color=GRID, linewidth=3, zorder=1)
    ax.scatter(ids, [a * 100 for a in acc_ab], s=26, color=SERIES[0], label="A → B", zorder=3)
    ax.scatter(ids, [a * 100 for a in acc_ba], s=26, color=SERIES[1], label="B → A", zorder=3)
    ax.scatter(ids, [a * 100 for a in acc_mean], s=46, marker="_", color=TEXT, linewidth=2,
               label="promedio del experimento", zorder=4)
    ax.set_xticks(ids)
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Accuracy de prueba (%)")
    ax.set_title(title, pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)   # fuera del eje: no tapa los puntos
    _save(fig, path)


def plot_all_error_curves(curves, title: str, ylabel: str, color: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for c in curves:
        ax.semilogy(range(len(c)), _np(c), color=color, alpha=0.25, linewidth=0.9)   # generación 0 ... generaciones
    ax.set_xlabel(ITERATION_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def plot_histogram(values, title: str, xlabel: str, path: str, color: str | None = None) -> None:
    v = [float(a) for a in _np(values)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(v, bins=min(20, max(5, len(set(v)))), color=color or SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    _save(fig, path)


def plot_subexperiment(sub: dict, title: str, x: torch.Tensor, y_idx: torch.Tensor, plot_dir: str) -> None:
    names = display_names(CLASS_NAMES)
    neuron = LIFNeuron(w=sub["w"], AFR=sub["AFR"], SDFR=sub["SDFR"])
    x_tilde = x[sub["test_idx"]]                      # (75×4) en el orden del fold
    fitness_history = sub["fitness_history"]
    ev = sub["eval"]

    plot_error_curve(range(len(fitness_history)), fitness_history,
                     f"{title} · fitness de entrenamiento ({TRAINER_LABEL})",
                     TRAIN_COLOR, os.path.join(plot_dir, "01_train_fitness.png"))
    plot_confusion_matrix(sub["confusion_matrix"], names, f"{title} · prueba",
                          os.path.join(plot_dir, "02_confusion_test.png"))
    plot_neuron_responses(ev["distances"], ev["y_true"], ev["y_pred"], names,
                          f"{title} · distancia de F_r a la AFR de cada clase (prueba; cl = argmin)",
                          "|AFR_k − F_r|", "|AFR_{k} − F_r|",
                          os.path.join(plot_dir, "03_class_distances_test.png"), target_lines=False)
    plot_membrane_potential(neuron.membrane_potential(x_tilde), ev["y_true"], names,
                            f"{title} · potencial de membrana V(t) de la neurona LIF (prueba)",
                            os.path.join(plot_dir, "04_membrane_potential_test.png"), neuron.cfg)
    plot_firing_rates(ev["F_r"], ev["y_true"], ev["y_pred"], sub["AFR"], sub["SDFR"], names,
                      f"{title} · razón de disparo F_r frente a AFR ± SDFR de entrenamiento (prueba)",
                      os.path.join(plot_dir, "05_firing_rate_test.png"))


def plot_global(sub_rows: list, exp_rows: list, cm_global: torch.Tensor, curves: list, plot_dir: str) -> None:
    n = len(sub_rows)
    names = display_names(CLASS_NAMES)
    plot_accuracy_distribution([r["test_accuracy"] for r in sub_rows],
                               f"Accuracy de prueba de los {n} subexperimentos",
                               os.path.join(plot_dir, "test_accuracy_subexperiments.png"))
    plot_experiment_accuracies([e["test_accuracy_A_to_B"] for e in exp_rows],
                               [e["test_accuracy_B_to_A"] for e in exp_rows],
                               [e["test_accuracy_mean"] for e in exp_rows],
                               "Accuracy de prueba por experimento", os.path.join(plot_dir, "experiments_accuracy.png"))
    plot_confusion_matrix(cm_global, names, f"Prueba · suma de los {n} subexperimentos",
                          os.path.join(plot_dir, "confusion_test_global.png"))
    plot_all_error_curves(curves, f"Fitness de entrenamiento ({TRAINER_LABEL}) · {n} subexperimentos",
                          "F entrenamiento (log)",
                          TRAIN_COLOR, os.path.join(plot_dir, "train_fitness_curves.png"))
    plot_histogram([r["generations_executed"] for r in sub_rows], f"{ITERATIONS_LABEL} registradas", ITERATIONS_LABEL,
                   os.path.join(plot_dir, "generations_hist.png"))
    plot_histogram([r["final_training_fitness"] for r in sub_rows], "Fitness final de entrenamiento",
                   "F final", os.path.join(plot_dir, "final_fitness_hist.png"), color=SERIES[1])
