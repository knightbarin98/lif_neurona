# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and running

Conda env `iris_nn` (Python 3.11, PyTorch 2.9.1, numpy/pandas/matplotlib):

```bash
conda env create -f environment-mac.yml   # macOS, CPU only
conda env create -f environment.yml       # Linux, CUDA 12.8 (CPU alternative commented inside)
conda activate iris_nn
python main.py                            # run from repo root: DATA_CSV/RESULTS_DIR are relative paths
```

Results go to `results_lif/`. There is no test suite, linter, or build step. Correctness checks are `assert`s inside the pipeline (fold stratification, prediction counts, best fitness == fitness of the rebuilt neuron).

## What this project is

A single **Leaky Integrate-and-Fire neuron** classifying Iris (`iris.csv`, 150 rows, header row), with its synaptic weights calibrated by **Differential Evolution**, following Vazquez & Cachón, "Integrate and Fire Neurons and their Application in Pattern Recognition" (CCE 2010) and the user's notes (`~/Documents/LIF Notes.md`). The notes win over the paper where they differ (e.g. `b = -0.0001`, `NP = 60`, `F = 0.9`, `CR = 0.8`), except `generaciones = 1000` (paper's MAXGEN; the notes say 10000, but the best fitness stops improving around generation 20 and 10000 gave the identical final fitness at 10× the time, so the user chose 1000). The notes say both `t = 100` and `t = 1000` ms; `t = 1000` is used (paper's `T = 1000`; with 100 ms the neuron fires 0–1 times and can't separate classes).

Variable names follow the notes on purpose (`x`, `w`, `I`, `theta`, `a`, `b`, `c`, `h`, `V_th`, `V_0`, `t`, `V_i`, `F(t_i, V_i)`, `FR`, `AFR`, `SDFR`, `distancia_AFR`, `F_i`, `NP`, `generaciones`, `F_r`, `cl`, `x_tilde`, `W`); keep them. The point of the project is the manual math: no `torch.nn`, `torch.autograd` or `torch.optim`; tensors are just a vectorized array library.

## Modules

- `config.py`: protocol constants, `LIFConfig` (`theta, a, b, c, h, V_th, V_0, t`) and `DEConfig` (`NP, dimension, F, CR, generaciones`), search bounds `[0, 1]`.
- `lif.py`: the neuron only. `input_current` (eq. 4, `I = x̄·w̄·θ`, an elementwise product+sum so a row's `I` doesn't depend on batch size), `F`, `euler_step` (`V_{i+1} = V_i + h F(t_i, V_i)`), `simulate` (explicit Euler loop over `t/h = 1000` steps, reset to `c` when `V_{i+1} ≥ V_th`, optional `V` trace), `firing_rate`, `class_firing_rates` (`AFR`, population-std `SDFR`), `classify` (eq. 3), and the `LIFNeuron` dataclass (`w`, `AFR`, `SDFR`, `cfg`). Every function accepts any shape of `I`, so the whole population × pattern grid (`NP×N`) is simulated at once.
- `fitness.py`: eq. 5, `F_i = 1/distancia(AFR) + Σ SDFR_k` (sum of pairwise `|AFR_k − AFR_l|`), and `fitness_population` (`(NP×n) -> (NP,)`), the DE objective.
- `algorithm/`: generic DE/rand/1/bin, a PyTorch port of `../de-algorithm/algorithm` (don't edit that project from here). It knows nothing about the neuron: it takes an objective `(P×d) -> (P,)`. Mutation `x_r1 + F(x_r2 − x_r3)`, binomial crossover with one forced mutant component, `repair_bounds` clipping of every trial (as in the original; without it the weights leave `[0, 1]` and the neuron saturates into a degenerate `AFR = [0, 1, 1]`), greedy selection `f(x_t) ≤ f(x_i)`. Each generation is vectorized, runs exactly `generaciones` generations (no early stop), and `best_fitness` stays a 0-d device tensor until `run()` ends.
- `training.py`: glue between the two. `train_subexperiment` runs DE on the training fold and returns `LIFNeuron.fit(W, ...)` (AFR/SDFR from the training fold), the best `F_i` per generation (`generaciones + 1` entries, generation 0 = initial population) and the `OptimizationResult`. `test_fitness_history` computes `F_i` on the test fold for each generation's best `w̄` after the run (only unique vectors are simulated); test data never drives training.
- `experiment.py`: protocol (`create_two_folds`, `check_folds`, `evaluate`, `run_subexperiment`, `run_experiment`). `main.py`: loop over experiments, logging, best model.

Speed: one generation (60 × 75 neurons × 1000 Euler steps) takes about 20 ms on CPU, so one subexperiment at 1000 generations takes about 20 s and the full 35 experiments (70 subexperiments) take about 25 min. It runs on CUDA when available; `DEVICE` does not select MPS.

## Experimental protocol (spans experiment.py and main.py)

- `NO_EXPERIMENTS = 35` experiments, each a stratified 2-fold split (25 samples per class per fold, 75/75 = the 50%/50% of the notes) built by `create_two_folds` and verified by `check_folds`.
- Each experiment has two subexperiments: train on A / test on B, then train on B / test on A, each from a fresh DE population.
- Features are used as they come from `iris.csv` (no standardisation, no bias).
- Prediction is `cl = argmin_k |AFR_k − F_r|` with the training fold's `AFR`. Per-experiment confusion matrix = sum of both subexperiments (must total 150); the global one must total `150 · NO_EXPERIMENTS`. Rows = true class, columns = predicted.
- Experiment `e` uses seed `BASE_SEED + e` for its fold split and its DE population; both subexperiments reuse it.
- Mean, median and std of `train_fitness` and `test_fitness` exist at three levels (subexperiment, experiment, all), always over per-generation values. `std` is the population std because `STD_POPULATION = True`.
- The "best model" is the subexperiment with the highest test accuracy, ties broken by lowest final training fitness (`main.better`).

## Outputs (`utils/`)

`utils/` mirrors the `utils/` package of the sibling project `../rbf_kmeans/rbf_iris` (same plot palette/rcParams, same file layout). The root modules hold the math and the protocol; `utils/` only reads their result dicts and writes files:

- `results.py`: per experiment `<RESULTS_DIR>/experiment_NN/` (split, aggregated confusion matrix, summaries) and per subexperiment `subexperiment_A_to_B|B_to_A/` (`generations.csv` with `generation, train_fitness, test_fitness`, `summary.json`, `weights.pt` with `w, AFR, SDFR`, `test_outputs.csv` with `I`, `F_r` and `dist_AFR_k` per test flower, `plots/`). `sub1` is A→B, `sub2` is B→A.
- Subexperiment plots: `01_train_fitness`, `02_confusion_test`, `03_class_distances_test` (`|AFR_k − F_r|`, one line per class), `04_membrane_potential_test` (heatmap of `V(t)`), `05_firing_rate_test` (`F_r` vs `AFR_k ± SDFR_k`).
- `global_results.py`: `<RESULTS_DIR>/global/` (`all_generations.csv.gz`, `subexperiments.csv`, `experiments.csv`, `global_statistics.json`, `config.json`, plots) and `<RESULTS_DIR>/best_model/` (`model.pt` + `model.json` loadable with `model_io.load_model`, which returns a `LIFNeuron`). `save_global` rebuilds the per-subexperiment curves by cutting the concatenated fitness lists with each row's `generations_executed`, so `main.py` must keep appending fitness, test fitness and rows in the same order.
- `SAVE_SUBEXPERIMENT_PLOTS` in `config.py` turns off the per-subexperiment PNGs.

## Conventions

- Style: `from __future__ import annotations`, type hints on every signature, `@dataclass` for state, shape comments like `# (NP×N) ... ej. (60×75)`.
- Keep the LIF model (`lif.py`, `fitness.py`) separate from the optimizer (`algorithm/`); `training.py` is the only place that joins them.
- Keep existing spellings such as `confussion_matrix_experimento`; the dict keys themselves are `"confusion_matrix"`.
- Comments, docstrings, log messages and assert messages are in Spanish; keep new ones in Spanish.
