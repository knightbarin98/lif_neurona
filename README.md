# lif_neurona

Una neurona LIF (Leaky Integrate-and-Fire) que clasifica Iris por razón de disparo; sus pesos sinápticos se calibran con evolución diferencial DE/rand/1/bin (Vazquez y Cachón, 2010).

```bash
conda activate iris_nn
python main.py      # resultados en resultados_conteo_rapido/
```

- `lif.py`: modelo LIF (I = x̄·w̄·θ, método de Euler, conteo rápido de disparos, FR, AFR, SDFR, cl = argmin |AFR_k − F_r|)
- `fitness.py`: F_i = 1/distancia(AFR) + Σ SDFR_k (ec. 5 del artículo)
- `algorithm/`: evolución diferencial genérica
- `training.py`: une la neurona con el DE
- `experiment.py`, `main.py`: protocolo 2-fold estratificado y bitácora
- `utils/`: archivos de salida y gráficas
