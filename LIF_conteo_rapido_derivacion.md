# Conteo rápido de disparos de una neurona LIF

## Derivación matemática, aplicación a Iris y evolución diferencial

Apuntes de estudio basados en `LIF Notes.md` y en la explicación del conteo rápido. El objetivo es entender cómo reemplazar las iteraciones temporales de Euler por una consulta a tablas y una fórmula aritmética, sin modificar el modelo matemático discreto.

## 1. El problema original

Se utiliza una sola neurona LIF para clasificar Iris. Cada flor tiene cuatro características: longitud y anchura del sépalo, longitud y anchura del pétalo. La evolución diferencial (DE) optimiza cuatro pesos sinápticos.

$$
\bar x_j\in\mathbb R^4,\qquad \bar w\in\mathbb R^4
$$

La corriente de entrada es un escalar:

$$
I_j=\theta(\bar x_j\cdot\bar w)=\theta(x_{j1}w_1+x_{j2}w_2+x_{j3}w_3+x_{j4}w_4).
$$

La ganancia es $\theta=0.1$. Cada flor produce una corriente constante durante su simulación. Con 60 candidatos de pesos y 75 flores de entrenamiento se realizan 4500 evaluaciones independientes del mismo modelo LIF, no se construye una red de 4500 neuronas.

```mermaid
flowchart LR
    X[Cuatro características] --> P[Producto escalar con cuatro pesos]
    P --> I[Corriente constante]
    I --> L[LIF]
    L --> FR[Frecuencia de disparo]
    FR --> F[Fitness por candidato]
    F --> DE[Evolución diferencial]
    DE --> P
```

### Parámetros y discrepancias

| Parámetro | Notas originales | Explicación recibida |
|---|---:|---:|
| $a$ | 0.5 | 0.5 |
| $b$ | -0.0001 | -0.001 |
| Voltaje inicial $V_0$ | -60 | -60 |
| Reinicio $c$ | -50 | -50 |
| Umbral $V_{th}$ | 50 | 50 |
| Paso $h$ | 1 ms | 1 ms |
| Tiempo de simulación $t$ | 100 y 1000 ms | 1000 ms |
| Generaciones DE | 10000 | 1000 |

En estos apuntes se usa $b=-0.0001$, $t=1000$ ms y 10000 generaciones, siguiendo las notas de entrenamiento. Las notas contienen una contradicción entre 100 y 1000 ms. Debe confirmarse la configuración real en `lif.py`.

## 2. De la ecuación diferencial a Euler

El modelo es:

$$
\boxed{\frac{dV}{dt}=I+a-bV}
$$

La derivada indica la velocidad de cambio del voltaje. Sustituyendo los parámetros:

$$
\frac{dV}{dt}=I+0.5+0.0001V.
$$

El signo merece atención: como $b$ es negativo, el término de voltaje tiene signo positivo, a diferencia de la fuga estabilizadora convencional. Se conserva el modelo de las notas.

Euler aproxima la evolución continua mediante pasos discretos:

$$
\begin{aligned}
V_{n+1}&=V_n+hF(t_n,V_n)\\
&=V_n+h(I+a-bV_n)\\
&=(1-hb)V_n+h(I+a).
\end{aligned}
$$

Con $h=1$:

$$
\boxed{V_{n+1}=1.0001V_n+I+0.5.}
$$

Esta es una **recurrencia lineal de primer orden**: el siguiente término depende del anterior.

### Ejemplo manual con $I=2$

Partiendo de $V_0=-60$:

$$
\begin{aligned}
V_1&=1.0001(-60)+2.5=-57.506,\\
V_2&=1.0001(-57.506)+2.5=-55.0117506,\\
V_3&=1.0001(-55.0117506)+2.5\approx-52.51725.
\end{aligned}
$$

El algoritmo original repite esta operación 1000 veces, comprueba si el voltaje alcanza 50 y reinicia a -50 si ocurre un disparo.

## 3. Solución cerrada de la recurrencia

Definimos:

$$
q=1-hb,\qquad d=h(I+a).
$$

Entonces $V_{n+1}=qV_n+d$. Desarrollamos:

$$
\begin{aligned}
V_1&=qV_0+d,\\
V_2&=q(qV_0+d)+d=q^2V_0+d(q+1),\\
V_3&=q[q^2V_0+d(q+1)]+d=q^3V_0+d(q^2+q+1),\\
V_4&=q^4V_0+d(q^3+q^2+q+1).
\end{aligned}
$$

Generalizando:

$$
V_n=q^nV_0+d\sum_{k=0}^{n-1}q^k.
$$

La suma es una serie geométrica finita. Si $S_n=1+q+\cdots+q^{n-1}$, entonces:

$$
qS_n-S_n=q^n-1\implies S_n=\frac{q^n-1}{q-1},\quad q\ne1.
$$

Por tanto:

$$
\boxed{V_n=q^nV_0+h(I+a)\frac{q^n-1}{q-1}.}
$$

Si $q=1$, la suma vale $n$. Esta expresión permite calcular directamente el voltaje de cualquier paso **antes del siguiente reinicio**. Es la solución de la recurrencia discreta de Euler, no la solución continua de la ecuación diferencial.

La corriente constante permite mantener el mismo $d$ en cada término. Si la corriente varía, aparece una suma ponderada de corrientes diferentes y ya no se garantiza la periodicidad entre reinicios.

También podemos escribir:

$$
V_n=A_n I+B_n,
\qquad A_n=hS_n,\qquad B_n=q^nV_0+haS_n.
$$

Los coeficientes no dependen de la flor ni de sus pesos: para cada paso, el voltaje es una función afín de la corriente.

## 4. Corriente mínima para alcanzar el umbral

La neurona dispara si $V_n\ge V_{th}$. Sustituimos y despejamos:

$$
\begin{aligned}
q^nV_0+h(I+a)S_n&\ge V_{th},\\
h(I+a)S_n&\ge V_{th}-q^nV_0,\\
I+a&\ge\frac{V_{th}-q^nV_0}{hS_n},\\
I&\ge\frac{V_{th}-q^nV_0}{hS_n}-a.
\end{aligned}
$$

La división conserva el sentido de la desigualdad porque $hS_n>0$ para los parámetros empleados.

$$
\boxed{I_{\min}(n;V_0)=\frac{V_{th}-q^nV_0}{hS_n}-a.}
$$

Esta corriente mínima no depende de $\bar x$ ni de $\bar w$. Solo depende de la configuración LIF, del estado inicial y del paso $n$.

### Tabla precalculada

Para $q=1.0001$, $h=1$, $a=0.5$ y $V_{th}=50$:

| Pasos $n$ | Desde -60 | Desde -50 |
|---:|---:|---:|
| 1 | 109.5060 | 99.5050 |
| 2 | 54.5033 | 49.5025 |
| 3 | 36.1690 | 32.8350 |
| 10 | 10.5011 | 9.5005 |
| 20 | 5.0008 | 4.5003 |
| 40 | 2.250641 | 2.000128 |
| 41 | 2.183565 | 1.939150 |
| 44 | 2.000629 | 1.772845 |
| 45 | 1.945071 | 1.722337 |
| 100 | 0.600564 | 0.500058 |
| 1000 | -0.389403 | -0.399912 |

La tabla decrece conforme aumenta el número de pasos. De hecho, usando $q^n=1+(q-1)S_n$:

$$
I_{\min}(n;V_0)=\frac{V_{th}-V_0}{hS_n}-\frac{(q-1)V_0}{h}-a.
$$

Como $V_{th}>V_0$ y $S_n$ crece para $q>0$, el primer término decrece. Esto justifica la búsqueda binaria.

### Ejemplo: $I=2$

Desde -60, el paso 44 requiere 2.000629, mayor que 2; el paso 45 requiere 1.945071, menor que 2. Por tanto, el primer disparo ocurre en el paso 45. Los voltajes sin reinicio en esos pasos son aproximadamente 49.9722 y 52.4773.

Desde -50, el paso 40 requiere 2.000128, mayor que 2; el paso 41 requiere 1.939150, menor que 2. Por tanto, el intervalo posterior al reinicio es de 41 pasos.

## 5. Por qué se necesitan dos tablas

El primer recorrido empieza en $V_0=-60$. Tras cada disparo, el voltaje se reinicia a $c=-50$. La corriente y los parámetros permanecen constantes, de modo que todos los recorridos posteriores son idénticos.

- $t_1$: número de pasos hasta el primer disparo.
- $T_p$: número de pasos entre disparos sucesivos.

Aquí ambos símbolos representan pasos discretos, no necesariamente milisegundos; coinciden numéricamente porque $h=1$ ms.

Con $I=2$:

$$
t_1=45,\qquad T_p=41.
$$

```text
Paso:     0        45       86       127      168      209 ...
          |---------*--------*--------*--------*--------*
             45         41       41       41       41
Estado:   -60       reset a -50 después de cada disparo
```

El instante del disparo número $k$ sigue una progresión aritmética:

$$
\boxed{t_k=t_1+(k-1)T_p.}
$$

## 6. Fórmula directa para contar disparos

Queremos contar todos los disparos que ocurran dentro de $N$ pasos. Un disparo cuenta si $t_k\le N$:

$$
\begin{aligned}
t_1+(k-1)T_p&\le N,\\
(k-1)T_p&\le N-t_1,\\
k&\le 1+\frac{N-t_1}{T_p}.
\end{aligned}
$$

Como $k$ es un entero, aplicamos la función piso $\lfloor\cdot\rfloor$:

$$
\boxed{N_{\text{spikes}}=\begin{cases}
0,&t_1>N,\\
1+\left\lfloor\dfrac{N-t_1}{T_p}\right\rfloor,&t_1\le N.
\end{cases}}
$$

Con 1000 pasos:

$$
N_{\text{spikes}}=1+\left\lfloor\frac{1000-45}{41}\right\rfloor=1+23=24.
$$

La razón de disparo definida en las notas es:

$$
\boxed{FR=\frac{N_{\text{spikes}}}{t}=\frac{24}{1000}=0.024\ \text{disparos/ms}.}
$$

Esto equivale a 24 Hz. En general $t=Nh$ si el tiempo es un múltiplo exacto del paso.

| Duración (ms) | Disparos con $I=2$ | FR (disparos/ms) |
|---:|---:|---:|
| 100 | 2 | 0.020 |
| 500 | 12 | 0.024 |
| 1000 | 24 | 0.024 |
| 3000 | 73 | 0.024333 |

## 7. Búsqueda binaria y `searchsorted`

Necesitamos encontrar el primer paso $n$ para el cual $I\ge I_{\min}(n)$.

Una búsqueda secuencial podría recorrer hasta 1000 entradas. Pero como la tabla es monótona decreciente, basta una búsqueda binaria, aproximadamente $\log_2(1000)\approx10$ comparaciones.

`searchsorted` espera una tabla ascendente. Cambiar el signo de todos los elementos invierte el orden:

$$
I_{\min}(1)>I_{\min}(2)>\cdots\implies -I_{\min}(1)<-I_{\min}(2)<\cdots.
$$

```python
indice = np.searchsorted(-tabla_I_min, -I, side="left")
pasos = indice + 1
```

El `+1` convierte el índice base cero en un número de pasos base uno. `side="left"` incluye la igualdad, coherente con la condición $V\ge V_{th}$. Si ninguna entrada cumple, devuelve el tamaño de la tabla, por lo que `pasos=N+1` indica que no hubo disparo dentro de la simulación.

Se realizan dos búsquedas: una para $t_1$ y otra para $T_p$. No es una búsqueda de optimización ni de convergencia.

## 8. Ejemplo completo con una flor de Iris

Sean:

$$
\bar x=(5.1,3.5,1.4,0.2),\qquad \bar w=(0.8,0.1,0.6,0.4).
$$

Con ganancia 0.1:

$$
\begin{aligned}
I&=0.1(5.1(0.8)+3.5(0.1)+1.4(0.6)+0.2(0.4))\\
&=0.1(4.08+0.35+0.84+0.08)\\
&=0.535.
\end{aligned}
$$

En la primera tabla, el paso 106 requiere aproximadamente 0.538297, mientras que el paso 107 requiere 0.528599. Por tanto, $t_1=107$.

La segunda tabla proporciona $T_p=97$.

$$
N_{\text{spikes}}=1+\left\lfloor\frac{1000-107}{97}\right\rfloor=10,
\qquad FR=0.010\ \text{disparos/ms}.
$$

Este escalar se utiliza para calcular los estadísticos de la clase correspondiente.

## 9. Conexión con evolución diferencial (DE)

DE sigue optimizando los pesos. El conteo rápido únicamente acelera la evaluación del fitness.

La población y los datos se pueden representar como matrices:

$$
W\in\mathbb R^{60\times4},\qquad X\in\mathbb R^{75\times4}.
$$

Las corrientes se calculan de forma vectorizada:

$$
\boxed{I=\theta WX^\top\in\mathbb R^{60\times75}.}
$$

Cada entrada representa una combinación candidato-flor. Se obtienen otras 4500 entradas de FR mediante las dos búsquedas y la fórmula de conteo.

Para cada candidato, sean $C_k$ los índices de las flores de clase $k$ y $p_k=|C_k|$:

$$
AFR_k=\frac{1}{p_k}\sum_{j\in C_k}FR_j.
$$

$$
SDFR_k=\sqrt{\frac{1}{p_k}\sum_{j\in C_k}(FR_j-AFR_k)^2}.
$$

Aquí se muestra la desviación estándar poblacional; debe conservarse la convención utilizada por el código real.

Para las tres clases:

$$
D=|AFR_1-AFR_2|+|AFR_1-AFR_3|+|AFR_2-AFR_3|.
$$

La función fitness es:

$$
\boxed{f(\bar w)=\frac{1}{D}+\sum_{k=1}^{3}SDFR_k.}
$$

El primer término favorece la separación de las medias; el segundo favorece la agrupación dentro de cada clase. Si $D=0$, se necesita contemplar la división por cero conforme a la implementación.

El objetivo de DE es:

$$
\bar w^*=\operatorname{argmin}_{\bar w} f(\bar w).
$$

DE sigue generando candidatos mediante mutación y cruza, evaluando todas las flores y seleccionando individuos. No puede obtener el fitness evaluando una sola flor porque necesita los estadísticos de las clases completas.

En la etapa de prueba, tras optimizar los pesos, se asigna la clase cuyo AFR de entrenamiento está más cerca del FR de la flor de prueba:

$$
cl=\operatorname{argmin}_k|AFR_k-FR|.
$$

Las dos tablas se comparten entre los 60 candidatos, las 75 flores, todas las generaciones y los subexperimentos, siempre que los parámetros LIF y la longitud necesaria de las tablas no cambien.

## 10. Complejidad y ahorro computacional

| Método | Operaciones para $M$ corrientes y $N$ pasos |
|---|---|
| Euler iterativo | $O(MN)$ |
| Construcción de tablas | $O(N)$, una vez por configuración |
| Evaluación con tablas | $O(M\log N)$ |
| Total con tablas | $O(N+M\log N)$ |

Con 60 candidatos, 75 flores y 10000 generaciones:

$$
60\times75\times10000=45\,000\,000\text{ evaluaciones de FR}.
$$

Euler requiere aproximadamente 45 mil millones de actualizaciones de voltaje por subexperimento para 1000 pasos. El método rápido mantiene las 45 millones de evaluaciones, pero reemplaza cada simulación por dos búsquedas binarias y operaciones aritméticas. Las cifras no incluyen las evaluaciones iniciales ni otros costos de DE.

La explicación recibida reportaba 17 ms frente a 0.17 ms por generación, y menos de un minuto para la corrida completa con 1000 generaciones. Esas mediciones no se han reproducido aquí. Con 10000 generaciones, no se puede garantizar el mismo tiempo total. Es necesario medir en el hardware y código reales.

## 11. Equivalencia y limitaciones

- La fórmula reproduce la recurrencia de Euler en aritmética real, no una discretización diferente de la ecuación diferencial.
- La corriente debe ser constante durante cada evaluación.
- El reinicio debe ser siempre al mismo potencial y sin otras variables dinámicas, adaptación ni periodos refractarios no contemplados.
- La búsqueda binaria requiere tablas ordenadas.
- El orden de operaciones en punto flotante puede modificar el paso de disparo cuando la corriente esté muy cerca de un umbral, especialmente en `float32`.
- La comparación reportada en 12010 corrientes es evidencia experimental, no una prueba de igualdad bit a bit para todas las entradas. Conviene comparar también corrientes inmediatamente por encima y por debajo de cada umbral.
- No se obtiene la traza completa $V(t)$. Para las gráficas de potencial de membrana se conserva `simulate`.

También sería posible despejar el paso de cruce mediante logaritmos bajo condiciones apropiadas; las tablas evitan manejar directamente muchos de esos casos y permiten búsquedas vectorizadas.

## 12. Ejercicios de comprensión (técnica Feynman)

Intenta responder sin consultar las secciones anteriores.

1. Si dos flores generan exactamente la misma corriente para un candidato, ¿pueden tener FR diferentes por pertenecer a clases distintas?
2. Si DE cambia uno de los cuatro pesos, ¿es necesario reconstruir las tablas de corriente mínima?
3. Si el primer disparo ocurre en el paso 120 y los siguientes cada 80 pasos, ¿cuántos disparos caben en 1000 pasos?
4. ¿Por qué la solución cerrada no debe aplicarse desde -60 durante toda la simulación sin considerar los reinicios?
5. Reconstruye la fórmula de corriente mínima partiendo de Euler, sin mirar las notas.

### Respuestas

1. No. Con la misma corriente y configuración LIF, los intervalos y el FR son idénticos; la etiqueta no participa en la dinámica.
2. No. Se recalculan las corrientes y los FR; las tablas dependen exclusivamente de la configuración LIF.
3. $1+\lfloor(1000-120)/80\rfloor=12$. El último ocurre en el paso 1000.
4. Porque el disparo modifica el estado: el siguiente recorrido empieza en -50, no continúa la trayectoria anterior.
5. Desarrollar la recurrencia, identificar la suma geométrica, sustituir la condición de umbral y despejar $I$ (secciones 2–4).

## Conclusión

El algoritmo DE continúa buscando cuatro pesos sinápticos óptimos para una sola neurona LIF. Lo único que cambia es cómo se evalúa la respuesta de cada flor:

```text
Original:  corriente → 1000 pasos de Euler → conteo → FR
Optimizado: corriente → buscar t1 y Tp → fórmula aritmética → FR
```

La lección transferible a programación científica es que, antes de optimizar un bucle, conviene investigar si existe una recurrencia, una invariante, una expresión cerrada o una estructura periódica que permita eliminarlo por completo.

---

**Nota de edición:** se adaptaron las visualizaciones interactivas de la respuesta original a diagramas, tablas y ejercicios estáticos compatibles con Markdown. Se conservaron las derivaciones y se comprobaron numéricamente los ejemplos principales con la configuración indicada. No se inspeccionó ni modificó la implementación real de `lif.py`.
