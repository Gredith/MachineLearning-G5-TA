# Contexto del proyecto y decisiones

Documento de referencia para el equipo: qué pide el curso, qué sabemos de los datos, qué problema elegimos y por qué. Las cifras salen del notebook [`01_auditoria_datos_v2.ipynb`](../notebooks/01_auditoria_datos_v2.ipynb) y del [Entregable 1](entregables/E1/Entregable1_G5.docx).

## 1. Encargo del curso

- Diseñar, implementar y evaluar modelos de ML sobre un problema real de transporte público en Cusco, con el dataset CSV entregado por los docentes.
- La tarea puede ser supervisada (clasificación, regresión) o no supervisada (clustering), con selección de características o reducción de dimensionalidad si hace falta.
- Se exige: revisión de trabajos relacionados, justificación metodológica, comparación de algoritmos con diseño experimental y métricas pertinentes, análisis crítico y **reproducibilidad**.
- Librerías especializadas (scikit-learn, PyTorch o TensorFlow). Paper en formato **IEEE**. Se recomienda una demo interactiva.
- Grupos de 3 o 4 integrantes con un responsable.

| Entrega | Fecha | Puntos | Qué se sube |
|---|---|---|---|
| E1: Propuesta y antecedentes | 6 nov | 5 | PDF con introducción, objetivos, trabajos relacionados, dataset, algoritmos y diseño experimental preliminar |
| E2: Implementación y evaluación | 20 nov | 5 | PDF (incluye el E1) con metodología definitiva, experimentos, resultados, discusión y conclusiones + código `.ipynb`/`.py` (GitHub recomendado) |
| Presentación final | 27 nov | 10 | Paper final PDF, código, diapositivas PDF; exposición de 15 min + 5 min de preguntas; demo cuando corresponda |

E2 debe incluir: EDA y preprocesamiento, implementación, ajuste de hiperparámetros, train/val/test, métricas, comparación con tablas y gráficos, y descripción del entorno (librerías, versiones, parámetros).

## 2. Lo que sabemos de los datos

**Confirmado** = calculado sobre los archivos reales. **Hipótesis** = interpretación coherente, no confirmada por los docentes (no hay ventana de consultas).

| Hecho | Etiqueta |
|---|---|
| Times: 4.707 vueltas, 45 vehículos, 27 días seguidos (6 ene–1 feb 2025), sin duplicados | Confirmado |
| Una fila = una vuelta con 18 puntos de control en listas; `lap` es un contador diario por vehículo (1 a 8) | Confirmado |
| `difference_time` = truncado a minutos de (`end_time` − `start_time`) en el 100 % de 76.980 puntos | Confirmado |
| `start_time` es el horario programado y `end_time` la hora real → `difference_time` < 0 adelanto, > 0 atraso | Hipótesis |
| 9,1 % de los puntos sin hora real (`end_time` = 00:00:00 y `difference_time` = "-") | Confirmado |
| 0,2 % de los puntos (156) con \|valor\| > 60 min, de −1052 a +631, casi todos en los puntos 11–18 | Confirmado |
| Sin extremos: mediana 0, p05 −5, p95 +10 min; 13,6 % atrasado ≥ 5 min, 9,3 % adelantado ≥ 3 min | Confirmado |
| La dispersión crece en la vuelta: desv. est. 1,8 min (punto 1) a 8,2 min (punto 18) | Confirmado |
| Correlación del atraso entre puntos consecutivos 0,86; punto 10 vs 18: 0,76; puntos 1–3 vs 18: ≈ 0 | Confirmado |
| Mayor atraso medio los sábados (+4,2 min) y de 17 a 18 h (+3,2 a +3,5 min) | Confirmado |
| 3,1 % de horarios (148 vueltas) no crecientes a lo largo de los 18 puntos | Confirmado |
| Monitoring: 1.630.431 puntos, 32 días (1 ene–1 feb); 1–5 ene sin filas en Times | Confirmado |
| Muestreo GPS irregular: mediana 25 s, p95 91 s; 6.690 filas repetidas sin contar el id | Confirmado |
| `lap` = 0 en Monitoring es vehículo fuera de operación (2,9 % de los puntos) | Dato de clase |
| `speed` en km/h; `bearing` 0° al norte | Hipótesis |
| Un solo trazado para todos los vehículos (Jaccard medio 0,79) | Hipótesis |
| Unión por vehículo, fecha y `lap`: 99,9 % de las vueltas tiene GPS; 95,1 % de los puntos cae en su ventana | Confirmado |
| Los 18 puntos de control **no tienen coordenadas**: el GPS se asocia por tiempo, no por lugar | Confirmado |

Verificaciones V1–V19: 17 cerradas, 2 parciales (V9: unidad de `speed`; V17: 5 % de vueltas empieza o termina lejos de la ruta). Ninguna bloquea el problema elegido. Detalle en el Anexo A del E1.

## 3. Formulación del problema

| Candidato | Tipo | Decisión |
|---|---|---|
| **P1.** Atraso en el punto 18 dado el corte k | Regresión | **Principal** |
| **P2.** Nivel de atraso final (atrasado / a tiempo / adelantado) | Clasificación | Variante sobre las mismas observaciones |
| P3. Atraso en el punto siguiente al corte | Regresión | Sensibilidad si sobra tiempo |
| P4. Perfiles de puntualidad (vector de 18 desviaciones) | Clustering | Complementario: describe el dataset y justifica umbrales de P2 |
| P5. Zonas de detención con GPS | Clustering espacial | Descartado: 0,72 % de puntos detenidos y sin verdad de referencia |

Por qué P1: predecir siempre "a tiempo" da MAE 5,07 min en el punto 18 (último 25 % de días); la persistencia del atraso del punto 9 baja a 3,95 min y la del 13 a 2,85 min. Hay margen para que hora, vehículo, flota y GPS aporten más. La duración de la vuelta (R1 de la v1) es P1 con k = 0.

- **Unidad de observación:** vuelta × corte k, con k ∈ {4, 6, 9, 12, 15}; k = 9 es la referencia.
- **Objetivo P1:** `difference_time` del punto 18, en minutos.
- **Objetivo P2:** atrasado (≥ 5 min), a tiempo, adelantado (≥ 3 min de adelanto). Umbrales provisionales.
- **Variables en el corte k** (grupos para ablación):
  - *Horario:* hora de inicio (cíclica), día de la semana, minutos programados restantes, tiempo programado por tramo restante.
  - *Atraso observado:* puntos 1 a k con indicador de faltante, último valor, media de los últimos tres, máximo, pendiente.
  - *Vehículo y contexto:* vehículo, número de vuelta del día, atraso final de la vuelta anterior (si terminó antes del corte), atraso medio de otros vehículos en los 30 min previos.
  - *GPS:* velocidad media, proporción detenida y puntos recibidos hasta el corte.
- **Excluido por fuga:** todo `end_time`/`difference_time` posterior a k, GPS posterior al corte, atrasos de vueltas que terminan después del corte, estadísticos calculados con días de prueba.
- **Filtros:** vueltas sin punto 18 válido (10,9 %); puntos GPS con `lap` = 0; \|valor\| > 60 min como regla provisional (se contrasta con detección de anomalías y se reporta con y sin ella). Faltantes en predictoras como NaN + indicador.

## 4. Diseño experimental

**Partición:** temporal por días. Entrenamiento/validación 6–25 ene (3.466 vueltas); prueba 26 ene–1 feb (1.241 vueltas), usada una sola vez. Nunca división aleatoria por fila.

**Hipótesis**

| N.º | Hipótesis | Contraste |
|---|---|---|
| H1 | El atraso observado mejora la predicción frente a solo el horario | MAE vs. B0/B1 por corte k |
| H2 | Los árboles superan a los lineales | Ridge vs. Random Forest y Gradient Boosting |
| H3 | El error baja con más puntos observados | Curva MAE vs. k |
| H4 | El GPS aporta información adicional | Ablación con y sin GPS |
| H5 | Pérdidas absoluta/Huber rinden mejor que la cuadrática ante la cola de atrasos | Mismo modelo con distintas pérdidas, MAE y RMSE |
| H6 | El contexto de vehículo y flota reduce el error en franjas de mayor atraso | Ablación y error por franja y día |

**Líneas base:** B0 "a tiempo" (0); B1 mediana del atraso final por hora de inicio y día; B2 persistencia (atraso del punto k); B3 regresión lineal con solo el atraso del punto k.

**Algoritmos** (solo los vistos en el curso; sin redes neuronales):

| Familia | Algoritmos | Uso |
|---|---|---|
| Lineales | Regresión lineal (también con gradiente descendente), Ridge, Lasso | P1 |
| Robusta | Huber | P1 (H5) |
| Vecinos | KNN | P1, P2 |
| Árboles | Árbol de decisión | P1, P2 |
| Kernels | SVR (RBF), SVM | P1, P2 |
| Ensambles | Random Forest, Bagging, AdaBoost, Gradient Boosting | P1, P2 |
| Clasificación | Regresión logística regularizada | P2 |
| Complementarias | Detección de anomalías, PCA, selección de variables | Datos, H4, H6 |
| Clustering | K-medias, K-medianas, jerárquico, GMM (EM), DBSCAN, OPTICS | P4 |

**Métricas**
- Regresión: **MAE** (principal), mediana del error absoluto, RMSE, R², % de vueltas con error ≤ 2 y ≤ 5 min. Por corte k y franja horaria.
- Clasificación: F1 de la clase atrasado, F1 macro, exactitud balanceada, AUC-PR, matriz de confusión.
- Clustering: silueta, Davies-Bouldin, Calinski-Harabasz, estabilidad por remuestreo (ARI).

**Validación y reproducibilidad:** hiperparámetros con validación de ventana creciente (4 pliegues por bloques de días). Imputación, escalado, codificación y umbrales se ajustan solo con entrenamiento. Semilla 42, versiones registradas, notebooks ejecutables de punta a punta. IC por bootstrap sobre días; Wilcoxon sobre errores absolutos para comparar pares de modelos.

**Comparaciones previstas:** todos los modelos vs. B0–B3 en cada k; ablación por grupos (horario → + atraso → + vehículo/flota → + GPS); pérdida cuadrática vs. absoluta vs. Huber; sensibilidad a la regla de 60 min; P1 vs. P2 y umbrales de P2.

## 5. Registro de decisiones

| Fecha | Decisión | Motivo / evidencia |
|---|---|---|
| 5 nov 2026 | Pasar del plan v1 al v2: la unidad es la vuelta con 18 puntos de control | `difference_time` es una lista por punto, no un valor por vuelta |
| 5 nov 2026 | Problema principal P1 (atraso en el punto 18 dado k), P2 como variante | Mejor margen frente a líneas base y objetivo bien definido (sección 3) |
| 5 nov 2026 | `start_time` = horario programado, declarado como supuesto | Minutos exactos, `time` = primer elemento, `difference_time` = diferencia |
| 5 nov 2026 | Usar solo GPS con `lap` > 0 | `lap` = 0 es fuera de operación (dato de clase) |
| 5 nov 2026 | Regla provisional de exclusión \|valor\| > 60 min | 0,2 % de puntos extremos; se reporta con y sin la regla |
| 5 nov 2026 | Partición temporal por días, prueba 26 ene–1 feb | Evitar fuga temporal |
| 5 nov 2026 | Solo algoritmos del curso, sin deep learning | Alcance del curso |
| 6 nov 2026 | Entregable 1 presentado | — |

Agregar aquí cada nueva decisión (fecha, qué y por qué).

## 6. Supuestos y riesgos

| Supuesto o riesgo | Cómo se maneja |
|---|---|
| `start_time`/`end_time` = programado/real, sin confirmación docente | Se declara como supuesto con la evidencia |
| Puntos de control sin coordenadas | GPS asociado por ventana de tiempo de la vuelta |
| 9,1 % de puntos sin hora real ("no registrado" o "no pasado") | Indicador de faltante; vueltas sin punto 18 fuera del objetivo; sensibilidad |
| 0,2 % de valores extremos | Regla de 60 min, resultados con y sin ella |
| Pocas vueltas (~3.500 para entrenar) | Regularización, validación por bloques de días, IC; no afirmar diferencias dentro del ruido |
| Solo 27 días de enero 2025 | Limitación de generalización; sin variables de mes o feriado |
| Fuga por estadísticos de vehículo/día | Todo estadístico solo con entrenamiento; revisión antes de cada experimento |
| 148 vueltas con horario no creciente | Se mantienen con indicador; sensibilidad |
| Monitoring: 6.690 filas repetidas y 5 días sin Times | Se eliminan repetidas; esos días solo para describir |
| V9 y V17 parciales | `speed` en km/h con nota; vueltas con GPS inconsistente se marcan |

## 7. Trabajos relacionados

1. V. Martyanov et al., "Interpretable machine learning for identifying the determinants of bus delays," *Arch. Transp.*, vol. 77, no. 1, 2026, doi: 10.61089/aot2026.mnrkh112. — Atraso por parada en Varsovia, RF/XGBoost/CatBoost con validación por semanas; MAE 3,6 min. El más cercano; no usa el atraso ya observado.
2. Meyer-Hollatz, Schwarz y Werner, "Punctuality predictions in public transportation: Quantifying the effect of external factors," *Wirtschaftsinformatik 2023*, paper 328. — Tranvías; XGBoost mejor; clima y feriados mejoran el MAE 17,33 %. Motiva la ablación.
3. M. Yan, B. Xie y G. Xu, "Identifying spatial–temporal characteristics and significant factors of bus bunching based on an eGA and DT model," *Appl. Sci.*, vol. 12, no. 22, 2022, doi: 10.3390/app122211778. — Árboles y selección de variables; pesan la secuencia de paradas y los tiempos previos.
4. Huang et al., "Bus single-trip time prediction based on ensemble learning," *Comput. Intell. Neurosci.*, 2022, doi: 10.1155/2022/6831167. — Lineal, KNN y ensambles con Wilcoxon/Friedman; RF mejor con MAE 3,23 min. Solo se cita su diseño y los ensambles.

**Vacío:** no se encontró trabajo que prediga la desviación al final de la vuelta comparando cortes intermedios, ni con datos de una ciudad peruana. Meta: ampliar a 6–10 referencias verificadas para el paper final.
