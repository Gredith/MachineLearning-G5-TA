# Contexto del proyecto y decisiones

Documento de referencia para el equipo: qué pide el curso, qué sabemos de los datos, qué problema elegimos y por qué. Las cifras salen del notebook [`01_auditoria_datos_v2.ipynb`](../notebooks/01_auditoria_datos_v2.ipynb), del [Entregable 1](entregables/E1/Entregable1_G5.docx) y de los notebooks 02 a 06 (Entregable 2).

## 1. Encargo del curso

- Diseñar, implementar y evaluar modelos de ML sobre un problema real de transporte público en Cusco, con el dataset CSV entregado por los docentes.
- La tarea puede ser supervisada (clasificación, regresión) o no supervisada (clustering), con selección de características o reducción de dimensionalidad si hace falta.
- Se exige: revisión de trabajos relacionados, justificación metodológica, comparación de algoritmos con diseño experimental y métricas pertinentes, análisis crítico y **reproducibilidad**.
- Librerías especializadas (scikit-learn, PyTorch o TensorFlow). Paper en formato **IEEE**. Se recomienda una demo interactiva.
- Grupos de 3 o 4 integrantes con un responsable.

**Indicaciones adicionales de los docentes** (comunicadas al grupo después del E1):
- Enfocarse en **un solo tipo de aprendizaje** (supervisado o no supervisado). El proyecto queda solo con aprendizaje supervisado; el clustering P4 se retiró.
- Buscar que las métricas de evaluación **superen el 80 %**.
- **Dar peso al preprocesamiento:** al ser datos reales, cada decisión de limpieza debe evaluarse antes de pasar a los modelos.

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
| Tras quitar `lap` = 0 y repetidas quedan 1.575.820 puntos GPS | Confirmado |
| 4.146 vueltas con punto 18 válido y \|valor\| ≤ 60 min (3.085 de entrenamiento y 1.061 de prueba) | Confirmado |
| `speed` en km/h (cociente mediano 1,07 frente a la velocidad calculada con las coordenadas) | Confirmado |
| `bearing` 0° al norte | Hipótesis |
| 1.853 puntos GPS aislados (se llega y se sale a más de 144 km/h) se eliminan; quedan 1.573.967 | Confirmado |
| El 7,4 % de las vueltas empieza a más de 1 km del inicio típico (V17) | Confirmado |
| Un solo trazado para todos los vehículos (Jaccard medio 0,79) | Hipótesis |
| Unión por vehículo, fecha y `lap`: 99,9 % de las vueltas tiene GPS; 95,1 % de los puntos cae en su ventana | Confirmado |
| Los 18 puntos de control **no tienen coordenadas**: el GPS se asocia por tiempo, no por lugar | Confirmado |

Verificaciones V1–V19: 18 cerradas y 1 parcial. V9 se cerró en el E2 (`speed` en km/h). V17 sigue parcial: las vueltas que empiezan lejos de la ruta se marcan con `gps_dist_inicio` en lugar de descartarse. Detalle en el Anexo A del informe.

## 3. Formulación del problema

| Candidato | Tipo | Decisión |
|---|---|---|
| **P1.** Atraso en el punto 18 dado el corte k | Regresión | **Principal** |
| **P2.** Nivel de atraso final (atrasado / a tiempo / adelantado) | Clasificación | Variante sobre las mismas observaciones |
| P3. Atraso en el punto siguiente al corte | Regresión | Sensibilidad si sobra tiempo |
| P4. Perfiles de puntualidad (vector de 18 desviaciones) | Clustering | Retirado en el E2: los docentes pidieron un solo tipo de aprendizaje |
| P5. Zonas de detención con GPS | Clustering espacial | Descartado: 0,72 % de puntos detenidos y sin verdad de referencia |

Por qué P1 (cifras exploratorias del E1, con el último 25 % de días como prueba): predecir siempre "a tiempo" da MAE 5,07 min en el punto 18; la persistencia del atraso del punto 9 baja a 3,95 min y la del 13 a 2,85 min. Con la partición definitiva del E2 son 5,05 y 3,93 min (`03_lineas_base.csv`). Hay margen para que hora, vehículo, flota y GPS aporten más. La duración de la vuelta (R1 de la v1) es P1 con k = 0.

- **Unidad de observación:** vuelta × corte k, con k ∈ {4, 6, 9, 12, 15}; k = 9 es la referencia.
- **Objetivo P1:** `difference_time` del punto 18, en minutos.
- **Objetivo P2:** atrasado (≥ 5 min), a tiempo, adelantado (≥ 3 min de adelanto). El de atraso sigue la ventana de puntualidad del TCQSM (de 1 min de adelanto a 5 de atraso); el de adelanto se fijó con los datos (con 1 min, el 23 % sería adelantada) y los terciles dan peor F1 macro.
- **Variables en el corte k** (grupos para ablación):
  - *Horario:* hora de inicio (cíclica y lineal), día de la semana, minutos programados restantes y totales, indicador de horario no creciente.
  - *Atraso observado:* puntos 1 a k (faltantes internos interpolados), último valor, distancia al último registro, media de los últimos tres, máximo, mínimo, pendiente y número de faltantes.
  - *Vehículo y contexto:* vehículo, número de vuelta del día, atraso final de la vuelta anterior (si terminó antes del corte), atraso medio de otros vehículos en los 30 min previos.
  - *GPS:* velocidad media, proporción detenida, velocidad de los últimos 10 min y puntos recibidos hasta el corte, y distancia del inicio de la vuelta al inicio típico.
- **Excluido por fuga:** todo `end_time`/`difference_time` posterior a k, GPS posterior al corte, atrasos de vueltas que terminan después del corte, estadísticos calculados con días de prueba.
- **Filtros:** vueltas sin punto 18 válido (10,9 %); puntos GPS con `lap` = 0, repetidos o aislados; \|valor\| > 60 min (contrastada con la distribución, la cerca de Tukey y la z robusta, y reportada con y sin ella).
- **Preprocesamiento elegido por validación (V3):** interpolación de faltantes internos, GPS limpio y selección de variables (sin indicadores de faltante ni tramos programados por punto). Iguala el MAE de las demás variantes con un 30 a 42 % menos de variables.

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

**Algoritmos** (los vistos en el curso; en el paper final se agrega una red neuronal simple como comparación, notebook 07):

| Familia | Algoritmos | Uso |
|---|---|---|
| Lineales | Regresión lineal (también con gradiente descendente), Ridge, Lasso | P1 |
| Robusta | Huber | P1 (H5) |
| Vecinos | KNN | P1, P2 |
| Árboles | Árbol de decisión | P1, P2 |
| Kernels | SVR (RBF), SVM | P1, P2 |
| Ensambles | Random Forest, Bagging, AdaBoost, Gradient Boosting | P1, P2 |
| Clasificación | Regresión logística regularizada | P2 |
| Complementarias | Contraste robusto de valores extremos, selección de variables, filtro de varianza | Preprocesamiento |
| Red neuronal | Perceptrón multicapa (`MLPRegressor`) con búsqueda equivalente a Gradient Boosting | P1, notebook 07 |

**Métricas**
- Regresión: **MAE** (principal), mediana del error absoluto, RMSE, R², % de vueltas con error ≤ 2 y ≤ 5 min. Por corte k y franja horaria.
- Clasificación: exactitud, F1 de la clase atrasado, F1 macro, exactitud balanceada, AUC-PR, matriz de confusión.
- Meta de los docentes: superar 0,80. Se reporta qué métricas lo logran en cada corte (`06_metricas_meta80.csv`).

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
| 18 nov 2026 | Instante de predicción = hora real de paso por el punto k (la programada si no se registró) | Todas las variables usan solo información anterior a ese instante |
| 18 nov 2026 | Un modelo por corte k | El número de variables cambia con k (30 a 41 tras la selección) |
| 18 nov 2026 | Pliegues: 8 días iniciales fijos y 4 bloques de validación de 3 días | Ventana creciente dentro de 6 a 25 ene, sin mezclar días |
| 18 nov 2026 | Hiperparámetros por MAE medio de validación; prueba evaluada una sola vez | Evitar elegir con la semana de prueba |
| 18 nov 2026 | Se mantiene la regla de 60 min | Sin ella el MAE de Gradient Boosting sube de 2,62 a 8,78 min en vueltas normales (k = 9) |
| 18 nov 2026 | Umbrales de P2 fijos en 5 y 3 min | Con terciles el F1 macro baja de 0,764 a 0,680 |
| 18 nov 2026 | Gradient Boosting como modelo de referencia | Menor MAE de validación en los cinco cortes |
| 18 nov 2026 | Las variables GPS se reportan sin aporte (H4 no se cumple) | La ablación no mejora el MAE en ningún k y su importancia por permutación es nula |
| 18 nov 2026 | SVM sin `probability=True` | Parámetro deprecado en scikit-learn 1.9; el AUC-PR usa el puntaje de decisión |
| 18 nov 2026 | Retirar el clustering P4; el proyecto queda solo supervisado | Indicación de los docentes: un solo tipo de aprendizaje |
| 18 nov 2026 | P1 y P2 se presentan como un mismo problema con dos enfoques | Mismas observaciones, variables y partición; P2 discretiza el objetivo de P1, y el regresor discretizado iguala al mejor clasificador |
| 18 nov 2026 | Mantener la regla de 60 min frente a Tukey y la z robusta | 24 puntos entre 40 y 60 min frente a 138 sobre 120 min; Tukey quitaría 210 vueltas reales y la z robusta 760 puntos |
| 18 nov 2026 | Limpiar el GPS (puntos aislados > 144 km/h) y cerrar V9 | Cociente speed / velocidad calculada de 1,07: speed en km/h |
| 18 nov 2026 | Preprocesamiento V3: interpolación, GPS limpio y selección de variables | Mismo MAE de validación que la base (±0,01 min) con 30 a 42 % menos variables; el recorte del objetivo empeora |
| 18 nov 2026 | Rejilla de Gradient Boosting de 16 a 81 combinaciones; exactitud agregada a P2 | Los valores elegidos estaban en el borde; meta del 80 % de los docentes |
| 25 nov 2026 | Demo v2 (`feat/demo-v2`): configuración global, idioma ES/EN, tema claro/oscuro/sistema, predicción en una vista | Observaciones de los docentes: demo con interfaz e interpretación del resultado |
| 26 nov 2026 | El paper final se trabaja en `docs/entregables/final/Paper_final_G5.docx`, copia del E2 | El E2 queda tal como se entregó |
| 26 nov 2026 | Preprocesamiento del paper ampliado: eliminaciones e imputación justificadas, Fig. 2 (flujo de limpieza), Fig. 3 (variantes) y Tablas D4–D5 | Observación de los docentes: abundar en el preprocesamiento; se imputa en lugar de eliminar porque 10–12 % de las vueltas tiene algún faltante en 1..k y 30 % no tiene vuelta anterior (`02_faltantes_imputacion.csv`) |
| 26 nov 2026 | No se adoptan otros cambios de preprocesamiento (quitar vueltas atípicas, codificar el vehículo por su atraso medio, atraso final reciente) | Ninguno mejora el MAE de validación de forma consistente (±0,03 min, `02_experimentos_adicionales.csv`) |
| 26 nov 2026 | Umbral de atraso de 5 min justificado con el TCQSM; el de adelanto (3 min), con los datos | Verificado en el manual original (TCQSM 3.ª ed., cap. 5, Anexo 5-21, p. 5-30): "a tiempo" es una salida entre 1 min de adelanto y 5 de atraso o una llegada al terminal con hasta 5 min de atraso; el punto 18 es el terminal. Con 1 min de adelanto, el 23 % de las vueltas sería adelantada; con 3 min, el 11,9 % |
| 26 nov 2026 | En resultados, figuras en lugar de tablas (Fig. 5 a 7) y tabla de lectura de métricas (Tabla V); las tablas pasan a D6–D8 | Observación de los docentes: más gráficos y más interpretabilidad de las métricas |
| 26 nov 2026 | Mejoras en un notebook aparte (07) sin reejecutar 04–06 | Evita cambiar todas las cifras a un día de la entrega; las referencias se leen de `04_*` y `05_*` |
| 26 nov 2026 | La pérdida de Huber se reporta como mejora, sin reemplazar el modelo del informe | Baja el MAE de validación en 4 de 5 cortes (−0,03 a −0,05) y el de prueba en los 5 (k = 9: 2,62 → 2,58) |
| 26 nov 2026 | La red neuronal (MLP) se compara con el mismo esfuerzo de ajuste que Gradient Boosting (81 combinaciones) y capacidad del mismo orden | Con una rejilla de 6 combinaciones la comparación no era justa; aun igualada, queda 0,08–0,22 min peor en validación |
| 26 nov 2026 | No se adoptan pesos balanceados en Gradient Boosting y AdaBoost (P2) | Bajan el F1 macro de validación entre 0,02 y 0,05 |
| 26 nov 2026 | Diagnóstico de sobreajuste/subajuste de todos los modelos (entrenamiento vs. validación vs. prueba, curvas de aprendizaje) | Pedido del equipo: comprobar que la mejora sea robusta y no forzada |
| 26 nov 2026 | La demo usa Gradient Boosting con pérdida de Huber en los cinco cortes | Mejor resultado de las pruebas; en k = 9 la validación empata (2,981 frente a 2,980) y en prueba gana en los 5 cortes. En prueba supera 0,80 en error ≤ 5 min y AUC-PR en todos los cortes, y en exactitud desde k = 6 (`07_modelo_demo_meta80.csv`) |
| 26 nov 2026 | Párrafo de técnicas en VI-F y referencias [6]–[8] (Cerqueira et al. 2020, Hooker et al. 2021, Friedman 2001), verificadas en Crossref | Justifican la validación temporal, la importancia por permutación por grupos en lugar de SHAP y la pérdida de Huber; formato IEEE como el resto del paper |
| 26 nov 2026 | Idioma y tema de la demo pasan a la esquina superior izquierda del área principal (a la derecha de la barra lateral y de su botón para abrirla) | En Streamlit Community Cloud la esquina derecha tiene los botones de la plataforma (compartir, GitHub, menú) y se superponían |

Agregar aquí cada nueva decisión (fecha, qué y por qué).

## 6. Resultados del Entregable 2

Cifras de prueba (26 ene a 1 feb) con los hiperparámetros elegidos en validación y el preprocesamiento V3. Tablas completas en `results/tables/`. P1 y P2 son el mismo problema: P1 predice los minutos de desviación en el punto 18 (regresión) y P2 el estado atrasado / a tiempo / adelantado (clasificación) con las mismas observaciones y variables.

| Corte k | B1 mediana hora y día | B2 persistencia | Ridge | Random Forest | Gradient Boosting |
|---|---|---|---|---|---|
| 4 | 3,20 | 4,94 | 3,89 | 2,72 | 2,66 |
| 6 | 3,20 | 3,93 | 3,39 | 2,71 | 2,67 |
| 9 | 3,20 | 3,93 | 3,42 | 2,70 | 2,62 |
| 12 | 3,20 | 3,56 | 3,04 | 2,50 | 2,46 |
| 15 | 3,20 | 2,38 | 2,26 | 2,09 | 2,05 |

MAE en minutos. En k = 9, Gradient Boosting tiene IC 95 % de 2,24 a 3,19 min (bootstrap por días), R² 0,75 y 86,8 % de vueltas con error de 5 min o menos.

| Hipótesis | Resultado |
|---|---|
| H1 | Se cumple desde k = 12: el atraso observado baja el MAE de 2,78 a 2,51 (k = 12) y 2,10 (k = 15); entre k = 4 y 9 la mejora es ≤ 0,04 min |
| H2 | Se cumple: Gradient Boosting y Random Forest superan a Ridge en todos los cortes (Wilcoxon p < 0,001) |
| H3 | Se cumple desde k = 12, con meseta entre k = 4 y 9 |
| H4 | No se cumple: el GPS no mejora el MAE |
| H5 | Con matices: pérdida absoluta 2,56 frente a cuadrática 2,62 en prueba, pero no en validación; Huber mejora a la regresión lineal (3,30 vs. 3,46) |
| H6 | Parcial: vehículo y flota mejoran 0,07 min en total, 0,24 min de 16 a 19 h y 0,33 min los sábados |

P2 (k = 9): Gradient Boosting obtiene exactitud 0,793, F1 macro 0,764 y F1 de atrasado 0,721; discretizar el regresor de P1 da casi lo mismo (exactitud 0,796, F1 macro 0,762).

**Mejoras del paper final** (notebook 07, `results/tables/07_*.csv`):
- Pérdida de Huber en Gradient Boosting: MAE de validación −0,03 a −0,05 min en 4 de 5 cortes (igual en k = 9) y de prueba −0,04 a −0,07 en los 5; en k = 9, 2,58 min (R² 0,758; 87,6 % con error ≤ 5 min).
- Red neuronal (MLP) con 81 combinaciones y 5 mil a 62 mil parámetros (Gradient Boosting: 2 mil a 12 mil hojas): 3,20 min de MAE de validación en k = 9 frente a 2,98; en prueba, 3,08 frente a 2,62. Queda por debajo de Random Forest en todos los cortes.
- Pesos balanceados en P2: F1 macro de validación −0,02 a −0,05; no se adoptan.
- Modelo de la demo (GB con Huber) en prueba: error ≤ 5 min 0,876 a 0,942 y AUC-PR 0,817 a 0,886 en los cinco cortes; exactitud al convertir en estados 0,796 (k = 4), 0,802, 0,800 (k = 9), 0,812 y 0,838 (k = 15).
- Diagnóstico (k = 9): Gradient Boosting 2,39 min en entrenamiento, 2,98 en validación y 2,62 en prueba (brecha moderada, mejor validación); Random Forest, Bagging y KNN sobreajustan más; los lineales subajustan (≈ 3,2 en entrenamiento). Agregar árboles no empeora la validación.

**Meta del 80 %** (`06_metricas_meta80.csv`): en k = 9 la superan el % de vueltas con error ≤ 5 min (86,8 %) y el AUC-PR de atrasado (0,80); quedan por debajo R² (0,75), exactitud (0,79) y F1 macro (0,76). En k = 15 la superan R² (0,84), exactitud (0,83), AUC-PR (0,88) y error ≤ 5 min (93,9 %); el F1 macro queda en 0,80. El preprocesamiento reforzado no movió el error: a mitad de la vuelta el límite lo pone la información disponible.

## 7. Supuestos y riesgos

| Supuesto o riesgo | Cómo se maneja |
|---|---|
| `start_time`/`end_time` = programado/real, sin confirmación docente | Se declara como supuesto con la evidencia |
| Puntos de control sin coordenadas | GPS asociado por ventana de tiempo de la vuelta |
| 9,1 % de puntos sin hora real ("no registrado" o "no pasado") | Indicador de faltante; vueltas sin punto 18 fuera del objetivo; sensibilidad |
| 0,2 % de valores extremos | Regla de 60 min contrastada con la distribución, Tukey y z robusta; resultados con y sin ella |
| Pocas vueltas (~3.500 para entrenar) | Regularización, validación por bloques de días, IC; no afirmar diferencias dentro del ruido |
| Solo 27 días de enero 2025 | Limitación de generalización; sin variables de mes o feriado |
| Fuga por estadísticos de vehículo/día | Todo estadístico solo con entrenamiento; revisión antes de cada experimento |
| 148 vueltas con horario no creciente | Se mantienen con indicador; sensibilidad |
| Monitoring: 6.690 filas repetidas y 5 días sin Times | Se eliminan repetidas; esos días solo para describir |
| V17 parcial (V9 cerrada en el E2) | Las vueltas que empiezan lejos de la ruta se marcan con `gps_dist_inicio` |

## 8. Trabajos relacionados

1. V. Martyanov et al., "Interpretable machine learning for identifying the determinants of bus delays," *Arch. Transp.*, vol. 77, no. 1, 2026, doi: 10.61089/aot2026.mnrkh112. — Atraso por parada en Varsovia, RF/XGBoost/CatBoost con validación por semanas; MAE 3,6 min. El más cercano; no usa el atraso ya observado.
2. T. Meyer-Hollatz, N. Schwarz y T. Werner, "Punctuality predictions in public transportation: Quantifying the effect of external factors," *Wirtschaftsinformatik 2023 Proceedings*, paper 328, 2023. https://aisel.aisnet.org/wi2023/73 — Tranvías; XGBoost mejor; clima y feriados mejoran el MAE 17,33 %. Motiva la ablación.
3. M. Yan, B. Xie y G. Xu, "Identifying spatial–temporal characteristics and significant factors of bus bunching based on an eGA and DT model," *Appl. Sci.*, vol. 12, no. 22, 2022, doi: 10.3390/app122211778. — Árboles y selección de variables; pesan la secuencia de paradas y los tiempos previos.
4. Huang et al., "Bus single-trip time prediction based on ensemble learning," *Comput. Intell. Neurosci.*, 2022, doi: 10.1155/2022/6831167. — Lineal, KNN y ensambles con Wilcoxon/Friedman; RF mejor con MAE 3,23 min. Solo se cita su diseño y los ensambles.

**Vacío:** no se encontró trabajo que prediga la desviación al final de la vuelta comparando cortes intermedios, ni con datos de una ciudad peruana. Meta: ampliar a 6–10 referencias verificadas para el paper final.
