# Tareas pendientes

Mapa de trabajo desde el Entregable 1 (entregado el 6 nov) hasta la presentación final (27 nov). Marcar con `[x]` al terminar y anotar decisiones nuevas en [CONTEXTO.md](CONTEXTO.md#5-registro-de-decisiones).

**Roles** (del plan v2): **A** Datos · **B** Literatura · **C** Método · **D** Responsable y ensamblaje. Con tres integrantes, D se suma a A. Asignar nombres:

| Rol | Integrante |
|---|---|
| A. Datos | _por definir_ |
| B. Literatura | _por definir_ |
| C. Método | _por definir_ |
| D. Responsable | _por definir_ |

## Estado al 18 nov

- **Rama `feat/entregable-2`** (8 commits sobre `main`, subida a GitHub, **sin fusionar**): notebooks 02 a 07, módulos de `src/`, 12 figuras y 35 tablas en `results/`, informe del E2 (PDF y DOCX) y documentación actualizada.
- **Fases 1 a 3 completas.** Fase 4 lista salvo la revisión del equipo, la fusión a `main` y la subida a la plataforma (vence el 20 nov). Fase 5 sin empezar.
- **Resultado principal:** Gradient Boosting gana en todos los cortes; en k = 9 el MAE de prueba es 2,64 min (IC 95 % 2,25–3,20), frente a 3,20 de B1 y 3,93 de B2.
- **Bloqueante para el E2:** el informe dice que el código está en el repositorio, pero `main` todavía no tiene los notebooks 02 a 07 ni `src/`. Hay que fusionar el PR antes de entregar.

## Revisión del E2 antes de entregar (19 nov) — todos

Puntos que conviene revisar porque el informe los afirma o porque son decisiones discutibles:

- [ ] **Redacción de H3.** Gradient Boosting casi no cambia entre k = 4 y k = 9 (2,66 → 2,64 min, dentro del IC). El informe lo presenta como "se cumple con una meseta"; considerar "se cumple solo desde k = 12".
- [ ] **Significancia por día.** Con 7 días de prueba, el p mínimo posible de Wilcoxon por día es 0,016. H1 en k = 4 (p = 0,47) y H6 (p ≥ 0,08) no son significativas por día. Aclarar si los p < 0,001 citados son por vuelta.
- [ ] **Regla de 60 min en prueba.** El MAE reportado excluye 11 vueltas de prueba con \|y\| > 60 min; con ellas el MAE de GB en k = 9 es 7,20 y el RMSE 62. Declararlo explícitamente en resultados y limitaciones (`07_sensibilidad_60min.csv`).
- [ ] **Rejilla de Gradient Boosting.** Los valores elegidos caen en los bordes de la rejilla (`n_estimators` = 500 en k = 4, 12 y 15; `learning_rate` = 0,02 en k = 4 y 15; `max_depth` = 2 en k = 12; `min_samples_leaf` = 10 en todos). Mencionarlo como limitación o ampliar la rejilla para la versión final.
- [ ] **Cambio de distribución en P2.** "Atrasado" es el 33,9 % en entrenamiento y el 27,7 % en prueba (`05_distribucion_clases.csv`). Comentarlo en la discusión.
- [ ] **Fuga en `ctx_flota_30`.** Es la 3.ª variable más importante. Revisé `src/variables.py` y solo usa pasos de otros vehículos con hora real anterior al corte; confirmar que el equipo está de acuerdo con incluir días de prueba en esa ventana (es información disponible en el instante de predicción).
- [ ] **Cifras del informe contra `results/tables/`** (tabla VI con `04_mae_test_por_k.csv`, tabla VIII con `07_ablacion.csv`, tabla X con `05_metricas_k9.csv`).
- [ ] **Reejecutar 02 a 07 con los CSV** en una máquina del equipo para confirmar que se reproducen las cifras (en este clon no hay datos en `data/raw/`).

## Fase 0 · Repositorio (16 nov)

- [x] Estructura de carpetas, `.gitignore`, `requirements.txt`
- [x] Notebook de auditoría en `notebooks/01_auditoria_datos_v2.ipynb`
- [x] Entregable 1 (DOCX) en `docs/entregables/E1/`
- [x] Guía de [convenciones](CONVENCIONES.md) para ramas, commits y notebooks
- [ ] Proteger `main` en GitHub (requerir PR con una revisión) (D)
- [ ] Agregar el PDF entregado del E1 en `docs/entregables/E1/` (D)
- [x] Cambiar `RUTA_TIMES`/`RUTA_MONITORING` del notebook 01 a `../data/raw/` y `DIR_SALIDA` a `../results/` (A)
- [x] Fijar versiones mínimas en `requirements.txt` y anotar las versiones probadas (C)
- [ ] Rellenar la tabla de roles (D)

## Fase 1 · Preprocesamiento y variables por corte (hasta 17 nov) — A, C

- [x] `02_preprocesamiento_variables_por_corte.ipynb`
- [x] Parsear las listas de 18 puntos de Times; marcar faltantes (`00:00:00`, `-`)
- [x] Aplicar filtros: vueltas sin punto 18 válido, regla \|valor\| > 60 min (con bandera para la sensibilidad), GPS con `lap` = 0, filas repetidas de Monitoring
- [x] Unir Times y Monitoring por vehículo, fecha y `lap`; GPS dentro de la ventana de la vuelta
- [x] Construir el dataset vuelta × k (k ∈ {4, 6, 9, 12, 15}) con los cuatro grupos de variables, sin fuga
- [x] Partición temporal: train/val 6–25 ene, test 26 ene–1 feb; pliegues de ventana creciente (4)
- [x] Mover a `src/` las funciones que se reutilicen (carga, parseo, variables, partición)
- [x] Guardar en `data/processed/` y documentar en el notebook las cifras finales de filas por k

## Fase 2 · Líneas base y modelos de regresión P1 (hasta 18 nov) — C

- [x] `03_lineas_base.ipynb`: B0, B1, B2, B3 por corte k
- [x] `04_regresion_P1.ipynb` con pipelines de scikit-learn (imputación/escalado ajustados solo en train)
- [x] Lineales: regresión lineal (y versión con gradiente descendente), Ridge, Lasso
- [x] Huber; KNN; árbol de decisión; SVR RBF
- [x] Ensambles: Random Forest, Bagging, AdaBoost, Gradient Boosting
- [x] Ajuste de hiperparámetros con validación de ventana creciente; registrar los mejores en `results/tables/`
- [x] Evaluación en prueba (una sola vez): MAE, MedAE, RMSE, R², % error ≤ 2 y ≤ 5 min, por k y franja

## Fase 3 · Clasificación P2 y experimentos (hasta 19 nov) — A, C

- [x] `05_clasificacion_P2.ipynb`: logística regularizada, KNN, árbol, SVM, RF, AdaBoost; F1 atrasado, F1 macro, exactitud balanceada, AUC-PR, matriz de confusión
- [x] Sensibilidad de umbrales de P2 (5/3 min vs. terciles)
- [x] `07_analisis_resultados.ipynb`:
  - [x] H1 y H3: curva MAE vs. k frente a líneas base
  - [x] H2: Ridge vs. RF y GB
  - [x] H4 y H6: ablación por grupos de variables; error por franja horaria y día
  - [x] H5: pérdida cuadrática vs. absoluta vs. Huber
  - [x] Sensibilidad con y sin la regla de 60 min
  - [x] IC por bootstrap sobre días y Wilcoxon entre pares de modelos
- [x] Figuras y tablas finales en `results/figures/` y `results/tables/`
- [x] (Opcional) `06_clustering_P4.ipynb`: perfiles de puntualidad con PCA, K-medias, GMM, jerárquico, DBSCAN; silueta, DB, CH, ARI

## Fase 4 · Entregable 2 (20 nov) — D con todos

- [x] Borrador del paper IEEE que **incluya el E1** más: metodología definitiva, EDA y preprocesamiento, experimentos, resultados, discusión y conclusiones
- [x] Sección de entorno computacional: Python, librerías y versiones, semilla, hiperparámetros
- [x] Revisión cruzada: cifras del paper contra las tablas de `results/`
- [x] Verificar que los notebooks corren de punta a punta en limpio
- [x] Copiar el PDF y el DOCX del E2 en `docs/entregables/E2/`
- [ ] Abrir PR `feat/entregable-2` → `main`, revisarlo y fusionarlo (el informe enlaza al repositorio)
- [ ] Marcar el commit entregado con `git tag e2` y `git push --tags`
- [ ] Subir el PDF a la plataforma y enlazar el repositorio de GitHub
- [ ] Confirmar la hora límite en la plataforma del curso

## Fase 5 · Presentación final (hasta 27 nov) — todos

- [ ] Incorporar observaciones de los docentes al E2
- [ ] Paper final en PDF (`docs/entregables/final/`)
- [ ] Diapositivas en PDF: introducción, trabajos relacionados, metodología, resultados, discusión, conclusiones y trabajo futuro, demo
- [ ] (Recomendado) Demo interactiva en `demo/` (por ejemplo, elegir una vuelta y un corte k y ver la predicción frente al real)
- [ ] Ensayo con cronómetro: 15 min + 5 min de preguntas
- [ ] Repositorio final: README actualizado, resultados y entregables

## Pendientes abiertos

- [ ] V9: confirmar la unidad de `speed` (km/h)
- [ ] V17: decidir cómo marcar las vueltas que empiezan o terminan lejos de la ruta
- [ ] Ampliar trabajos relacionados de 4 a 6–10 referencias verificadas (DOI que abra) — B. El informe del E2 sigue con las 4 del E1
- [ ] Completar la referencia [2] del informe (faltan iniciales de los autores y enlace o DOI)
- [x] Justificar los umbrales de P2 con la distribución observada (o con P4)
- [ ] Contrastar la regla de 60 min con detección de anomalías (la sensibilidad con y sin la regla ya está en el notebook 07)
