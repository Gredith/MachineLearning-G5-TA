# Tareas pendientes

Mapa de trabajo desde el Entregable 1 (entregado el 6 nov) hasta la presentación final (27 nov). Marcar con `[x]` al terminar y anotar decisiones nuevas en [CONTEXTO.md](CONTEXTO.md#5-registro-de-decisiones).

**Roles** (del plan v2): **A** Datos · **B** Literatura · **C** Método · **D** Responsable y ensamblaje. Con tres integrantes, D se suma a A. Asignar nombres:

| Rol | Integrante |
|---|---|
| A. Datos | _por definir_ |
| B. Literatura | _por definir_ |
| C. Método | _por definir_ |
| D. Responsable | _por definir_ |

## Fase 0 · Repositorio (16 nov)

- [x] Estructura de carpetas, `.gitignore`, `requirements.txt`
- [x] Notebook de auditoría en `notebooks/01_auditoria_datos_v2.ipynb`
- [x] Entregable 1 (DOCX) en `docs/entregables/E1/`
- [x] Guía de [convenciones](CONVENCIONES.md) para ramas, commits y notebooks
- [ ] Proteger `main` en GitHub (requerir PR con una revisión) (D)
- [ ] Agregar el PDF entregado del E1 en `docs/entregables/E1/` (D)
- [ ] Cambiar `RUTA_TIMES`/`RUTA_MONITORING` del notebook 01 a `../data/raw/` y `DIR_SALIDA` a `../results/` (A)
- [ ] Fijar versiones en `requirements.txt` (`pip freeze`) una vez probado el entorno (C)
- [ ] Rellenar la tabla de roles (D)

## Fase 1 · Preprocesamiento y variables por corte (hasta 17 nov) — A, C

- [ ] `02_preprocesamiento_variables_por_corte.ipynb`
- [ ] Parsear las listas de 18 puntos de Times; marcar faltantes (`00:00:00`, `-`)
- [ ] Aplicar filtros: vueltas sin punto 18 válido, regla \|valor\| > 60 min (con bandera para la sensibilidad), GPS con `lap` = 0, filas repetidas de Monitoring
- [ ] Unir Times y Monitoring por vehículo, fecha y `lap`; GPS dentro de la ventana de la vuelta
- [ ] Construir el dataset vuelta × k (k ∈ {4, 6, 9, 12, 15}) con los cuatro grupos de variables, sin fuga
- [ ] Partición temporal: train/val 6–25 ene, test 26 ene–1 feb; pliegues de ventana creciente (4)
- [ ] Mover a `src/` las funciones que se reutilicen (carga, parseo, variables, partición)
- [ ] Guardar en `data/processed/` y documentar en el notebook las cifras finales de filas por k

## Fase 2 · Líneas base y modelos de regresión P1 (hasta 18 nov) — C

- [ ] `03_lineas_base.ipynb`: B0, B1, B2, B3 por corte k
- [ ] `04_regresion_P1.ipynb` con pipelines de scikit-learn (imputación/escalado ajustados solo en train)
- [ ] Lineales: regresión lineal (y versión con gradiente descendente), Ridge, Lasso
- [ ] Huber; KNN; árbol de decisión; SVR RBF
- [ ] Ensambles: Random Forest, Bagging, AdaBoost, Gradient Boosting
- [ ] Ajuste de hiperparámetros con validación de ventana creciente; registrar los mejores en `results/tables/`
- [ ] Evaluación en prueba (una sola vez): MAE, MedAE, RMSE, R², % error ≤ 2 y ≤ 5 min, por k y franja

## Fase 3 · Clasificación P2 y experimentos (hasta 19 nov) — A, C

- [ ] `05_clasificacion_P2.ipynb`: logística regularizada, KNN, árbol, SVM, RF, AdaBoost; F1 atrasado, F1 macro, exactitud balanceada, AUC-PR, matriz de confusión
- [ ] Sensibilidad de umbrales de P2 (5/3 min vs. terciles)
- [ ] `07_analisis_resultados.ipynb`:
  - [ ] H1 y H3: curva MAE vs. k frente a líneas base
  - [ ] H2: Ridge vs. RF y GB
  - [ ] H4 y H6: ablación por grupos de variables; error por franja horaria y día
  - [ ] H5: pérdida cuadrática vs. absoluta vs. Huber
  - [ ] Sensibilidad con y sin la regla de 60 min
  - [ ] IC por bootstrap sobre días y Wilcoxon entre pares de modelos
- [ ] Figuras y tablas finales en `results/figures/` y `results/tables/`
- [ ] (Opcional) `06_clustering_P4.ipynb`: perfiles de puntualidad con PCA, K-medias, GMM, jerárquico, DBSCAN; silueta, DB, CH, ARI

## Fase 4 · Entregable 2 (20 nov) — D con todos

- [ ] Borrador del paper IEEE que **incluya el E1** más: metodología definitiva, EDA y preprocesamiento, experimentos, resultados, discusión y conclusiones
- [ ] Sección de entorno computacional: Python, librerías y versiones, semilla, hiperparámetros
- [ ] Revisión cruzada: cifras del paper contra las tablas de `results/`
- [ ] Verificar que los notebooks corren de punta a punta en limpio
- [ ] Subir PDF a la plataforma y copiarlo en `docs/entregables/E2/`; enlazar el repositorio de GitHub
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
- [ ] Ampliar trabajos relacionados de 4 a 6–10 referencias verificadas (DOI que abra) — B
- [ ] Justificar los umbrales de P2 con la distribución observada (o con P4)
- [ ] Contrastar la regla de 60 min con detección de anomalías
