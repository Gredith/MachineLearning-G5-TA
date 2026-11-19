# Tareas pendientes

Mapa de trabajo desde el Entregable 1 (entregado el 6 nov) hasta la presentación final (27 nov). Marcar con `[x]` al terminar y anotar decisiones nuevas en [CONTEXTO.md](CONTEXTO.md#5-registro-de-decisiones).

**Roles** (del plan v2): **A** Datos · **B** Literatura · **C** Método · **D** Responsable y ensamblaje. Con tres integrantes, D se suma a A. Asignar nombres:

| Rol | Integrante |
|---|---|
| A. Datos | _por definir_ |
| B. Literatura | _por definir_ |
| C. Método | _por definir_ |
| D. Responsable | _por definir_ |

## Estado al 18 nov (noche)

- **Ramas:** `feat/entregable-2` (trabajo original del E2) y `feat/entregable-2-v2` (sale de la anterior y aplica las indicaciones adicionales de los docentes). Ninguna está fusionada en `main`.
- **Indicaciones de los docentes aplicadas** en `feat/entregable-2-v2`:
  - Un solo tipo de aprendizaje: se retiró el clustering P4; quedan P1 (regresión) y P2 (clasificación), que son el mismo problema con dos enfoques.
  - Preprocesamiento: contraste de la regla de 60 min, limpieza del GPS (V9 cerrada), interpolación, selección de variables y comparación de variantes por validación (notebook 02, Tabla III del informe).
  - Meta del 80 %: se agregó la exactitud y una tabla de métricas frente a 0,80. **No se cumple en todas las métricas en k = 9** (R² 0,75, exactitud 0,79, F1 macro 0,76); sí en el error ≤ 5 min (86,8 %) y el AUC-PR (0,80). En k = 15 la cumplen casi todas.
- **Resultado principal:** Gradient Boosting gana en todos los cortes; en k = 9 el MAE de prueba es 2,62 min (IC 95 % 2,24–3,19), frente a 3,20 de B1 y 3,93 de B2.
- **Informe del E2** actualizado en la rama nueva: más conciso, con el material secundario en el Anexo D y el enlace al repositorio.
- **Bloqueante para el E2:** el informe enlaza al repositorio, pero `main` todavía no tiene el código del E2. Hay que fusionar las ramas antes de entregar.

## Revisión del E2 antes de entregar (19 nov) — todos

- [x] **Redacción de H3 y H1.** Ahora dicen "se cumple desde k = 12", con la meseta entre k = 4 y 9.
- [x] **Significancia por día.** El informe aclara qué p son por vuelta y cuáles por día.
- [x] **Regla de 60 min en prueba.** El informe declara que las métricas excluyen 11 vueltas de prueba (con ellas el MAE de GB en k = 9 es 7,19).
- [x] **Rejilla de Gradient Boosting.** Se amplió a 81 combinaciones; en k = 9 y 15 elige tasa 0,01 con 1.000 árboles, el compromiso habitual tasa-árboles, con el mismo MAE de validación.
- [x] **Cambio de distribución en P2.** Mencionado en el informe (33,9 % atrasado en entrenamiento y 27,7 % en prueba).
- [ ] **Fuga en `ctx_flota_30`.** Solo usa pasos de otros vehículos con hora real anterior al corte; confirmar que el equipo está de acuerdo.
- [ ] **Modelos lineales.** La selección de variables se eligió con Gradient Boosting y a los lineales les resta 0,1–0,15 min; está declarado como limitación. Decidir si se acepta.
- [ ] **Meta del 80 %.** Decidir con el equipo cómo presentarla: hoy el informe reporta con transparencia qué métricas la superan.
- [ ] **Revisar el PDF del E2** (redacción y formato) y las cifras contra `results/tables/`.

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
- [x] `06_analisis_resultados.ipynb` (antes 07):
  - [x] H1 y H3: curva MAE vs. k frente a líneas base
  - [x] H2: Ridge vs. RF y GB
  - [x] H4 y H6: ablación por grupos de variables; error por franja horaria y día
  - [x] H5: pérdida cuadrática vs. absoluta vs. Huber
  - [x] Sensibilidad con y sin la regla de 60 min
  - [x] IC por bootstrap sobre días y Wilcoxon entre pares de modelos
- [x] Figuras y tablas finales en `results/figures/` y `results/tables/`
- [x] ~~(Opcional) `06_clustering_P4.ipynb`~~ retirado: los docentes pidieron un solo tipo de aprendizaje

## Fase 4 · Entregable 2 (20 nov) — D con todos

- [x] Borrador del paper IEEE que **incluya el E1** más: metodología definitiva, EDA y preprocesamiento, experimentos, resultados, discusión y conclusiones
- [x] Sección de entorno computacional: Python, librerías y versiones, semilla, hiperparámetros
- [x] Revisión cruzada: cifras del paper contra las tablas de `results/`
- [x] Verificar que los notebooks corren de punta a punta en limpio
- [x] Copiar el PDF y el DOCX del E2 en `docs/entregables/E2/`
- [ ] Fusionar `feat/entregable-2-v2` (que contiene `feat/entregable-2`) en `main` mediante PR revisado (el informe enlaza al repositorio)
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

- [x] V9: `speed` está en km/h (cociente 1,07 con la velocidad calculada)
- [x] V17: las vueltas que empiezan lejos de la ruta se marcan con `gps_dist_inicio`
- [ ] Ampliar trabajos relacionados de 4 a 6–10 referencias verificadas (DOI que abra) — B. El informe del E2 sigue con las 4 del E1
- [ ] Completar la referencia [2] del informe (faltan iniciales de los autores y enlace o DOI)
- [x] Justificar los umbrales de P2 con la distribución observada (o con P4)
- [x] Contrastar la regla de 60 min con la distribución, la cerca de Tukey y la z robusta (notebook 02)
