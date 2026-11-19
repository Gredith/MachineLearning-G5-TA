# Tareas pendientes

Mapa de trabajo desde el Entregable 1 (entregado el 6 nov) hasta la presentación final (27 nov). Marcar con `[x]` al terminar y anotar decisiones nuevas en [CONTEXTO.md](CONTEXTO.md#5-registro-de-decisiones).

## Estado al 18 nov (noche)

- **Ramas:** `feat/entregable-2` (trabajo original del E2) y `feat/entregable-2-v2` (sale de la anterior y aplica las indicaciones adicionales de los docentes). Ambas están en GitHub; ninguna está fusionada en `main`.
- **Indicaciones de los docentes aplicadas** en `feat/entregable-2-v2`:
  - Un solo tipo de aprendizaje: se retiró el clustering P4; quedan P1 (regresión) y P2 (clasificación), que son el mismo problema con dos enfoques.
  - Preprocesamiento: contraste de la regla de 60 min, limpieza del GPS (V9 cerrada), interpolación, selección de variables y comparación de variantes por validación (notebook 02, Tabla III del informe).
  - Meta del 80 %: se agregó la exactitud y una tabla de métricas frente a 0,80. **No se cumple en todas las métricas en k = 9** (R² 0,75, exactitud 0,79, F1 macro 0,76); sí en el error ≤ 5 min (86,8 %) y el AUC-PR (0,80). En k = 15 la cumplen casi todas.
- **Resultado principal:** Gradient Boosting gana en todos los cortes; en k = 9 el MAE de prueba es 2,62 min (IC 95 % 2,24–3,19), frente a 3,20 de B1 y 3,93 de B2.
- **Informe del E2** actualizado en la rama nueva: más conciso, con el material secundario en el Anexo D, el enlace al repositorio y la referencia [2] completa.
- **Bloqueante para el E2:** el informe enlaza al repositorio, pero `main` todavía no tiene el código del E2. Hay que fusionar las ramas antes de entregar.

## Fase 0 · Repositorio (16 nov)

- [x] Estructura de carpetas, `.gitignore`, `requirements.txt`
- [x] Notebook de auditoría en `notebooks/01_auditoria_datos_v2.ipynb`
- [x] Entregable 1 (DOCX) en `docs/entregables/E1/`
- [x] Guía de [convenciones](CONVENCIONES.md) para ramas, commits y notebooks
- [x] Cambiar `RUTA_TIMES`/`RUTA_MONITORING` del notebook 01 a `../data/raw/` y `DIR_SALIDA` a `../results/`
- [x] Fijar versiones mínimas en `requirements.txt` y anotar las versiones probadas

## Fase 1 · Preprocesamiento y variables por corte (hasta 17 nov)

- [x] `02_preprocesamiento_variables_por_corte.ipynb`
- [x] Parsear las listas de 18 puntos de Times; marcar faltantes (`00:00:00`, `-`)
- [x] Aplicar filtros: vueltas sin punto 18 válido, regla \|valor\| > 60 min (con bandera para la sensibilidad), GPS con `lap` = 0, filas repetidas de Monitoring
- [x] Unir Times y Monitoring por vehículo, fecha y `lap`; GPS dentro de la ventana de la vuelta
- [x] Construir el dataset vuelta × k (k ∈ {4, 6, 9, 12, 15}) con los cuatro grupos de variables, sin fuga
- [x] Partición temporal: train/val 6–25 ene, test 26 ene–1 feb; pliegues de ventana creciente (4)
- [x] Mover a `src/` las funciones que se reutilicen (carga, parseo, variables, partición)
- [x] Guardar en `data/processed/` y documentar en el notebook las cifras finales de filas por k
- [x] Contrastar la regla de 60 min con la distribución, la cerca de Tukey y la z robusta
- [x] Limpiar el GPS (puntos aislados a más de 144 km/h) y verificar la unidad de `speed`
- [x] Comparar variantes de preprocesamiento por validación (interpolación, GPS limpio, selección de variables, recorte del objetivo) y adoptar V3

## Fase 2 · Líneas base y modelos de regresión P1 (hasta 18 nov)

- [x] `03_lineas_base.ipynb`: B0, B1, B2, B3 por corte k
- [x] `04_regresion_P1.ipynb` con pipelines de scikit-learn (imputación, filtro de varianza y escalado ajustados solo en train)
- [x] Lineales: regresión lineal (y versión con gradiente descendente), Ridge, Lasso
- [x] Huber; KNN; árbol de decisión; SVR RBF
- [x] Ensambles: Random Forest, Bagging, AdaBoost, Gradient Boosting (rejilla ampliada a 81 combinaciones)
- [x] Ajuste de hiperparámetros con validación de ventana creciente; registrar los mejores en `results/tables/`
- [x] Evaluación en prueba (una sola vez): MAE, MedAE, RMSE, R², % error ≤ 2 y ≤ 5 min, por k y franja

## Fase 3 · Clasificación P2 y experimentos (hasta 19 nov)

- [x] `05_clasificacion_P2.ipynb`: logística regularizada, KNN, árbol, SVM, RF, AdaBoost, Gradient Boosting; exactitud, F1 atrasado, F1 macro, exactitud balanceada, AUC-PR, matriz de confusión
- [x] Sensibilidad de umbrales de P2 (5/3 min vs. terciles)
- [x] `06_analisis_resultados.ipynb` (antes 07):
  - [x] H1 y H3: curva MAE vs. k frente a líneas base
  - [x] H2: Ridge vs. RF y GB
  - [x] H4 y H6: ablación por grupos de variables; error por franja horaria y día
  - [x] H5: pérdida cuadrática vs. absoluta vs. Huber
  - [x] Sensibilidad con y sin la regla de 60 min
  - [x] IC por bootstrap sobre días y Wilcoxon entre pares de modelos
  - [x] Métricas frente a la meta de 0,80
- [x] Figuras y tablas finales en `results/figures/` y `results/tables/`
- [x] ~~(Opcional) `06_clustering_P4.ipynb`~~ retirado: los docentes pidieron un solo tipo de aprendizaje

## Fase 4 · Entregable 2 (20 nov)

- [x] Borrador del paper IEEE que **incluya el E1** más: metodología definitiva, EDA y preprocesamiento, experimentos, resultados, discusión y conclusiones
- [x] Sección de entorno computacional: Python, librerías y versiones, semilla, hiperparámetros
- [x] Revisión cruzada: cifras del paper contra las tablas de `results/`
- [x] Verificar que los notebooks corren de punta a punta en limpio (02 a 06 reejecutados con los datos reales el 18 nov)
- [x] Copiar el PDF y el DOCX del E2 en `docs/entregables/E2/`
- [x] Enlace al repositorio en el informe (resumen y sección de entorno)
- [ ] Fusionar `feat/entregable-2-v2` (que contiene `feat/entregable-2`) en `main` mediante PR revisado (el informe enlaza al repositorio)
- [ ] Marcar el commit entregado con `git tag e2` y `git push --tags`
- [ ] Subir el PDF a la plataforma y enlazar el repositorio de GitHub
- [ ] Confirmar la hora límite en la plataforma del curso

## Fase 5 · Presentación final (hasta 27 nov)

- [ ] Incorporar observaciones de los docentes al E2
- [ ] Paper final en PDF (`docs/entregables/final/`)
- [ ] Diapositivas en PDF: introducción, trabajos relacionados, metodología, resultados, discusión, conclusiones y trabajo futuro, demo
- [ ] (Recomendado) Demo interactiva en `demo/` (por ejemplo, elegir una vuelta y un corte k y ver la predicción frente al real)
- [ ] Ensayo con cronómetro: 15 min + 5 min de preguntas
- [ ] Repositorio final: README actualizado, resultados y entregables

## Pendientes abiertos

- [x] V9: `speed` está en km/h. Verificado en `results/tables/02_limpieza_gps.csv` y en la salida del notebook 02: el cociente mediano entre `speed` y la velocidad calculada con las coordenadas es 1,072 (correlación 0,65). Si `speed` estuviera en m/s, el cociente sería cercano a 0,28. El Anexo A del informe la marca como cerrada.
- [x] V17: las vueltas que empiezan lejos de la ruta (7,4 % a más de 1 km del inicio típico) se marcan con `gps_dist_inicio`; la verificación sigue como parcial en el Anexo A porque no se descartan.
- [x] Completar la referencia [2] del informe: T. Meyer-Hollatz, N. Schwarz y T. Werner, *Wirtschaftsinformatik 2023 Proceedings*, paper 328, https://aisel.aisnet.org/wi2023/73
- [x] Justificar los umbrales de P2 con la distribución observada (terciles: F1 macro 0,680 frente a 0,764)
- [x] Contrastar la regla de 60 min con la distribución, la cerca de Tukey y la z robusta (notebook 02)
- [ ] Ampliar trabajos relacionados de 4 a 6–10 referencias verificadas (DOI que abra) para el paper final. Requiere leer cada artículo y reescribir la Sección III.

## Puntos de decisión para el equipo

Decisiones que el informe del E2 ya refleja y conviene que el equipo valide antes de entregar.

| Punto | Estado | Detalle |
|---|---|---|
| Redacción de H1 y H3 | Resuelto | Dicen "se cumple desde k = 12": el error se mantiene en 2,62–2,67 min entre k = 4 y 9 y baja a 2,46 y 2,05 en k = 12 y 15. |
| Significancia por día | Resuelto | El informe distingue los p por vuelta de los p por MAE diario; con 7 días de prueba el mínimo posible por día es 0,016. |
| Vueltas extremas en prueba | Resuelto | Las métricas excluyen 11 vueltas de prueba con \|y\| > 60 min; con ellas el MAE de GB en k = 9 sube a 7,19 min. Está declarado en resultados y limitaciones. |
| Rejilla de Gradient Boosting | Resuelto | Se amplió a 81 combinaciones; en k = 9 y 15 elige tasa 0,01 con 1.000 árboles (compromiso tasa-árboles) con el mismo MAE de validación. |
| Cambio de distribución en P2 | Resuelto | "Atrasado" es el 33,9 % en entrenamiento y el 27,7 % en prueba; se menciona en el informe. |
| Variable `ctx_flota_30` | Por validar | Es la 3.ª más importante. Solo usa pasos de otros vehículos con hora real anterior al corte, incluidos días de prueba: es información disponible en el instante de predicción, no fuga. |
| Modelos lineales | Por validar | La selección de variables se eligió con Gradient Boosting y a los lineales les resta 0,1–0,15 min; está declarado como limitación. |
| Meta del 80 % | Por validar | El informe reporta con transparencia qué métricas superan 0,80 (Tabla IX); en k = 9 no lo logran R², exactitud ni F1 macro. |
| Revisión final del PDF | Por validar | Redacción, formato IEEE y cifras contra `results/tables/`. |
