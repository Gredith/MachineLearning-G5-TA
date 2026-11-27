# Predicción del adelanto y atraso de buses urbanos en su último punto de control

Tarea académica final del curso **Aprendizaje de Máquina (1INF02)**, PUCP, semestre 2026-2. Grupo 5.
Docentes: Dr. César A. Beltrán Castañón y Dr. Edwin Alvarez Mamani.

## Problema

Con datos de un sistema de transporte inteligente de Cusco, se predice cuántos minutos de adelanto o atraso tendrá una vuelta de bus en su **punto de control 18** (el último), usando el horario programado y lo observado hasta un punto intermedio *k* ∈ {4, 6, 9, 12, 15}.

- **P1 (principal):** regresión de `difference_time` del punto 18. Métrica principal: MAE en minutos.
- **P2 (variante):** clasificación en atrasado (≥ 5 min), a tiempo y adelantado (≥ 3 min de adelanto).
- Validación temporal por días: entrenamiento 6–25 ene 2025, prueba 26 ene–1 feb 2025.

Datos: `Times.csv` (4.707 vueltas, 45 vehículos, 27 días, 18 puntos de control) y `Monitoring.csv` (1,63 millones de puntos GPS). Ver [`data/README.md`](data/README.md).

## Estructura

```
├── data/                 # CSV originales (raw) y derivados (processed); no se versionan
├── notebooks/            # notebooks numerados en orden de ejecución
├── src/                  # funciones reutilizables extraídas de los notebooks
├── results/              # figuras y tablas finales (prefijo = notebook que las genera)
├── demo/                 # demo interactiva con Streamlit (ver demo/README.md)
└── docs/
    ├── CONTEXTO.md       # contexto del proyecto y registro de decisiones
    ├── TAREAS.md         # tareas pendientes hasta la entrega final
    ├── CONVENCIONES.md   # ramas, commits, notebooks y buenas prácticas
    └── entregables/      # E1, E2 y entrega final (PDF/DOCX)
```

El proyecto usa solo **aprendizaje supervisado** (regresión y clasificación).

| Notebook | Qué hace | Tiempo aprox. (12 núcleos) |
|---|---|---|
| `01_auditoria_datos_v2` | Auditoría de los datos (E1) | 3 min |
| `02_preprocesamiento_variables_por_corte` | Limpieza (contraste de la regla de 60 min, GPS, unidades), flujo de la limpieza, variantes de preprocesamiento, faltantes e imputación, experimentos adicionales, dataset vuelta × k, partición | 6 min |
| `03_lineas_base` | Líneas base B0 a B3 | 10 s |
| `04_regresion_P1` | 12 regresores con búsqueda en rejilla y evaluación en prueba | 15–25 min |
| `05_clasificacion_P2` | 7 clasificadores, referencias y sensibilidad de umbrales | 5–10 min |
| `06_analisis_resultados` | Hipótesis H1 a H6: IC por bootstrap, Wilcoxon, ablación, pérdidas, regla de 60 min, métricas frente al 80 % y figuras del paper final (incluidos el esquema por corte, los umbrales de P2 y la clasificación en k = 9) | 6 min |
| `07_mejoras_modelos` | Mejoras sobre los mejores modelos sin reejecutar 04–06: pérdida de Huber, red neuronal (MLP) con búsqueda equivalente, pesos balanceados en P2 y diagnóstico de sobreajuste/subajuste | 20 min |

| Módulo de `src/` | Contenido |
|---|---|
| `datos.py` | Carga y parseo de Times y Monitoring, regla de 60 min, velocidad calculada y limpieza del GPS |
| `variables.py` | Construcción de las variables en el corte k sin fuga, interpolación de faltantes internos y selección de variables |
| `particion.py` | Partición temporal y pliegues de ventana creciente |
| `modelos.py` | Pipelines (imputación, filtro de varianza, escalado, one-hot) y espacios de hiperparámetros |
| `evaluacion.py` | Métricas, bootstrap por días y Wilcoxon |
| `graficos.py` | Estilo y colores de las figuras |

## Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Coloca `Times.csv` y `Monitoring.csv` en `data/raw/`.
2. Ejecuta los notebooks de `notebooks/` en orden numérico (02 a 06 son los del E2 y 07 el del paper final; cada uno usa las salidas del anterior). Semilla fija: 42.

Desde la terminal, en la carpeta `notebooks/`:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 02_preprocesamiento_variables_por_corte.ipynb
```

## Demo interactiva

Aplicación con Streamlit para la presentación final, en español e inglés y con tema claro u oscuro: predice una vuelta de la semana de prueba eligiendo el punto de corte k, permite subir un archivo con el formato de `Times.csv`, resume los resultados del informe y explica el modelo y la limpieza de los datos. Detalle en [`demo/README.md`](demo/README.md).

```bash
python demo/entrenar_modelos.py   # prepara los datos si faltan y entrena (menos de 1 min)
cd demo
python -m streamlit run app.py
```

## Resultados del E2

El mismo problema se aborda con dos enfoques supervisados: **regresión** de los minutos de desviación en el punto 18 (P1) y **clasificación** en atrasado, a tiempo o adelantado (P2), con las mismas observaciones y variables.

Gradient Boosting es el mejor modelo en validación y en prueba en los cinco cortes. En k = 9 (mitad de la vuelta) predice la desviación del punto 18 con un MAE de 2,62 min (R² 0,75; 86,8 % de vueltas con error ≤ 5 min), frente a 3,20 min de la mejor línea base, y con 2,05 min en k = 15 (R² 0,84). La clasificación alcanza una exactitud de 0,79 en k = 9 y 0,83 en k = 15. El preprocesamiento reforzado (contraste de la regla de 60 min, limpieza del GPS, interpolación y selección de variables) mantiene el error con un 30 a 42 % menos de variables. Detalle en `docs/CONTEXTO.md` y en el informe del E2.

**Paper final (notebook 07):** la pérdida de Huber mejora levemente a Gradient Boosting (en k = 9, 2,58 frente a 2,62 min en prueba); una red neuronal con el mismo esfuerzo de ajuste (81 combinaciones) no lo supera, los pesos balanceados empeoran el F1 macro y el diagnóstico de ajuste no muestra sobreajuste en el modelo elegido. La demo usa Gradient Boosting con Huber: en la semana de prueba, al menos 87,6 % de vueltas con error ≤ 5 min en todos los cortes y exactitud de 0,80 o más desde k = 6.

## Entregables

| Entrega | Fecha | Puntos | Estado |
|---|---|---|---|
| E1: Propuesta y antecedentes | 6 nov | 5 | Entregado ([docs/entregables/E1](docs/entregables/E1)) |
| E2: Implementación y evaluación experimental | 20 nov | 5 | Entregado ([docs/entregables/E2](docs/entregables/E2)) |
| Presentación final (exposición, solución, paper) | 27 nov | 10 | Paper en revisión ([docs/entregables/final](docs/entregables/final)); faltan las diapositivas |

## Integrantes

Jorge Vicente Mateo · Jose Gutierrez Lopez · Yaser Fernandez Ccerhuayo · Letizia Castillo Soto

## Documentación

- [Contexto y decisiones](docs/CONTEXTO.md)
- [Tareas pendientes](docs/TAREAS.md)
- [Convenciones de trabajo](docs/CONVENCIONES.md) (leer antes de contribuir)
