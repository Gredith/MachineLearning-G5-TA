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
├── demo/                 # demostración interactiva (opcional)
└── docs/
    ├── CONTEXTO.md       # contexto del proyecto y registro de decisiones
    ├── TAREAS.md         # tareas pendientes hasta la entrega final
    ├── CONVENCIONES.md   # ramas, commits, notebooks y buenas prácticas
    └── entregables/      # E1, E2 y entrega final (PDF/DOCX)
```

| Notebook | Qué hace | Tiempo aprox. (2 núcleos) |
|---|---|---|
| `01_auditoria_datos_v2` | Auditoría de los datos (E1) | 3 min |
| `02_preprocesamiento_variables_por_corte` | Limpieza, unión Times y Monitoring, dataset vuelta × k, partición | 15 s |
| `03_lineas_base` | Líneas base B0 a B3 | 10 s |
| `04_regresion_P1` | 12 regresores con búsqueda en rejilla y evaluación en prueba | 20 min |
| `05_clasificacion_P2` | 7 clasificadores, referencias y sensibilidad de umbrales | 10 min |
| `06_clustering_P4` | Perfiles de puntualidad (complementario) | 1 min |
| `07_analisis_resultados` | Hipótesis H1 a H6: IC por bootstrap, Wilcoxon, ablación, pérdidas, regla de 60 min | 6 min |

| Módulo de `src/` | Contenido |
|---|---|
| `datos.py` | Carga y parseo de Times y Monitoring, regla de 60 min |
| `variables.py` | Construcción de las variables en el corte k sin fuga |
| `particion.py` | Partición temporal y pliegues de ventana creciente |
| `modelos.py` | Pipelines y espacios de hiperparámetros |
| `evaluacion.py` | Métricas, bootstrap por días y Wilcoxon |
| `graficos.py` | Estilo y colores de las figuras |

## Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Coloca `Times.csv` y `Monitoring.csv` en `data/raw/`.
2. Ejecuta los notebooks de `notebooks/` en orden numérico (02 a 07 son los del E2; cada uno usa las salidas del anterior). Semilla fija: 42.

Desde la terminal, en la carpeta `notebooks/`:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 02_preprocesamiento_variables_por_corte.ipynb
```

## Resultados del E2

Gradient Boosting es el mejor modelo en validación y en prueba en los cinco cortes. En k = 9 (mitad de la vuelta) predice la desviación del punto 18 con un MAE de 2,64 min en la semana de prueba, frente a 3,20 min de la mejor línea base (mediana por hora y día) y 3,93 min de la persistencia. El horario y el atraso observado explican casi todo el desempeño; el contexto de vehículo y flota mejora poco, sobre todo a la hora punta de la tarde y los sábados, y las variables GPS no aportan. Detalle en `docs/CONTEXTO.md` y en el informe del E2.

## Entregables

| Entrega | Fecha | Puntos | Estado |
|---|---|---|---|
| E1: Propuesta y antecedentes | 6 nov | 5 | Entregado ([docs/entregables/E1](docs/entregables/E1)) |
| E2: Implementación y evaluación experimental | 20 nov | 5 | Listo para entregar ([docs/entregables/E2](docs/entregables/E2)) |
| Presentación final (exposición, solución, paper) | 27 nov | 10 | Pendiente |

## Integrantes

Jorge Vicente Mateo · Jose Gutierrez Lopez · Yaser Fernandez Ccerhuayo · Letizia Castillo Soto

## Documentación

- [Contexto y decisiones](docs/CONTEXTO.md)
- [Tareas pendientes](docs/TAREAS.md)
- [Convenciones de trabajo](docs/CONVENCIONES.md) (leer antes de contribuir)
