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
├── results/              # figuras y tablas finales
├── demo/                 # demostración interactiva (opcional)
└── docs/
    ├── CONTEXTO.md       # contexto del proyecto y registro de decisiones
    ├── TAREAS.md         # tareas pendientes hasta la entrega final
    ├── CONVENCIONES.md   # ramas, commits, notebooks y buenas prácticas
    └── entregables/      # E1, E2 y entrega final (PDF/DOCX)
```

## Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Coloca `Times.csv` y `Monitoring.csv` en `data/raw/`.
2. Abre los notebooks de `notebooks/` en orden numérico y ejecuta todas las celdas. Semilla fija: 42.

## Entregables

| Entrega | Fecha | Puntos | Estado |
|---|---|---|---|
| E1: Propuesta y antecedentes | 6 nov | 5 | Entregado ([docs/entregables/E1](docs/entregables/E1)) |
| E2: Implementación y evaluación experimental | 20 nov | 5 | En curso |
| Presentación final (exposición, solución, paper) | 27 nov | 10 | Pendiente |

## Integrantes

Jorge Vicente Mateo · Jose Gutierrez Lopez · Yaser Fernandez Ccerhuayo · Letizia Castillo Soto

## Documentación

- [Contexto y decisiones](docs/CONTEXTO.md)
- [Tareas pendientes](docs/TAREAS.md)
- [Convenciones de trabajo](docs/CONVENCIONES.md) (leer antes de contribuir)
