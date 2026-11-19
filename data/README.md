# Datos

Los archivos los proporcionan los docentes del curso y **no se suben al repositorio** (están en `.gitignore`).

| Archivo | Dónde va | Contenido |
|---|---|---|
| `Times.csv` | `data/raw/` | 4.707 vueltas (una fila por vuelta) de 45 vehículos, 6 ene a 1 feb 2025, con 18 puntos de control en listas |
| `Monitoring.csv` | `data/raw/` | 1.630.431 puntos GPS de los mismos vehículos, 1 ene a 1 feb 2025 |

`data/processed/` guarda las tablas derivadas: `corte_k{k}.pkl` (dataset vuelta × corte k con la regla de 60 min), `corte_k{k}_sin_regla.pkl` (para la sensibilidad) y las predicciones de prueba de los notebooks 03 y 04 (`03_pred_test.pkl`, `04_pred_test.pkl`). También se ignora en git: se regenera ejecutando los notebooks 02 a 04.

## Variables principales

| Variable | Archivo | Descripción |
|---|---|---|
| `id`, `date`, `time` | Ambos | Identificador, fecha y hora. En Times, `time` es igual al primer `start_time` |
| `vehicle_id_id` | Ambos | Vehículo (45) |
| `lap` | Times | Contador diario de vueltas por vehículo (1 a 8) |
| `start_time` | Times | Lista de 18 horas programadas |
| `end_time` | Times | Lista de 18 horas reales; `00:00:00` = sin registro |
| `difference_time` | Times | Lista de 18 enteros en minutos (real − programado); `-` = sin registro |
| `lap` | Monitoring | 0 = fuera de operación; > 0 = número de vuelta |
| `latitude`, `longitude` | Monitoring | Coordenadas GPS |
| `speed` | Monitoring | Velocidad, aparentemente en km/h (V9 parcial) |
| `bearing` | Monitoring | Rumbo en grados, 0° al norte, sentido horario |

Unión entre archivos: por vehículo, fecha y `lap`. Más detalle en [`docs/CONTEXTO.md`](../docs/CONTEXTO.md).
