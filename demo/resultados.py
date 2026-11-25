"""Tablas de resultados del proyecto para la pestaña «Resultados del proyecto».

Lee los CSV versionados en results/tables/ (los mismos del informe del E2); no necesita los datos crudos.
"""
from pathlib import Path

import pandas as pd

DIR_TABLAS = Path(__file__).resolve().parents[1] / "results" / "tables"
CORTES = (4, 6, 9, 12, 15)
LINEAS_BASE = ("B0 a tiempo", "B1 mediana hora-día", "B2 persistencia", "B3 lineal atraso k")


def _leer(nombre, **kwargs):
    return pd.read_csv(DIR_TABLAS / f"{nombre}.csv", encoding="utf-8", **kwargs)


def mejor_por_k():
    return _leer("04_mejor_modelo_por_k").set_index("k")


def mae_por_k():
    """MAE de prueba por modelo (filas) y corte k (columnas)."""
    t = _leer("04_mae_test_por_k").set_index("modelo")
    t.columns = [int(c) for c in t.columns]
    return t


def intervalos():
    return _leer("06_ic_bootstrap")


def importancia_grupos():
    return _leer("06_importancia_grupos_k9")


def ablacion():
    t = _leer("06_ablacion")
    return t[t["modelo"] == "Gradient Boosting"].pivot_table(index="variables", columns="k", values="MAE", sort=False)


def clasificacion_k9():
    return _leer("05_metricas_k9").set_index("modelo")


def matriz_confusion():
    return _leer("05_matriz_confusion_k9", index_col=0)


def meta_80():
    t = _leer("06_metricas_meta80")
    t["fila"] = t["métrica"] + " (" + t["tarea"].str.split().str[0] + ")"
    orden = ["R² (P1)", "Vueltas con error ≤ 5 min (P1)", "Exactitud (P2)", "F1 macro (P2)", "AUC-PR atrasado (P2)"]
    return t.pivot_table(index="fila", columns="k", values="valor").reindex(orden)


# Resultado de las hipótesis, tal como lo resume la Tabla X del informe del E2.
HIPOTESIS = [
    ("H1", "El atraso observado mejora la predicción frente a usar solo el horario.", "Se cumple desde k = 12",
     "parcial", "Solo horario 2,78 min frente a 2,51 (k = 12) y 2,10 (k = 15) al agregar el atraso; entre k = 4 y 9 "
                "la mejora es de 0,04 min o menos."),
    ("H2", "Los modelos de árboles superan a los lineales.", "Se cumple", "si",
     "Gradient Boosting y Random Forest superan a Ridge en los cinco cortes (Wilcoxon, p < 0,001)."),
    ("H3", "El error baja al observar más puntos de control.", "Se cumple desde k = 12", "parcial",
     "El MAE de Gradient Boosting se mantiene entre 2,62 y 2,67 min de k = 4 a 9 y baja a 2,46 y 2,05."),
    ("H4", "El GPS aporta información adicional.", "No se cumple", "no",
     "Agregar el GPS cambia el MAE en ± 0,01 min y su importancia es casi nula."),
    ("H5", "Las pérdidas absoluta y Huber rinden mejor que la cuadrática.", "Con matices", "parcial",
     "En prueba la absoluta da 2,56 frente a 2,62 min, pero no en validación; en el modelo lineal Huber mejora "
     "de 3,46 a 3,30 min."),
    ("H6", "El contexto de vehículo y flota mejora la predicción en las horas de más atraso.", "Parcial", "parcial",
     "Mejora 0,07 min en total, pero 0,24 min de 16 a 19 h y 0,33 min los sábados (k = 9)."),
]
