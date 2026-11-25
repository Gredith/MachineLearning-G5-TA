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


def filtros():
    """Conteo de vueltas en cada paso de la limpieza (notebook 02)."""
    return _leer("02_filtros", index_col=0)["vueltas"]


def limpieza_gps():
    """Conteos de la limpieza del GPS (notebook 02)."""
    return _leer("02_limpieza_gps", index_col=0)["valor"]
