"""Entrena y guarda los modelos que usa la demo.

Reentrena Gradient Boosting (el mejor modelo de P1) para cada corte k con los hiperparámetros
elegidos en validación (results/tables/04_hiperparametros.csv), sobre el mismo entrenamiento del
informe (6 a 25 de enero). No modifica notebooks ni resultados del proyecto.

Entradas:  data/raw/Times.csv y data/processed/corte_k{k}.pkl (generados por el notebook 02).
Salidas:   demo/modelos/modelo_k{k}.joblib y demo/datos/semana_prueba.pkl.

Uso, desde la raíz del repositorio:
    python demo/entrenar_modelos.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.datos import COL_DIF, COL_PROG, COL_REAL, aplicar_regla_extremos, cargar_times  # noqa: E402
from src.evaluacion import metricas_regresion  # noqa: E402
from src.modelos import REGRESORES, pipeline  # noqa: E402
from src.variables import CORTES, GRUPOS, columnas_modelo  # noqa: E402

MODELO = "Gradient Boosting"
MAE_INFORME = {4: 2.66, 6: 2.67, 9: 2.62, 12: 2.46, 15: 2.05}  # Tabla D2 del informe
DIR_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "demo" / "modelos"
DIR_DATOS = RAIZ / "demo" / "datos"
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def main():
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    hiper = pd.read_csv(RAIZ / "results" / "tables" / "04_hiperparametros.csv")

    predicciones = []
    for k in CORTES:
        t0 = time.time()
        x = pd.read_pickle(DIR_PROC / f"corte_k{k}.pkl")
        tr = x[x["conjunto"] == "train"].reset_index(drop=True)
        te = x[x["conjunto"] == "test"].reset_index(drop=True)
        cols = columnas_modelo(x)

        params = json.loads(hiper.query("k == @k and modelo == @MODELO")["mejores_parametros"].item())
        modelo = pipeline(cols, clone(REGRESORES[MODELO][0]).set_params(**params)).fit(tr[cols], tr["y"])

        pred = modelo.predict(te[cols])
        err = np.abs(te["y"].to_numpy() - pred)
        metricas = metricas_regresion(te["y"], pred)
        if abs(metricas["MAE"] - MAE_INFORME[k]) > 0.01:
            raise RuntimeError(f"k = {k}: MAE {metricas['MAE']:.3f} no coincide con el informe ({MAE_INFORME[k]})")

        por_franja = pd.Series(err).groupby(te["franja"]).mean().round(2).to_dict()
        por_dia = pd.Series(err).groupby(te["dia_semana"].map(dict(enumerate(DIAS)))).mean().round(2).to_dict()
        joblib.dump({
            "k": k,
            "modelo": modelo,
            "nombre_modelo": MODELO,
            "hiperparametros": params,
            "columnas": cols,
            "grupos": {g: columnas_modelo(x, (g,)) for g in GRUPOS},
            "metricas_prueba": metricas,
            "mae_por_franja": por_franja,
            "mae_por_dia": por_dia,
        }, DIR_MODELOS / f"modelo_k{k}.joblib", compress=3)

        predicciones.append(te.assign(pred=pred))
        print(f"k = {k:2d}: MAE {metricas['MAE']:.3f} (informe {MAE_INFORME[k]}), "
              f"R² {metricas['R2']:.3f}, {time.time() - t0:.0f} s")

    # Semana de prueba para la pestaña 1: curvas de los 18 puntos + variables de cada corte.
    vueltas = aplicar_regla_extremos(cargar_times(RAIZ / "data" / "raw" / "Times.csv"))
    ids_prueba = predicciones[0]["id"]
    curvas = vueltas[vueltas["id"].isin(ids_prueba)][["id", "fecha", "vehiculo", "lap"] + COL_PROG + COL_REAL + COL_DIF]
    variables = pd.concat(predicciones, ignore_index=True)
    pd.to_pickle({"curvas": curvas.reset_index(drop=True), "variables": variables}, DIR_DATOS / "semana_prueba.pkl")
    print(f"Semana de prueba: {len(curvas)} vueltas guardadas en {DIR_DATOS.relative_to(RAIZ)}")

    # Archivo de ejemplo para la pestaña «Probar con otros datos»: la semana de prueba en el formato original.
    times = pd.read_csv(RAIZ / "data" / "raw" / "Times.csv", dtype=str)
    ejemplo = times[pd.to_datetime(times["date"]) >= pd.Timestamp("2025-01-26")]
    ejemplo.to_csv(DIR_DATOS / "ejemplo_times.csv", index=False)
    print(f"Archivo de ejemplo: {len(ejemplo)} vueltas en formato Times.csv")


if __name__ == "__main__":
    main()
