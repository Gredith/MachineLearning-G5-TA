"""Entrena y guarda los modelos que usa la demo.

Reentrena el mejor modelo de las pruebas para cada corte k: Gradient Boosting con pérdida de Huber,
con los hiperparámetros elegidos en validación (results/tables/04_hiperparametros.csv) y el alpha de
Huber elegido en el notebook 07 (results/tables/07_huber.csv), sobre el mismo entrenamiento del
informe (6 a 25 de enero). No modifica notebooks ni resultados del proyecto.

Entradas:  data/raw/Times.csv y data/raw/Monitoring.csv.
Salidas:   data/processed/corte_k{k}.pkl (solo si faltan), demo/modelos/modelo_k{k}.joblib,
           demo/datos/semana_prueba.pkl y demo/datos/ejemplo_times.csv.

Si faltan los datos por corte, se construyen con las mismas funciones de src/ que usa el notebook 02
(regla de 60 min, GPS sin saltos, interpolación y selección de variables), sin necesidad de Jupyter.

Uso, desde la raíz del repositorio:
    python demo/entrenar_modelos.py               # usa data/processed si ya existe
    python demo/entrenar_modelos.py --regenerar   # reconstruye data/processed desde data/raw
"""
import argparse
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
from src.datos import (COL_DIF, COL_PROG, COL_REAL, aplicar_regla_extremos, cargar_monitoring,  # noqa: E402
                       cargar_times, limpiar_saltos_gps)
from src.evaluacion import metricas_regresion  # noqa: E402
from src.modelos import REGRESORES, pipeline  # noqa: E402
from src.particion import asignar_particion  # noqa: E402
from src.variables import (CORTES, GRUPOS, columnas_detalle, columnas_modelo, construir_corte,  # noqa: E402
                           indexar_gps, pasadas_por_punto)

MODELO = "Gradient Boosting"
NOMBRE_MODELO = "Gradient Boosting (pérdida de Huber)"
DIR_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "demo" / "modelos"
DIR_DATOS = RAIZ / "demo" / "datos"
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def preparar_cortes(destino=DIR_PROC, regenerar=False):
    """Dataset vuelta × corte k (como el notebook 02, sección 5) si todavía no existe en `destino`."""
    rutas = [destino / f"corte_k{k}.pkl" for k in CORTES]
    if not regenerar and all(r.exists() for r in rutas):
        print(f"Datos por corte: se usan los de {destino.relative_to(RAIZ)}")
        return
    for archivo in ("Times.csv", "Monitoring.csv"):
        if not (RAIZ / "data" / "raw" / archivo).exists():
            sys.exit(f"Falta data/raw/{archivo}. Copia los datos del curso en data/raw/ y vuelve a ejecutar.")
    t0 = time.time()
    print("Datos por corte: construyendo desde data/raw (menos de 1 min)...")
    base = aplicar_regla_extremos(cargar_times(RAIZ / "data" / "raw" / "Times.csv"))
    gps, _ = limpiar_saltos_gps(cargar_monitoring(RAIZ / "data" / "raw" / "Monitoring.csv"))
    indice, pasadas = indexar_gps(gps), pasadas_por_punto(base)
    destino.mkdir(parents=True, exist_ok=True)
    for k, ruta in zip(CORTES, rutas):
        x = asignar_particion(construir_corte(base, k, indice, pasadas, interpolar=True))
        x.drop(columns=columnas_detalle(x)).to_pickle(ruta)
    print(f"Datos por corte listos en {time.time() - t0:.0f} s")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentos.add_argument("--regenerar", action="store_true", help="reconstruye data/processed desde data/raw")
    preparar_cortes(regenerar=argumentos.parse_args().regenerar)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    hiper = pd.read_csv(RAIZ / "results" / "tables" / "04_hiperparametros.csv")
    huber = pd.read_csv(RAIZ / "results" / "tables" / "07_huber.csv").query("modelo == 'GB Huber'").set_index("k")

    predicciones = []
    for k in CORTES:
        t0 = time.time()
        x = pd.read_pickle(DIR_PROC / f"corte_k{k}.pkl")
        tr = x[x["conjunto"] == "train"].reset_index(drop=True)
        te = x[x["conjunto"] == "test"].reset_index(drop=True)
        cols = columnas_modelo(x)

        params = json.loads(hiper.query("k == @k and modelo == @MODELO")["mejores_parametros"].item())
        params |= {"loss": "huber", "alpha": float(huber.loc[k, "parametro"].split("= ")[1])}
        modelo = pipeline(cols, clone(REGRESORES[MODELO][0]).set_params(**params)).fit(tr[cols], tr["y"])

        pred = modelo.predict(te[cols])
        err = np.abs(te["y"].to_numpy() - pred)
        metricas = metricas_regresion(te["y"], pred)
        esperado = huber.loc[k, "MAE"]  # MAE de prueba del notebook 07
        if abs(metricas["MAE"] - esperado) > 0.01:
            raise RuntimeError(f"k = {k}: MAE {metricas['MAE']:.3f} no coincide con el notebook 07 ({esperado:.3f})")

        por_franja = pd.Series(err).groupby(te["franja"]).mean().round(2).to_dict()
        por_dia = pd.Series(err).groupby(te["dia_semana"].map(dict(enumerate(DIAS)))).mean().round(2).to_dict()
        joblib.dump({
            "k": k,
            "modelo": modelo,
            "nombre_modelo": NOMBRE_MODELO,
            "hiperparametros": params,
            "columnas": cols,
            "grupos": {g: columnas_modelo(x, (g,)) for g in GRUPOS},
            "metricas_prueba": metricas,
            "mae_por_franja": por_franja,
            "mae_por_dia": por_dia,
        }, DIR_MODELOS / f"modelo_k{k}.joblib", compress=3)

        predicciones.append(te.assign(pred=pred))
        print(f"k = {k:2d}: MAE {metricas['MAE']:.3f} (notebook 07: {esperado:.3f}), "
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
