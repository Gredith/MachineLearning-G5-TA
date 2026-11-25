"""Lógica de la demo: cargar modelos, predecir y explicar.

No depende de Streamlit ni del idioma: devuelve números y códigos, y app.py los convierte en texto.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DIR_DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR_DEMO.parent))
from src.datos import aplicar_regla_extremos, cargar_times  # noqa: E402
from src.variables import construir_corte  # noqa: E402

COLUMNAS_TIMES = ("id", "date", "vehicle_id_id", "lap", "start_time", "end_time", "difference_time")
FIN_ENTRENAMIENTO = pd.Timestamp("2025-01-25")
CORTES = (4, 6, 9, 12, 15)
UMBRAL_ATRASO, UMBRAL_ADELANTO = 5, 3  # mismos umbrales que P2 en el informe
CATEGORICAS = ("hor_dia_semana", "ctx_vehiculo")
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]  # claves de mae_por_dia


def cargar_modelos():
    return {k: joblib.load(DIR_DEMO / "modelos" / f"modelo_k{k}.joblib") for k in CORTES}


def cargar_semana_prueba():
    datos = pd.read_pickle(DIR_DEMO / "datos" / "semana_prueba.pkl")
    return datos["curvas"], datos["variables"]


def predecir(paquete, filas):
    return paquete["modelo"].predict(filas[paquete["columnas"]])


def clasificar(minutos):
    """Estado de la vuelta: 'atrasado', 'a tiempo' o 'adelantado' (los textos los traduce app.py)."""
    if minutos >= UMBRAL_ATRASO:
        return "atrasado"
    if minutos <= -UMBRAL_ADELANTO:
        return "adelantado"
    return "a tiempo"


def rango(pred, paquete):
    """Rango probable: la predicción ± el error medio del modelo en la semana de prueba, redondeado a minutos."""
    mae = paquete["metricas_prueba"]["MAE"]
    return int(round(pred - mae)), int(round(pred + mae))


def predecir_por_k(paquetes, variables, id_vuelta):
    """Predicción de la misma vuelta con el modelo de cada punto de corte."""
    filas = []
    for k in CORTES:
        fila = variables[(variables["k"] == k) & (variables["id"] == id_vuelta)]
        if len(fila):
            filas.append({"k": k, "pred": float(predecir(paquetes[k], fila)[0]),
                          "mae": paquetes[k]["metricas_prueba"]["MAE"]})
    return pd.DataFrame(filas)


def explicar(paquete, fila):
    """Cuánto cambia la predicción si el modelo no conociera cada grupo de variables (hor, atr, ctx, gps).

    El grupo se reemplaza por un valor típico del entrenamiento (la mediana que aplica el propio pipeline; las
    categorías pasan a 'desconocida'). Es la idea de la ablación del informe aplicada a una sola vuelta. Los
    efectos no suman exactamente la predicción.
    """
    base = float(predecir(paquete, fila)[0])
    efectos = {}
    for grupo, cols in paquete["grupos"].items():
        sin_grupo = fila.copy()
        for c in cols:
            sin_grupo[c] = "desconocida" if c in CATEGORICAS else np.nan
        efectos[grupo] = base - float(predecir(paquete, sin_grupo)[0])
    return pd.Series(efectos).sort_values(key=np.abs, ascending=False)


def senales(paquete, fila):
    """Situaciones en las que el modelo suele fallar más: lista de (código, valores) que app.py traduce."""
    mae = paquete["metricas_prueba"]["MAE"]
    franja = fila["franja"].iloc[0]
    dia = int(fila["dia_semana"].iloc[0])
    lista = []
    mae_franja = paquete["mae_por_franja"].get(franja)
    if mae_franja and mae_franja > 1.2 * mae:
        lista.append(("franja", {"franja": franja, "v": mae_franja}))
    mae_dia = paquete["mae_por_dia"].get(DIAS_ES[dia])
    if mae_dia and mae_dia > 1.2 * mae:
        lista.append(("dia", {"dia": dia, "v": mae_dia}))
    faltantes = int(fila["atr_n_faltantes"].iloc[0])
    if faltantes:
        lista.append(("faltantes", {"n": faltantes}))
    if pd.isna(fila["ctx_flota_30"].iloc[0]):
        lista.append(("sin_flota", {}))
    return lista


class ArchivoInvalido(ValueError):
    """Archivo que no se puede usar. `clave` identifica el motivo y `datos` completa el mensaje."""

    def __init__(self, clave, **datos):
        super().__init__(clave)
        self.clave = clave
        self.datos = datos


def leer_archivo_times(archivo):
    """Lee un CSV con el formato de Times.csv y aplica la misma regla de 60 min del proyecto."""
    try:
        columnas = set(pd.read_csv(archivo, nrows=0).columns)
    except Exception as e:
        raise ArchivoInvalido("error_lectura") from e
    faltan = [c for c in COLUMNAS_TIMES if c not in columnas]
    if faltan:
        raise ArchivoInvalido("error_columnas", faltan=", ".join(faltan))
    archivo.seek(0)
    if pd.read_csv(archivo, usecols=["id"]).empty:
        raise ArchivoInvalido("error_vacio")
    archivo.seek(0)
    try:
        vueltas = cargar_times(archivo)
    except Exception as e:
        raise ArchivoInvalido("error_listas") from e
    return aplicar_regla_extremos(vueltas)


def predecir_vueltas(vueltas, paquete):
    """Predicción en el punto 18 para cada vuelta del archivo, con la información disponible hasta k.

    Las vueltas sin punto 18 (en curso o sin registro) también se predicen: se les asigna un valor
    provisional en dif_18 solo para que construir_corte no las descarte. Su hora real en el punto 18 sigue
    vacía, así que ninguna variable usa ese valor, y después se borra del resultado.
    El GPS no viene en el archivo: sus variables quedan vacías y el modelo usa valores típicos (H4: no aporta).
    """
    k = paquete["k"]
    v = vueltas.copy()
    sin_final = set(v.loc[v["dif_18"].isna(), "id"])
    v.loc[v["dif_18"].isna(), "dif_18"] = 0.0
    x = construir_corte(v, k, {}, interpolar=True)
    for c in paquete["grupos"]["gps"]:
        x[c] = np.nan
    x["pred"] = predecir(paquete, x)
    x.loc[x["id"].isin(sin_final), "y"] = np.nan
    x["estado"] = x["pred"].map(clasificar)
    x["estado_real"] = x["y"].map(lambda v: clasificar(v) if pd.notna(v) else None)
    x["error"] = (x["y"] - x["pred"]).abs()
    return x


def dias_de_entrenamiento(vueltas):
    """Fechas del archivo que el modelo ya vio al entrenar (6 a 25 de enero de 2025)."""
    fechas = vueltas["fecha"].drop_duplicates()
    return sorted(fechas[(fechas >= pd.Timestamp("2025-01-06")) & (fechas <= FIN_ENTRENAMIENTO)])
