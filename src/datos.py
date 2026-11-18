"""Carga y parseo de Times.csv y Monitoring.csv."""
import numpy as np
import pandas as pd

PUNTOS = 18
UMBRAL_EXTREMO = 60
LAP_FUERA_DE_OPERACION = 0
VEL_DETENIDO = 2


def _columnas(n, prefijo):
    return [f"{prefijo}_{j}" for j in range(1, n + 1)]


COL_PROG = _columnas(PUNTOS, "prog")
COL_REAL = _columnas(PUNTOS, "real")
COL_DIF = _columnas(PUNTOS, "dif")


def _a_minutos(valores):
    return (pd.to_timedelta(valores).total_seconds() / 60).to_numpy()


def _lista(serie):
    return serie.str.strip("{}").str.split(",", expand=True).to_numpy()


def cargar_times(ruta):
    """Una fila por vuelta con 18 columnas de hora programada, real y diferencia (en minutos)."""
    t = pd.read_csv(ruta, dtype={"vehicle_id_id": str})
    prog = _a_minutos(_lista(t["start_time"]).ravel()).reshape(-1, PUNTOS)
    fin = _lista(t["end_time"])
    real = _a_minutos(fin.ravel()).reshape(-1, PUNTOS).copy()
    real[fin == "00:00:00"] = np.nan
    dif = pd.DataFrame(_lista(t["difference_time"])).replace("-", np.nan).astype(float).to_numpy()

    v = pd.DataFrame({
        "id": t["id"].to_numpy(),
        "fecha": pd.to_datetime(t["date"]),
        "vehiculo": t["vehicle_id_id"].astype(str).to_numpy(),
        "lap": t["lap"].to_numpy(),
    })
    partes = [pd.DataFrame(m, columns=c) for m, c in ((prog, COL_PROG), (real, COL_REAL), (dif, COL_DIF))]
    return pd.concat([v] + partes, axis=1)


def cargar_monitoring(ruta):
    """Puntos GPS en operación (lap > 0), sin filas repetidas, con la hora en minutos."""
    m = pd.read_csv(ruta, dtype={"vehicle_id_id": str})
    m = m[m["lap"] != LAP_FUERA_DE_OPERACION]
    m = m.drop_duplicates(subset=[c for c in m.columns if c != "id"])
    return pd.DataFrame({
        "vehiculo": m["vehicle_id_id"].astype(str).to_numpy(),
        "fecha": pd.to_datetime(m["date"]).to_numpy(),
        "lap": m["lap"].to_numpy(),
        "t": _a_minutos(m["time"].to_numpy()),
        "speed": m["speed"].to_numpy(dtype=float),
    })


def aplicar_regla_extremos(v, umbral=UMBRAL_EXTREMO):
    """Marca como faltante todo punto con |difference_time| > umbral (diferencia y hora real)."""
    v = v.copy()
    dif = v[COL_DIF].to_numpy()
    extremo = np.abs(dif) > umbral
    v[COL_DIF] = np.where(extremo, np.nan, dif)
    v[COL_REAL] = np.where(extremo, np.nan, v[COL_REAL].to_numpy())
    return v
