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
        "lat": m["latitude"].to_numpy(dtype=float),
        "lon": m["longitude"].to_numpy(dtype=float),
        "speed": m["speed"].to_numpy(dtype=float),
    })


VEL_MAX_GPS = 144


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def velocidad_calculada(gps):
    """Velocidad (km/h) entre cada punto y el anterior del mismo vehículo y día, y el intervalo en segundos."""
    g = gps.sort_values(["vehiculo", "fecha", "t"])
    mismo = (g["vehiculo"].to_numpy()[1:] == g["vehiculo"].to_numpy()[:-1]) & \
            (g["fecha"].to_numpy()[1:] == g["fecha"].to_numpy()[:-1])
    dist = _haversine_km(g["lat"].to_numpy()[:-1], g["lon"].to_numpy()[:-1], g["lat"].to_numpy()[1:], g["lon"].to_numpy()[1:])
    dt = np.diff(g["t"].to_numpy()) * 60
    vel = np.where(mismo & (dt > 0), dist / np.where(dt > 0, dt, 1) * 3600, np.nan)
    g = g.assign(dist_km=np.r_[np.nan, np.where(mismo, dist, np.nan)],
                 dt_s=np.r_[np.nan, np.where(mismo, dt, np.nan)],
                 vel_calc=np.r_[np.nan, vel])
    return g


def limpiar_saltos_gps(gps, vel_max=VEL_MAX_GPS):
    """Elimina los puntos aislados fuera de lugar: se llega a ellos y se sale de ellos a más de vel_max km/h."""
    g = velocidad_calculada(gps)
    salto = (g["vel_calc"] > vel_max) & (g["vel_calc"].shift(-1) > vel_max)
    return g.loc[~salto, gps.columns].reset_index(drop=True), int(salto.sum())


def aplicar_regla_extremos(v, umbral=UMBRAL_EXTREMO):
    """Marca como faltante todo punto con |difference_time| > umbral (diferencia y hora real)."""
    v = v.copy()
    dif = v[COL_DIF].to_numpy()
    extremo = np.abs(dif) > umbral
    v[COL_DIF] = np.where(extremo, np.nan, dif)
    v[COL_REAL] = np.where(extremo, np.nan, v[COL_REAL].to_numpy())
    return v
