"""Construcción del dataset vuelta x corte k sin información posterior al corte."""
import re
import warnings

import numpy as np
import pandas as pd

from .datos import COL_DIF, COL_PROG, COL_REAL, PUNTOS, VEL_DETENIDO, _haversine_km

CORTES = (4, 6, 9, 12, 15)
VENTANA_FLOTA = 30
VENTANA_GPS_RECIENTE = 10
CATEGORICAS = ("hor_dia_semana", "ctx_vehiculo")
GRUPOS = ("hor", "atr", "ctx", "gps")
NOMBRE_GRUPO = {"hor": "Horario", "atr": "Atraso observado", "ctx": "Vehículo y contexto", "gps": "GPS"}
FRANJAS = [4, 7, 10, 13, 16, 19, 22]


def franja_horaria(hora):
    etiquetas = [f"{a:02d}-{b:02d} h" for a, b in zip(FRANJAS[:-1], FRANJAS[1:])]
    return pd.cut(hora, bins=FRANJAS, right=False, labels=etiquetas).astype(str)


def _ultimo_valido(m):
    """Índice de la última columna no nula por fila (-1 si no hay ninguna)."""
    valido = ~np.isnan(m)
    idx = m.shape[1] - 1 - np.argmax(valido[:, ::-1], axis=1)
    return np.where(valido.any(axis=1), idx, -1)


def _pendiente(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _pendiente_mc(m)


def _pendiente_mc(m):
    x = np.arange(1, m.shape[1] + 1, dtype=float)
    valido = ~np.isnan(m)
    n = valido.sum(axis=1)
    xm = np.where(valido, x, np.nan)
    mx, my = np.nanmean(xm, axis=1), np.nanmean(m, axis=1)
    cov = np.nansum((xm - mx[:, None]) * (m - my[:, None]), axis=1)
    var = np.nansum((xm - mx[:, None]) ** 2, axis=1)
    return np.where((n >= 2) & (var > 0), cov / np.where(var > 0, var, 1), np.nan)


def pasadas_por_punto(v):
    """Formato largo de todos los pasos registrados por punto de control (para el contexto de flota)."""
    real = v[COL_REAL].to_numpy()
    dif = v[COL_DIF].to_numpy()
    fila, punto = np.nonzero(~np.isnan(real) & ~np.isnan(dif))
    return pd.DataFrame({
        "fecha": v["fecha"].to_numpy()[fila],
        "vehiculo": v["vehiculo"].to_numpy()[fila],
        "t": real[fila, punto],
        "dif": dif[fila, punto],
    })


def _suma_ventana(t_ref, d_ref, t_ini, t_fin):
    """Suma y conteo de d_ref con t_ref en [t_ini, t_fin); t_ref debe venir ordenado."""
    acum = np.concatenate([[0.0], np.cumsum(d_ref)])
    a = np.searchsorted(t_ref, t_ini, side="left")
    b = np.searchsorted(t_ref, t_fin, side="left")
    return acum[b] - acum[a], b - a


def _contexto_flota(v, t_corte, pasadas):
    suma = np.zeros(len(v))
    n = np.zeros(len(v))
    fechas = v["fecha"].to_numpy()
    vehiculos = v["vehiculo"].to_numpy()
    for fecha, p in pasadas.groupby("fecha"):
        p = p.sort_values("t")
        filas = np.flatnonzero(fechas == fecha)
        s, c = _suma_ventana(p["t"].to_numpy(), p["dif"].to_numpy(), t_corte[filas] - VENTANA_FLOTA, t_corte[filas])
        suma[filas] += s
        n[filas] += c
        for veh, pv in p.groupby("vehiculo"):
            sub = filas[vehiculos[filas] == veh]
            if len(sub):
                s, c = _suma_ventana(pv["t"].to_numpy(), pv["dif"].to_numpy(), t_corte[sub] - VENTANA_FLOTA, t_corte[sub])
                suma[sub] -= s
                n[sub] -= c
    return np.where(n > 0, suma / np.where(n > 0, n, 1), np.nan), n


def _vuelta_anterior(v, t_corte):
    clave = v[["vehiculo", "fecha", "lap"]].copy()
    previa = v[["vehiculo", "fecha", "lap", "real_18", "dif_18"]].copy()
    previa["lap"] = previa["lap"] + 1
    unida = clave.merge(previa, on=["vehiculo", "fecha", "lap"], how="left")
    termino = unida["real_18"].to_numpy() < t_corte
    return np.where(termino, unida["dif_18"].to_numpy(), np.nan)


def indexar_gps(gps):
    """Diccionario (vehículo, fecha, lap) -> (horas ordenadas, velocidades, km del primer punto al inicio típico).

    El inicio típico es la mediana de los primeros puntos de todas las vueltas; sirve para marcar vueltas
    que empiezan lejos de la ruta (V17).
    """
    gps = gps.sort_values(["vehiculo", "fecha", "lap", "t"])
    primeros = gps.groupby(["vehiculo", "fecha", "lap"], sort=False)[["lat", "lon"]].first()
    lat0, lon0 = primeros["lat"].median(), primeros["lon"].median()
    return {
        (veh, pd.Timestamp(fecha), lap): (g["t"].to_numpy(), g["speed"].to_numpy(),
                                          float(_haversine_km(g["lat"].iloc[0], g["lon"].iloc[0], lat0, lon0)))
        for (veh, fecha, lap), g in gps.groupby(["vehiculo", "fecha", "lap"], sort=False)
    }


def _gps_hasta_corte(v, t_corte, indice):
    salida = np.full((len(v), 5), np.nan)
    salida[:, 0] = 0
    claves = zip(v["vehiculo"], v["fecha"], v["lap"])
    for i, clave in enumerate(claves):
        if clave not in indice:
            continue
        t, vel, dist_inicio = indice[clave]
        fin = np.searchsorted(t, t_corte[i], side="right")
        if fin == 0:
            continue
        reciente = vel[np.searchsorted(t, t_corte[i] - VENTANA_GPS_RECIENTE, side="left"):fin]
        salida[i] = [fin, vel[:fin].mean(), (vel[:fin] < VEL_DETENIDO).mean(),
                     reciente.mean() if len(reciente) else np.nan, dist_inicio]
    return salida


def interpolar_internos(m):
    """Interpola linealmente los faltantes que quedan entre dos puntos observados de la misma fila.

    Solo usa valores de la propia fila y del rango observado (no extrapola), así que aplicado a los puntos
    1..k no introduce información posterior al corte.
    """
    m = m.copy()
    x = np.arange(m.shape[1])
    for i in np.flatnonzero(np.isnan(m).any(axis=1)):
        ok = ~np.isnan(m[i])
        if ok.sum() < 2:
            continue
        a, b = np.flatnonzero(ok)[[0, -1]]
        hueco = ~ok & (x > a) & (x < b)
        m[i, hueco] = np.interp(x[hueco], x[ok], m[i, ok])
    return m


def construir_corte(v, k, indice_gps, pasadas=None, interpolar=False):
    """Observaciones con punto 18 válido y variables disponibles al pasar por el punto de control k.

    Con interpolar=True, los faltantes internos de los puntos 1..k se interpolan antes de calcular las
    variables de atraso (los indicadores de faltante siguen marcando el dato original).
    """
    pasadas = pasadas_por_punto(v) if pasadas is None else pasadas
    v = v[v["dif_18"].notna()].reset_index(drop=True)
    prog = v[COL_PROG].to_numpy()
    real = v[COL_REAL].to_numpy()
    dif = v[COL_DIF].to_numpy()
    obs = dif[:, :k]
    falta = np.isnan(obs)
    if interpolar:
        obs = interpolar_internos(obs)

    # Instante de predicción: hora real en k; si no se registró, la programada.
    t_corte = np.where(np.isnan(real[:, k - 1]), prog[:, k - 1], real[:, k - 1])

    inicio = prog[:, 0]
    x = pd.DataFrame({
        "id": v["id"], "fecha": v["fecha"], "vehiculo": v["vehiculo"], "lap": v["lap"], "k": k,
        "y": dif[:, PUNTOS - 1],
        "hora_inicio": inicio / 60,
        "dia_semana": v["fecha"].dt.dayofweek,
    })
    x["franja"] = franja_horaria(x["hora_inicio"])

    x["hor_inicio_sin"] = np.sin(2 * np.pi * inicio / 1440)
    x["hor_inicio_cos"] = np.cos(2 * np.pi * inicio / 1440)
    x["hor_inicio_h"] = inicio / 60
    x["hor_dia_semana"] = x["dia_semana"].astype(str)
    x["hor_restante"] = prog[:, PUNTOS - 1] - prog[:, k - 1]
    x["hor_total"] = prog[:, PUNTOS - 1] - prog[:, 0]
    x["hor_no_creciente"] = (np.diff(prog, axis=1) < 0).any(axis=1).astype(int)
    for j in range(k + 1, PUNTOS + 1):
        x[f"hor_tramo_{j}"] = prog[:, j - 1] - prog[:, j - 2]

    for j in range(1, k + 1):
        x[f"atr_d_{j}"] = obs[:, j - 1]
        x[f"atr_falta_{j}"] = falta[:, j - 1].astype(int)
    ult = _ultimo_valido(obs)
    filas = np.arange(len(v))
    hay = ult >= 0
    x["atr_ultimo"] = np.where(hay, obs[filas, np.maximum(ult, 0)], np.nan)
    x["atr_ultimo_cont"] = np.where(hay, (real - prog)[filas, np.maximum(ult, 0)], np.nan)
    x["atr_dist_ultimo"] = np.where(hay, k - 1 - ult, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x["atr_media3"] = np.nanmean(obs[:, -3:], axis=1)
        x["atr_max"] = np.nanmax(np.where(np.isnan(obs), -np.inf, obs), axis=1)
        x["atr_min"] = np.nanmin(np.where(np.isnan(obs), np.inf, obs), axis=1)
    x.loc[~np.isfinite(x["atr_max"]), "atr_max"] = np.nan
    x.loc[~np.isfinite(x["atr_min"]), "atr_min"] = np.nan
    x["atr_pendiente"] = _pendiente(obs)
    x["atr_n_faltantes"] = falta.sum(axis=1)

    x["ctx_vehiculo"] = v["vehiculo"]
    x["ctx_lap"] = v["lap"]
    previa = _vuelta_anterior(v, t_corte)
    x["ctx_previa_final"] = previa
    x["ctx_previa_disp"] = (~np.isnan(previa)).astype(int)
    flota, n_flota = _contexto_flota(v, t_corte, pasadas)
    x["ctx_flota_30"] = flota
    x["ctx_flota_n"] = n_flota

    g = _gps_hasta_corte(v, t_corte, indice_gps)
    x["gps_n"] = g[:, 0]
    x["gps_vel_media"] = g[:, 1]
    x["gps_prop_detenido"] = g[:, 2]
    x["gps_vel_reciente"] = g[:, 3]
    x["gps_dist_inicio"] = g[:, 4]
    return x


def columnas_modelo(df, grupos=GRUPOS):
    return [c for c in df.columns if c.split("_")[0] in grupos]


def columnas_detalle(df):
    """Variables de detalle por punto que la selección descarta: indicadores de faltante y tramos programados
    individuales (su información queda resumida en atr_n_faltantes, atr_dist_ultimo y hor_restante)."""
    return [c for c in df.columns if re.fullmatch(r"atr_falta_\d+|hor_tramo_\d+", c)]
