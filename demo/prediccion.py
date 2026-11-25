"""Lógica de la demo: cargar modelos, predecir, explicar y traducir el resultado a lenguaje simple.

No depende de Streamlit, así que se puede probar por separado y reutilizar en otras pestañas.
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
NOMBRE_GRUPO = {
    "hor": "Horario (hora de inicio y día)",
    "atr": "Atraso observado hasta k",
    "ctx": "Vehículo y otros buses",
    "gps": "GPS",
}


def cargar_modelos():
    return {k: joblib.load(DIR_DEMO / "modelos" / f"modelo_k{k}.joblib") for k in CORTES}


def cargar_semana_prueba():
    datos = pd.read_pickle(DIR_DEMO / "datos" / "semana_prueba.pkl")
    return datos["curvas"], datos["variables"]


def predecir(paquete, filas):
    return paquete["modelo"].predict(filas[paquete["columnas"]])


def clasificar(minutos):
    if minutos >= UMBRAL_ATRASO:
        return "atrasado"
    if minutos <= -UMBRAL_ADELANTO:
        return "adelantado"
    return "a tiempo"


def explicar(paquete, fila):
    """Cuánto cambia la predicción si el modelo no conociera cada grupo de variables.

    El grupo se reemplaza por un valor típico del entrenamiento (la mediana que aplica el propio
    pipeline; las categorías pasan a 'desconocida'). Es la misma idea de la ablación del informe,
    aplicada a una sola vuelta. Los efectos no suman exactamente la predicción.
    """
    base = float(predecir(paquete, fila)[0])
    efectos = {}
    for grupo, cols in paquete["grupos"].items():
        sin_grupo = fila.copy()
        for c in cols:
            sin_grupo[c] = "desconocida" if c in CATEGORICAS else np.nan
        efectos[grupo] = base - float(predecir(paquete, sin_grupo)[0])
    return pd.Series(efectos).rename(NOMBRE_GRUPO).sort_values(key=np.abs, ascending=False)


def num(valor, decimales=1):
    """Número con coma decimal, como en el informe."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def _minutos(valor):
    m = abs(round(valor))
    return f"{m} min" if m != 1 else "1 min"


def mensaje(pred, k):
    estado = clasificar(pred)
    if estado == "atrasado":
        texto = f"llegará al punto 18 con unos **{_minutos(pred)} de atraso**"
    elif estado == "adelantado":
        texto = f"llegará al punto 18 con unos **{_minutos(pred)} de adelanto**"
    else:
        sentido = "de atraso" if pred > 0.5 else "de adelanto" if pred < -0.5 else ""
        texto = f"llegará al punto 18 **a tiempo** ({_minutos(pred)} {sentido})".replace(" )", ")")
    return f"Con lo observado hasta el punto {k}, se estima que el bus {texto}.", estado


def confianza(paquete):
    m = paquete["metricas_prueba"]
    return (f"En la semana de prueba, con información hasta el punto {paquete['k']}, el modelo se equivocó "
            f"en promedio **{num(m['MAE'])} min**, y en el **{m['pct_err_5']:.0f} %** de las vueltas el error "
            f"fue de 5 min o menos.")


def advertencias(paquete, fila, franja, dia):
    """Avisos cuando la vuelta se parece a los casos donde el modelo suele fallar más."""
    avisos = []
    mae = paquete["metricas_prueba"]["MAE"]
    mae_franja = paquete["mae_por_franja"].get(franja)
    mae_dia = paquete["mae_por_dia"].get(dia)
    if mae_franja and mae_franja > 1.2 * mae:
        avisos.append(f"En la franja **{franja}** el error medio sube a **{num(mae_franja)} min**.")
    if mae_dia and mae_dia > 1.2 * mae:
        avisos.append(f"Los **{dia}s** el error medio sube a **{num(mae_dia)} min**.")
    faltantes = int(fila["atr_n_faltantes"].iloc[0])
    if faltantes:
        avisos.append(f"Faltan registros en **{faltantes}** de los primeros {paquete['k']} puntos; "
                      "se completaron con los puntos vecinos.")
    if pd.isna(fila["ctx_flota_30"].iloc[0]):
        avisos.append("No hubo otros buses en los 30 min previos, así que el modelo no conoce la congestión del momento.")
    return avisos


class ArchivoInvalido(ValueError):
    """Error con un mensaje pensado para mostrarse al usuario de la demo."""


def leer_archivo_times(archivo):
    """Lee un CSV con el formato de Times.csv y aplica la misma regla de 60 min del proyecto."""
    try:
        columnas = set(pd.read_csv(archivo, nrows=0).columns)
    except Exception as e:
        raise ArchivoInvalido("No se pudo leer el archivo como CSV.") from e
    faltan = [c for c in COLUMNAS_TIMES if c not in columnas]
    if faltan:
        raise ArchivoInvalido("Faltan columnas obligatorias: " + ", ".join(faltan) + ".")
    archivo.seek(0)
    try:
        vueltas = cargar_times(archivo)
    except Exception as e:
        raise ArchivoInvalido("Las columnas start_time, end_time y difference_time deben tener 18 valores entre llaves, "
                              "como en Times.csv: {05:45:00,05:47:00,...}.") from e
    if vueltas.empty:
        raise ArchivoInvalido("El archivo no tiene filas.")
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
