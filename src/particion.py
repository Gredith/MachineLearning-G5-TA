"""Partición temporal por días y pliegues de ventana creciente."""
import numpy as np
import pandas as pd

INICIO_PRUEBA = pd.Timestamp("2025-01-26")
DIAS_INICIALES = 8
DIAS_POR_BLOQUE = 3
N_PLIEGUES = 4


def asignar_particion(df):
    """Agrega 'conjunto' (train/test) y 'bloque' (-1 = solo entrenamiento, 0..3 = validación del pliegue)."""
    df = df.copy()
    df["conjunto"] = np.where(df["fecha"] >= INICIO_PRUEBA, "test", "train")
    dias = np.sort(df.loc[df["conjunto"] == "train", "fecha"].unique())
    orden = {d: i for i, d in enumerate(dias)}
    pos = df["fecha"].map(orden)
    bloque = np.floor((pos - DIAS_INICIALES) / DIAS_POR_BLOQUE)
    df["bloque"] = np.where(df["conjunto"] == "test", np.nan, np.clip(bloque, -1, N_PLIEGUES - 1))
    return df


def pliegues(bloques):
    """Lista de (índices de entrenamiento, índices de validación) para un vector de bloques de train."""
    bloques = np.asarray(bloques)
    return [(np.flatnonzero(bloques < f), np.flatnonzero(bloques == f)) for f in range(N_PLIEGUES)]
