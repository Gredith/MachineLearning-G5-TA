"""Métricas, intervalos por bootstrap de días y pruebas de Wilcoxon."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, median_absolute_error, r2_score)

SEMILLA = 42


def metricas_regresion(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.abs(y - p)
    return {
        "MAE": mean_absolute_error(y, p),
        "MedAE": median_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p),
        "pct_err_2": 100 * np.mean(e <= 2),
        "pct_err_5": 100 * np.mean(e <= 5),
        "n": len(y),
    }


def metricas_clasificacion(y, p, prob_atrasado=None, clase_atrasado="atrasado"):
    r = {
        "F1_atrasado": f1_score(y, p, labels=[clase_atrasado], average="macro", zero_division=0),
        "F1_macro": f1_score(y, p, average="macro", zero_division=0),
        "exactitud_bal": balanced_accuracy_score(y, p),
    }
    if prob_atrasado is not None:
        r["AUC_PR_atrasado"] = average_precision_score(np.asarray(y) == clase_atrasado, prob_atrasado)
    return r


def ic_bootstrap_dias(error_abs, dias, n=2000, alfa=0.05, semilla=SEMILLA):
    """IC del MAE remuestreando días completos con reemplazo."""
    df = pd.DataFrame({"e": np.asarray(error_abs, float), "d": np.asarray(dias)})
    por_dia = df.groupby("d")["e"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(por_dia), size=(n, len(por_dia)))
    muestras = por_dia[idx, 0].sum(axis=1) / por_dia[idx, 1].sum(axis=1)
    return np.quantile(muestras, [alfa / 2, 1 - alfa / 2])


def wilcoxon_pares(errores, dias):
    """Wilcoxon por pares sobre errores absolutos por vuelta y sobre el MAE diario emparejado."""
    filas = []
    nombres = list(errores)
    por_dia = pd.DataFrame(errores).assign(d=np.asarray(dias)).groupby("d").mean()
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            ea, eb = np.asarray(errores[a]), np.asarray(errores[b])
            filas.append({
                "modelo_a": a, "modelo_b": b,
                "MAE_a": ea.mean(), "MAE_b": eb.mean(),
                "p_vuelta": wilcoxon(ea, eb, zero_method="zsplit").pvalue,
                "p_dia": wilcoxon(por_dia[a], por_dia[b]).pvalue,
            })
    return pd.DataFrame(filas)
