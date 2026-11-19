"""Pipelines de scikit-learn y espacios de hiperparámetros."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, LogisticRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .evaluacion import SEMILLA
from .variables import CATEGORICAS


def preprocesador(columnas):
    """Imputación por mediana, descarte de columnas constantes y escalado en numéricas; one-hot en categóricas.

    Todo se ajusta solo con las filas de entrenamiento que recibe el pipeline.
    """
    cat = [c for c in columnas if c in CATEGORICAS]
    num = [c for c in columnas if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("var", VarianceThreshold(0.0)),
                          ("esc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def pipeline(columnas, modelo):
    return Pipeline([("pre", preprocesador(columnas)), ("modelo", modelo)])


def _arbol(prof, semilla=SEMILLA):
    return DecisionTreeRegressor(max_depth=prof, random_state=semilla)


REGRESORES = {
    "Regresión lineal": (LinearRegression(), {}),
    "Regresión lineal (GD)": (
        SGDRegressor(max_iter=5000, tol=1e-4, early_stopping=False, random_state=SEMILLA),
        {"alpha": [1e-4, 1e-2, 1e-1], "eta0": [1e-3, 1e-2]},
    ),
    "Ridge": (Ridge(), {"alpha": [0.1, 1, 10, 100, 1000]}),
    "Lasso": (Lasso(max_iter=20000), {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5]}),
    "Huber": (HuberRegressor(max_iter=2000), {"epsilon": [1.35, 1.75, 2.5], "alpha": [1e-4, 1e-2, 1]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [10, 25, 50, 100], "weights": ["uniform", "distance"]}),
    "Árbol de decisión": (
        DecisionTreeRegressor(random_state=SEMILLA),
        {"max_depth": [3, 5, 8, None], "min_samples_leaf": [10, 30, 60]},
    ),
    "SVR (RBF)": (SVR(kernel="rbf"), {"C": [1, 10, 30], "epsilon": [0.5, 1.0], "gamma": ["scale", 0.01]}),
    "Random Forest": (
        RandomForestRegressor(n_estimators=300, random_state=SEMILLA),
        {"max_depth": [None, 10], "min_samples_leaf": [3, 10, 25], "max_features": [0.33, 1.0]},
    ),
    "Bagging": (
        BaggingRegressor(estimator=_arbol(None), n_estimators=150, random_state=SEMILLA),
        {"max_samples": [0.5, 1.0], "max_features": [0.5, 1.0], "estimator__min_samples_leaf": [5, 20]},
    ),
    "AdaBoost": (
        AdaBoostRegressor(estimator=_arbol(3), random_state=SEMILLA),
        {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1, 0.5], "estimator__max_depth": [3, 5]},
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(subsample=0.8, random_state=SEMILLA),
        {"n_estimators": [300, 600, 1000], "learning_rate": [0.01, 0.02, 0.05], "max_depth": [2, 3, 4],
         "min_samples_leaf": [5, 10, 20]},
    ),
}

CLASIFICADORES = {
    "Regresión logística": (
        LogisticRegression(max_iter=5000, class_weight="balanced"),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [10, 25, 50], "weights": ["uniform", "distance"]}),
    "Árbol de decisión": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEMILLA),
        {"max_depth": [3, 5, 8], "min_samples_leaf": [10, 30, 60]},
    ),
    "SVM (RBF)": (
        SVC(kernel="rbf", class_weight="balanced", random_state=SEMILLA),
        {"C": [0.3, 1, 10], "gamma": ["scale", 0.01]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEMILLA),
        {"max_depth": [None, 10], "min_samples_leaf": [3, 10, 25]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=SEMILLA),
                           random_state=SEMILLA),
        {"n_estimators": [100, 300], "learning_rate": [0.05, 0.3]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(subsample=0.8, random_state=SEMILLA),
        {"n_estimators": [200, 500], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4]},
    ),
}


def rejilla(grid):
    return {f"modelo__{k}": v for k, v in grid.items()}


def _ajustar_y_predecir(modelo, cols, ent, ev):
    return clone(modelo).fit(ent[cols], ent["y"]).predict(ev[cols])


def validar_y_probar(modelo, cols, tr, te, n_pliegues=4, n_jobs=-1):
    """MAE medio de los pliegues de ventana creciente y predicciones en prueba con el modelo ajustado en todo train."""
    tareas = [(tr[tr["bloque"] < f], tr[tr["bloque"] == f]) for f in range(n_pliegues)] + [(tr, te)]
    preds = Parallel(n_jobs=n_jobs)(delayed(_ajustar_y_predecir)(modelo, cols, a, b) for a, b in tareas)
    mae_val = [np.mean(np.abs(p - b["y"].to_numpy())) for p, (_, b) in zip(preds[:-1], tareas[:-1])]
    return float(np.mean(mae_val)), preds[-1]
