"""Estilo común de figuras y colores fijos por entidad."""
import matplotlib.pyplot as plt

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS = "#8a8984"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"

COLOR_MODELO = {
    "Gradient Boosting": PALETA[0],
    "Ridge": PALETA[1],
    "Random Forest": PALETA[2],
    "SVR (RBF)": PALETA[3],
    "Bagging": PALETA[4],
    "KNN": PALETA[6],
    "B2 persistencia": GRIS,
    "B3 lineal atraso k": "#b5b4ae",
    "B1 mediana hora-día": "#5f5e5a",
    "B0 a tiempo": "#c9c8c2",
}


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TEXTO_SEC, "xtick.color": TEXTO_SEC,
        "ytick.color": TEXTO_SEC, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=PALETA),
    })


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta)
    plt.show()
    plt.close(fig)
