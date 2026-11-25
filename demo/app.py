"""Demo interactiva: predicción del adelanto o atraso de un bus en su último punto de control.

Ejecutar desde la carpeta demo/ (antes: python entrenar_modelos.py):
    streamlit run app.py
"""
import io
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import prediccion as p
import resultados as r

# Paleta índigo y ámbar: índigo (5B4A9B) para lo observado y ámbar (C4741D) para la predicción;
# el par pasa las pruebas de contraste y daltonismo sobre el fondo gris perla (F7F7F9).
AZUL, AZUL_CLARO, NARANJA, GRIS, TINTA = "#5b4a9b", "#b3aad6", "#c4741d", "#8a8984", "#1c1838"
SIN_REGISTRO = "Sin registro (completado por el modelo)"
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = {1: "ene", 2: "feb"}
FUENTE = "IBM Plex Sans"

st.set_page_config(page_title="Predicción de atraso de buses", page_icon=":material/directions_bus:", layout="wide",
                   initial_sidebar_state="auto")


@st.cache_resource
def modelos():
    return p.cargar_modelos()


@st.cache_data
def semana():
    return p.cargar_semana_prueba()


def nombre_fecha(f):
    return f"{DIAS[f.day_of_week]} {f.day} {MESES[f.month]}"


def elegir_al_azar(vueltas):
    fila = vueltas.sample(1).iloc[0]
    st.session_state.update(fecha=fila["fecha"], vehiculo=fila["vehiculo"], lap=int(fila["lap"]))


def grafico_curva(curva, k, pred, mae, revelar=True):
    puntos = np.arange(1, 19)
    dif = curva[[f"dif_{j}" for j in puntos]].to_numpy(dtype=float)
    obs = pd.DataFrame({"punto": puntos[:k], "minutos": dif[:k], "serie": "Observado hasta k"})
    capas = []

    zona = pd.DataFrame({"y0": [-p.UMBRAL_ADELANTO], "y1": [p.UMBRAL_ATRASO]})
    capas.append(alt.Chart(zona).mark_rect(color="#ecebf4", opacity=0.9).encode(
        y="y0:Q", y2="y1:Q"))
    capas.append(alt.Chart(pd.DataFrame({"t": ["zona «a tiempo» (−3 a +5 min)"], "x": [1], "y": [4.4]}))
                 .mark_text(align="left", color="#52514e", fontSize=11).encode(x="x:Q", y="y:Q", text="t:N"))
    capas.append(alt.Chart(pd.DataFrame({"x": [k]})).mark_rule(strokeDash=[4, 4], color=GRIS).encode(x="x:Q"))

    # Puntos sin registro entre dos puntos observados: el modelo los completa por interpolación lineal
    # (src/variables.py, interpolar_internos). Se dibujan como tramo punteado con marcador hueco.
    obs["tramo"] = "observado"
    huecos = []
    validos = np.flatnonzero(~np.isnan(dif[:k]))
    if len(validos) >= 2:
        completo = np.interp(np.arange(k), validos, dif[:k][validos])
        faltan = [i for i in range(validos[0], validos[-1]) if np.isnan(dif[i])]
        for i in faltan:
            a = max(j for j in validos if j < i)
            b = min(j for j in validos if j > i)
            huecos.append(pd.DataFrame({"punto": puntos[a:b + 1], "minutos": completo[a:b + 1],
                                        "serie": SIN_REGISTRO, "tramo": f"hueco_{a}"}))
    huecos = pd.concat(huecos, ignore_index=True).drop_duplicates() if huecos else pd.DataFrame(
        columns=["punto", "minutos", "serie", "tramo"])

    series = [obs, huecos]
    if revelar:
        despues = pd.DataFrame({"punto": puntos[k - 1:], "minutos": dif[k - 1:], "serie": "Lo que pasó después",
                                "tramo": "despues"})
        series.append(despues.dropna())
    lineas = pd.concat(series, ignore_index=True)
    lineas["minutos"] = lineas["minutos"].astype(float)
    llenas = lineas[(lineas["serie"] == "Observado hasta k") |
                    ((lineas["serie"] == "Lo que pasó después") & (lineas["punto"] != k))].dropna()
    marcas = pd.concat([llenas, pd.DataFrame({"punto": [18], "minutos": [pred], "serie": ["Predicción del modelo"]})]
                       + ([pd.DataFrame({"punto": [18], "minutos": [dif[-1]], "serie": ["Valor real"]})] if revelar else []),
                       ignore_index=True)
    huecas = huecos[np.isnan(dif[huecos["punto"].astype(int) - 1])] if len(huecos) else huecos

    # Cada serie conserva siempre su color; la leyenda solo muestra las que aparecen en esta vuelta.
    estilo = {"Observado hasta k": (AZUL, "circle"), SIN_REGISTRO: (AZUL_CLARO, "circle"),
              "Lo que pasó después": (GRIS, "circle"), "Predicción del modelo": (NARANJA, "circle"),
              "Valor real": (TINTA, "diamond")}
    presentes = set(lineas["serie"]) | set(marcas["serie"])
    dominio = [s for s in estilo if s in presentes]
    color = alt.Color("serie:N", title=None,
                      scale=alt.Scale(domain=dominio, range=[estilo[s][0] for s in dominio]),
                      legend=alt.Legend(orient="top", labelLimit=320, labelFontSize=12))
    forma = alt.Shape("serie:N", scale=alt.Scale(domain=dominio, range=[estilo[s][1] for s in dominio]), legend=None)
    tooltip = [alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("punto:Q", title="Punto de control", format="d"),
               alt.Tooltip("minutos:Q", title="Minutos (− adelanto, + atraso)", format=".1f")]
    eje_x = alt.X("punto:Q", title="Punto de control", scale=alt.Scale(domain=[0.5, 18.5], nice=False),
                  axis=alt.Axis(values=list(range(1, 19)), format="d", grid=False))
    # Margen de 2 min por encima y por debajo de todo lo que se dibuja, para que nada quede al borde.
    valores = np.r_[dif[~np.isnan(dif)], pred - mae, pred + mae, -p.UMBRAL_ADELANTO, p.UMBRAL_ATRASO]
    y_min, y_max = np.floor(valores.min()) - 2, np.ceil(valores.max()) + 2
    eje_y = alt.Y("minutos:Q", title="Minutos respecto al horario (− adelanto, + atraso)",
                  scale=alt.Scale(domain=[y_min, y_max], nice=False))

    capas.append(alt.Chart(lineas).mark_line(strokeWidth=2).encode(
        x=eje_x, y=eje_y, color=color, detail="tramo:N",
        strokeDash=alt.condition(alt.datum.serie == "Observado hasta k", alt.value([1, 0]), alt.value([5, 4]))))
    if len(huecas):
        capas.append(alt.Chart(huecas).mark_point(filled=False, size=90, strokeWidth=2).encode(
            x=eje_x, y=eje_y, color=color, tooltip=tooltip))
    banda = pd.DataFrame({"punto": [18], "lo": [pred - mae], "hi": [pred + mae]})
    capas.append(alt.Chart(banda).mark_rule(color=NARANJA, strokeWidth=2).encode(
        x="punto:Q", y="lo:Q", y2="hi:Q",
        tooltip=[alt.Tooltip("lo:Q", title="Desde (min)", format=".1f"), alt.Tooltip("hi:Q", title="Hasta (min)", format=".1f")]))
    capas.append(alt.Chart(marcas).mark_point(filled=True, size=110, stroke="#f7f7f9", strokeWidth=2).encode(
        x=eje_x, y=eje_y, color=color, shape=forma, tooltip=tooltip))
    capas.append(alt.Chart(pd.DataFrame({"x": [k], "y": [y_min + 0.6], "t": [f"ahora (punto {k})"]}))
                 .mark_text(align="left", dx=5, color="#52514e", fontSize=11).encode(x="x:Q", y="y:Q", text="t:N"))
    return alt.layer(*capas).properties(height=420).configure(font=FUENTE, background="transparent").configure_view(
        stroke=None).configure_axis(labelColor="#403b5c", titleColor="#403b5c", gridColor="#e4e2ec",
                                    domainColor="#c3bfd3", titleFontWeight=500).configure_legend(labelColor="#1c1838")


def grafico_explicacion(efectos):
    datos = efectos.reset_index()
    datos.columns = ["grupo", "efecto"]
    datos["sentido"] = np.where(datos["efecto"] >= 0, "Empuja hacia atraso", "Empuja hacia adelanto")
    datos["etiqueta"] = datos["efecto"].map(lambda v: f"{v:+.1f} min".replace(".", ","))
    lim = max(1.0, datos["efecto"].abs().max() * 1.35)
    base = alt.Chart(datos).encode(
        y=alt.Y("grupo:N", title=None, sort=list(datos["grupo"]),
                axis=alt.Axis(labelFontSize=13, labelPadding=12, ticks=False, domain=False)),
        x=alt.X("efecto:Q", title="Cambio en la predicción (min)", scale=alt.Scale(domain=[-lim, lim]),
                axis=alt.Axis(labelFontSize=12, titleFontSize=12, titlePadding=10, tickCount=5,
                              labelExpr="replace(format(datum.value, '.1f'), '.', ',')")))
    barras = base.mark_bar(height=22, cornerRadiusEnd=2).encode(
        color=alt.Color("sentido:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Empuja hacia atraso", "Empuja hacia adelanto"], range=[NARANJA, AZUL])),
        tooltip=[alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("efecto:Q", title="Efecto (min)", format="+.2f")])
    texto = base.mark_text(dx=alt.expr(alt.expr.if_(alt.datum.efecto >= 0, 8, -8)), color="#1c1838", fontSize=13, font="IBM Plex Mono",
                           align=alt.expr(alt.expr.if_(alt.datum.efecto >= 0, "left", "right"))).encode(text="etiqueta:N")
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6c6883").encode(x="x:Q")
    return alt.layer(cero, barras, texto).properties(height=70 * len(datos)).configure(
        font=FUENTE, background="transparent").configure_view(stroke=None).configure_axis(
        labelColor="#403b5c", titleColor="#403b5c", gridColor="#e4e2ec", domainColor="#c3bfd3",
        labelLimit=300, titleFontWeight=500).configure_legend(labelFontSize=12, labelColor="#1c1838")


ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --tinta: #1c1838; --tinta-2: #403b5c; --tinta-3: #6c6883;
    --papel: #f7f7f9; --papel-2: #ebe9f3; --regla: #d6d3e3; --teal: #5b4a9b;
}
.stApp { background: var(--papel); color: var(--tinta); font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block-container { max-width: 980px; padding-top: 2.2rem; }
header[data-testid="stHeader"] { background: transparent; }
.stApp p, .stApp label, .stApp li { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

/* Barra lateral: columna de controles */
section[data-testid="stSidebar"] { background: var(--papel-2); border-right: 1px solid var(--regla); }
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* Cabecera tipo diario */
.stApp .cabecera { border-bottom: 3px double var(--tinta); padding-bottom: 22px; margin-bottom: 14px; }
.stApp .antetitulo {
    font: 500 0.78rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--teal); margin: 0 0 22px 0;
}
.stApp .cabecera h1 {
    font-family: 'Newsreader', Georgia, serif; font-weight: 500; font-size: 2.6rem; line-height: 1.08;
    letter-spacing: -0.01em; color: var(--tinta); margin: 0; padding: 0;
}
.stApp .bajada {
    font-family: 'Newsreader', Georgia, serif; font-style: italic; font-size: 1.15rem;
    color: var(--tinta-2); margin: 16px 0 0 0;
}

/* Pestañas como índice de secciones */
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid var(--regla); }
.stTabs [data-baseweb="tab"] { padding: 10px 0; }
.stTabs [data-baseweb="tab"] p {
    font: 500 0.78rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.1em; text-transform: uppercase;
}

/* Secciones separadas solo por reglas finas */
.stApp .seccion { border-top: 2px solid var(--tinta); margin: 60px 0 22px 0; padding-top: 16px;
           display: flex; align-items: baseline; gap: 16px; }
.stApp .seccion .num { font: 500 1rem 'IBM Plex Mono', monospace; color: var(--teal); }
.stApp .seccion .nombre { font: 700 1.15rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.06em;
                   text-transform: uppercase; color: var(--tinta); }
.stApp .entradilla { font-family: 'Newsreader', Georgia, serif; font-size: 1.2rem; line-height: 1.5;
              color: var(--tinta); margin: 4px 0 16px 0; }

/* Resultado */
.stApp .fecha-linea { font: 500 0.72rem 'IBM Plex Mono', monospace; letter-spacing: 0.06em;
               text-transform: uppercase; color: var(--tinta-3); margin: 0 0 18px 0; }
.stApp .titular { font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; line-height: 1.3;
           font-weight: 500; color: var(--tinta); margin: 0 0 28px 0; }
.stApp .titular strong { font-weight: 600; box-shadow: inset 0 -0.32em 0 var(--resalte); }
.stApp .cifras { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--regla);
          border-bottom: 1px solid var(--regla); }
.stApp .cifra { padding: 16px 16px 18px 0; }
.stApp .cifra + .cifra { border-left: 1px solid var(--regla); padding-left: 16px; }
.stApp .cifra .rotulo { font: 500 0.68rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.1em;
                 text-transform: uppercase; color: var(--tinta-3); }
.stApp .cifra .valor { font: 400 2rem 'IBM Plex Mono', monospace; color: var(--tinta); margin-top: 4px;
                letter-spacing: -0.02em; }
.stApp .cifra .valor small { font-size: 0.95rem; color: var(--tinta-3); margin-left: 4px; }
.stApp .cifra .valor.estado { display: flex; align-items: center; gap: 10px;
    font: 500 1.55rem 'IBM Plex Sans', sans-serif; letter-spacing: 0; padding-top: 4px; }
.stApp .estado i { width: 10px; height: 10px; display: inline-block; }

/* Notas al margen en lugar de cajas de aviso */
.stApp .nota { border-left: 2px solid #c4741d; padding: 4px 0 4px 16px; margin: 18px 0; }
.stApp .nota .rotulo { font: 600 0.66rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.12em;
                text-transform: uppercase; color: #8a4f10; }
.stApp .nota p { margin: 2px 0 0 0; color: var(--tinta-2); }
.stApp .pie { font: 0.84rem/1.55 'IBM Plex Sans', sans-serif; color: var(--tinta-3); margin: 10px 0 6px 0; }

/* Nota metodológica */
.stApp .metodologia { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px 40px;
               font-size: 0.9rem; color: var(--tinta-2); }
.stApp .metodologia b { color: var(--tinta); font-weight: 600; }

/* Tablas editoriales */
.stApp .subtitulo { font: 600 0.95rem 'IBM Plex Sans', sans-serif; color: var(--tinta); margin: 30px 0 10px 0; }
.stApp .tabla { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 4px 0 6px 0; }
.stApp .tabla th, .stApp .tabla td { border: 0; border-bottom: 1px solid var(--regla); padding: 9px 10px;
                                     background: transparent; }
.stApp .tabla thead th { font: 500 0.7rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.1em;
                         color: var(--tinta-3); text-align: right;
                         border-bottom: 2px solid var(--tinta); }
.stApp .tabla th.fila { text-align: left; font-weight: 500; color: var(--tinta); }
.stApp .tabla td { text-align: right; font-family: 'IBM Plex Mono', monospace; color: var(--tinta-2); }
.stApp .tabla td.ok { color: var(--teal); font-weight: 600; }

/* Hipótesis */
.stApp .lista-hipotesis { border-top: 1px solid var(--regla); }
.stApp .hipotesis { display: grid; grid-template-columns: 64px 1fr; gap: 18px; padding: 20px 0;
                    border-bottom: 1px solid var(--regla); }
.stApp .hipotesis .codigo { font: 500 1.5rem 'IBM Plex Mono', monospace; color: var(--teal); line-height: 1.2; }
.stApp .hipotesis .enunciado { font-family: 'Newsreader', Georgia, serif; font-size: 1.2rem; line-height: 1.35;
                               color: var(--tinta); margin: 0 0 8px 0; }
.stApp .hipotesis .veredicto { font: 600 0.72rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.1em;
                               text-transform: uppercase; margin: 0 0 6px 0; display: inline-block;
                               padding-bottom: 2px; border-bottom: 2px solid; }
.stApp .veredicto.si { color: #2f7a57; border-color: #2f8f6a; }
.stApp .veredicto.parcial { color: #8a4f10; border-color: #c4741d; }
.stApp .veredicto.no { color: #6c6883; border-color: #a9a6b8; }
.stApp .hipotesis .evidencia { font-size: 0.9rem; line-height: 1.55; color: var(--tinta-2); margin: 0; }

/* Menú de Streamlit oculto: evita cambios de tema durante la exposición */
[data-testid="stMainMenu"], #MainMenu { display: none !important; }

/* Archivo en uso */
.stApp .en-uso { border-left: 3px solid var(--teal); background: var(--papel-2); padding: 14px 18px; }
.stApp .en-uso .rotulo { font: 600 0.68rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.12em;
                         text-transform: uppercase; color: var(--teal); }
.stApp .en-uso .nombre { font: 500 1.05rem 'IBM Plex Mono', monospace; color: var(--tinta); margin-top: 4px; }
.stApp .en-uso .detalle { font-size: 0.85rem; color: var(--tinta-3); margin-top: 2px; }
.stApp .stButton button[kind="primary"] { background: var(--tinta); color: var(--papel); padding: 0.5rem 1.4rem; }
.stApp .stButton button p { white-space: nowrap; }
.stApp .stButton button[kind="primary"]:hover { background: var(--teal); border-color: var(--teal); }

/* Carga de archivos y tabla de datos con el mismo lenguaje sin esquinas */
.stApp [data-testid="stFileUploaderDropzone"] { border-radius: 0; border: 1px dashed var(--tinta-3);
                                                background: var(--papel); }
.stApp [data-testid="stFileUploaderDropzone"] button { border-radius: 0; }
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stDataFrame"] > div { border-radius: 0; }
.stApp .stDownloadButton button { border-radius: 0; border: 1px solid var(--tinta); background: transparent;
                                  font: 500 0.8rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.06em;
                                  text-transform: uppercase; }
.stApp .stDownloadButton button:hover { background: var(--tinta); color: var(--papel); }
.stApp code { font-family: 'IBM Plex Mono', monospace; font-size: 0.85em; background: var(--papel-2);
              color: var(--tinta); padding: 1px 4px; border-radius: 0; }

/* Controles */
.stApp .control-titulo { font: 600 0.7rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.14em;
                  text-transform: uppercase; color: var(--tinta); border-bottom: 1px solid var(--tinta);
                  padding-bottom: 6px; margin-bottom: 10px; }
.stButton button { border-radius: 0; border: 1px solid var(--tinta); background: transparent;
                   font: 500 0.8rem 'IBM Plex Sans', sans-serif; letter-spacing: 0.06em; text-transform: uppercase; }
.stButton button:hover { background: var(--tinta); color: var(--papel); border-color: var(--tinta); }
.stApp div[data-baseweb="select"] > div { background: var(--papel); }
.stApp div[data-baseweb="select"], .stApp div[data-baseweb="select"] div { border-radius: 0 !important; }
.stApp .cifra .rotulo { min-height: 2.3em; }
@media (max-width: 700px) {
    .stApp .cifras { grid-template-columns: repeat(2, 1fr); }
    .stApp .cifra:nth-child(3) { border-left: 0; padding-left: 0; }
    .stApp .cifra:nth-child(n+3) { border-top: 1px solid var(--regla); }
    .stApp .metodologia { grid-template-columns: 1fr; }
    .stApp .cabecera h1 { font-size: 1.9rem; }
}
</style>
"""

COLOR_ESTADO = {"atrasado": "#c4741d", "a tiempo": "#5f9a78", "adelantado": "#5b4a9b"}
RESALTE = {"atrasado": "rgba(196,116,29,0.22)", "a tiempo": "rgba(95,154,120,0.24)",
           "adelantado": "rgba(91,74,155,0.18)"}


def html(texto):
    st.markdown(texto, unsafe_allow_html=True)


def negritas(texto):
    """Convierte **texto** de los mensajes a <strong> para el HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto)


def seccion(numero, nombre):
    html(f'<div class="seccion"><span class="num">{numero:02d}</span><span class="nombre">{nombre}</span></div>')


def _lista_puntos(puntos):
    nombres = [str(j) for j in puntos]
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def notas_faltantes(curva, k):
    """Explica los puntos sin registro hasta k, distinguiendo cómo los trata el modelo."""
    dif = curva[[f"dif_{j}" for j in range(1, k + 1)]].to_numpy(dtype=float)
    validos = np.flatnonzero(~np.isnan(dif))
    faltan = np.flatnonzero(np.isnan(dif))
    if not len(faltan):
        return []
    if not len(validos):
        return [f"Esta vuelta no tiene registros entre los puntos 1 y {k}; el modelo usa valores típicos."]
    internos = [i + 1 for i in faltan if validos[0] < i < validos[-1]]
    externos = [i + 1 for i in faltan if not validos[0] < i < validos[-1]]
    notas = []
    if internos:
        sujeto = (f"El punto {internos[0]} no tiene registro: el modelo lo completa" if len(internos) == 1
                  else f"Los puntos {_lista_puntos(internos)} no tienen registro: el modelo los completa")
        notas.append(f"{sujeto} uniendo los puntos vecinos (círculos huecos sobre la línea punteada).")
    if externos:
        if len(externos) == 1:
            notas.append(f"El punto {externos[0]} no tiene registro y no queda entre dos puntos registrados, así que "
                         "no se puede completar ni dibujar; para ese dato el modelo usa un valor típico del entrenamiento.")
        else:
            notas.append(f"Los puntos {_lista_puntos(externos)} no tienen registro y no quedan entre dos puntos "
                         "registrados, así que no se pueden completar ni dibujar; para esos datos el modelo usa "
                         "valores típicos del entrenamiento.")
    return notas


def controles(vueltas):
    """Barra lateral con la selección de la vuelta y del corte k."""
    with st.sidebar:
        html('<div class="control-titulo">Selección de la vuelta</div>')
        st.caption("Vueltas de la semana de prueba (26 ene – 1 feb 2025), que el modelo no vio al entrenar.")
        fechas = sorted(vueltas["fecha"].unique())
        st.session_state.setdefault("fecha", fechas[0])
        fecha = st.selectbox("Día", fechas, key="fecha", format_func=lambda f: nombre_fecha(pd.Timestamp(f)))
        del_dia = vueltas[vueltas["fecha"] == fecha]
        vehiculos = sorted(del_dia["vehiculo"].unique(), key=lambda v: (len(v), v))
        if st.session_state.get("vehiculo") not in vehiculos:
            st.session_state["vehiculo"] = vehiculos[0]
        vehiculo = st.selectbox("Bus", vehiculos, key="vehiculo")
        laps = sorted(del_dia.loc[del_dia["vehiculo"] == vehiculo, "lap"].astype(int))
        if st.session_state.get("lap") not in laps:
            st.session_state["lap"] = laps[0]
        lap = st.selectbox("Vuelta del día", laps, key="lap")
        st.button("Vuelta al azar", icon=":material/shuffle:", on_click=elegir_al_azar, args=(vueltas,), width="stretch")

        st.write("")
        html('<div class="control-titulo">Punto de corte</div>')
        k = st.select_slider("Punto de control observado (k): cuánto del recorrido ya se conoce",
                             options=list(p.CORTES), value=9)
    id_vuelta = del_dia.loc[(del_dia["vehiculo"] == vehiculo) & (del_dia["lap"] == lap), "id"].iloc[0]
    return id_vuelta, fecha, vehiculo, lap, k


def pestana_vuelta(paquetes, curvas, variables, id_vuelta, fecha, vehiculo, lap, k):
    paquete = paquetes[k]
    fila = variables[(variables["k"] == k) & (variables["id"] == id_vuelta)]
    curva = curvas[curvas["id"] == id_vuelta].iloc[0]
    pred = float(p.predecir(paquete, fila)[0])
    real = float(fila["y"].iloc[0])
    texto, estado = p.mensaje(pred, k)
    dia = DIAS[int(fila["dia_semana"].iloc[0])]
    franja = fila["franja"].iloc[0]
    mae = paquete["metricas_prueba"]["MAE"]
    hora = fila["hora_inicio"].iloc[0]

    seccion(1, "Resultado de la predicción")
    html(f'<p class="fecha-linea">Bus {vehiculo} · vuelta {lap} · {nombre_fecha(pd.Timestamp(fecha))} · '
         f'salida programada {int(hora):02d}:{round(hora % 1 * 60):02d} h</p>'
         f'<p class="titular" style="--resalte:{RESALTE[estado]}">{negritas(texto)}</p>')
    html('<div class="cifras">'
         f'<div class="cifra"><div class="rotulo">Predicción en el punto 18</div>'
         f'<div class="valor">{p.num(pred, 1)}<small>min</small></div></div>'
         f'<div class="cifra"><div class="rotulo">Estado previsto</div><div class="valor estado">'
         f'<i style="background:{COLOR_ESTADO[estado]}"></i>{estado.capitalize()}</div></div>'
         f'<div class="cifra" title="Estado real: {p.clasificar(real)}"><div class="rotulo">Valor real</div>'
         f'<div class="valor">{p.num(real, 0)}<small>min</small></div></div>'
         f'<div class="cifra"><div class="rotulo">Error de esta predicción</div>'
         f'<div class="valor">{p.num(abs(real - pred), 1)}<small>min</small></div></div>'
         '</div>')

    seccion(2, "Evolución del atraso en la vuelta")
    st.altair_chart(grafico_curva(curva, k, pred, mae), width="stretch", theme=None)
    html(f'<p class="pie">La barra naranja en el punto 18 marca ± {p.num(mae)} min, el error medio del modelo con '
         f'k = {k}. Pasa el cursor sobre los puntos para ver los valores.</p>')
    for nota in notas_faltantes(curva, k):
        html(f'<p class="pie">{nota}</p>')

    seccion(3, "Explicación de la predicción")
    efectos = p.explicar(paquete, fila)
    html(f'<p class="entradilla">El factor que más influye en esta predicción es '
         f'<strong>{efectos.index[0].lower()}</strong>: sin él, la predicción cambiaría en '
         f'<strong>{p.num(abs(efectos.iloc[0]))} min</strong>.</p>')
    st.altair_chart(grafico_explicacion(efectos), width="stretch", theme=None)
    html('<p class="pie">Cada barra compara la predicción con la que daría el modelo si ese grupo de datos se '
         'reemplazara por un valor típico. Los efectos no suman exactamente la predicción.</p>')

    seccion(4, "Confiabilidad del modelo")
    html(f'<p class="entradilla">{negritas(p.confianza(paquete))}</p>')
    for aviso in p.advertencias(paquete, fila, franja, dia):
        html(f'<div class="nota"><div class="rotulo">Atención</div><p>{negritas(aviso)}</p></div>')

    seccion(5, "Metodología del modelo")
    html('<div class="metodologia">'
         f'<p><b>Modelo.</b> {paquete["nombre_modelo"]}, el de menor error en validación en los cinco cortes; '
         'se entrenó un modelo por cada k con datos del 6 al 25 de enero de 2025.</p>'
         '<p><b>Variables.</b> El horario programado, el atraso observado en los puntos 1 a k, el vehículo, '
         'la vuelta anterior del mismo bus, el atraso de otros buses en los 30 min previos y el GPS.</p>'
         '<p><b>Sin fuga de información.</b> No se usa nada posterior al instante de predicción.</p>'
         '<p><b>Estados.</b> Atrasado si el atraso es de 5 min o más; adelantado si el adelanto es de 3 min o más.</p>'
         '</div>')


COLOR_MODELO = {"Gradient Boosting": "#5b4a9b", "Random Forest": "#2f8f6a", "Ridge": "#c4741d",
                "B1 mediana hora-día": "#6c6883", "B2 persistencia": "#a9a6b8"}
NOMBRE_CORTO = {"B1 mediana hora-día": "B1 · mediana hora y día", "B2 persistencia": "B2 · persistencia",
                "B0 a tiempo": "B0 · a tiempo", "B3 lineal atraso k": "B3 · lineal con atraso k"}


def _config(grafico, alto):
    return grafico.properties(height=alto).configure(font=FUENTE, background="transparent").configure_view(
        stroke=None).configure_axis(labelColor="#403b5c", titleColor="#403b5c", gridColor="#e4e2ec",
                                    domainColor="#c3bfd3", titleFontWeight=500, labelFontSize=12, titleFontSize=12)


def grafico_mae_k(tabla):
    datos = tabla.loc[list(COLOR_MODELO)].reset_index().melt(id_vars="modelo", var_name="k", value_name="MAE")
    datos["k"] = datos["k"].astype(int)
    datos["tipo"] = np.where(datos["modelo"].isin(r.LINEAS_BASE), "Línea base", "Modelo")
    datos["nombre"] = datos["modelo"].map(lambda m: NOMBRE_CORTO.get(m, m))
    dominio = list(COLOR_MODELO)
    color = alt.Color("modelo:N", legend=None, scale=alt.Scale(domain=dominio, range=list(COLOR_MODELO.values())))
    x = alt.X("k:Q", title="Punto de control observado (k)", scale=alt.Scale(domain=[3.4, 17.6], nice=False),
              axis=alt.Axis(values=list(p.CORTES), format="d", grid=False))
    y = alt.Y("MAE:Q", title="Error absoluto medio en prueba (min)", scale=alt.Scale(domain=[1.8, 5.2], nice=False),
              axis=alt.Axis(labelExpr="replace(format(datum.value, '.1f'), '.', ',')"))
    tooltip = [alt.Tooltip("nombre:N", title="Modelo"), alt.Tooltip("k:Q", title="k", format="d"),
               alt.Tooltip("MAE:Q", title="MAE (min)", format=".2f")]
    lineas = alt.Chart(datos).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color, detail="modelo:N",
        strokeDash=alt.condition(alt.datum.tipo == "Línea base", alt.value([5, 4]), alt.value([1, 0])))
    puntos = alt.Chart(datos).mark_point(filled=True, size=70, stroke="#f7f7f9", strokeWidth=2).encode(
        x=x, y=y, color=color, tooltip=tooltip)
    # Etiquetas al final de cada línea, separadas al menos 0,13 min para que no se monten.
    finales = datos[datos["k"] == 15].sort_values("MAE").copy()
    posiciones = []
    for v in finales["MAE"]:
        posiciones.append(max(v, posiciones[-1] + 0.13) if posiciones else v)
    finales["y_etiqueta"] = posiciones
    etiquetas = alt.Chart(finales).mark_text(align="left", dx=10, fontSize=12).encode(
        x=x, y=alt.Y("y_etiqueta:Q"), text="nombre:N", color=color)
    return _config(alt.layer(lineas, puntos, etiquetas), 380)


def grafico_comparacion(ic, k):
    datos = ic[ic["k"] == k].copy()
    datos["nombre"] = datos["modelo"].map(lambda m: NOMBRE_CORTO.get(m, m))
    datos["grupo"] = np.select([datos["modelo"] == "Gradient Boosting", datos["modelo"].isin(r.LINEAS_BASE)],
                               ["Mejor modelo", "Línea base"], "Otros modelos")
    datos = datos.sort_values("MAE")
    orden = list(datos["nombre"])
    escala = alt.Scale(domain=["Mejor modelo", "Otros modelos", "Línea base"], range=["#5b4a9b", "#403b5c", "#a9a6b8"])
    color = alt.Color("grupo:N", title=None, legend=alt.Legend(orient="top", labelFontSize=12), scale=escala)
    y = alt.Y("nombre:N", title=None, sort=orden, axis=alt.Axis(ticks=False, domain=False, labelPadding=10))
    x = alt.X("MAE:Q", title="Error absoluto medio en prueba (min), con IC 95 % por bootstrap de días",
              axis=alt.Axis(labelExpr="replace(format(datum.value, '.1f'), '.', ',')"))
    tooltip = [alt.Tooltip("nombre:N", title="Modelo"), alt.Tooltip("MAE:Q", title="MAE (min)", format=".2f"),
               alt.Tooltip("IC_inf:Q", title="IC 95 % desde", format=".2f"),
               alt.Tooltip("IC_sup:Q", title="IC 95 % hasta", format=".2f")]
    rango = alt.Chart(datos).mark_rule(strokeWidth=2, opacity=0.45).encode(
        y=y, x="IC_inf:Q", x2="IC_sup:Q", tooltip=tooltip,
        color=alt.Color("grupo:N", legend=None, scale=escala))
    punto = alt.Chart(datos).mark_point(filled=True, opacity=1, stroke="#f7f7f9", strokeWidth=2).encode(
        y=y, x=x, color=color, tooltip=tooltip,
        size=alt.condition(alt.datum.grupo == "Mejor modelo", alt.value(150), alt.value(80)))
    datos["etiqueta"] = datos["MAE"].map(lambda v: p.num(v, 2))
    texto = alt.Chart(datos).mark_text(align="left", dx=8, dy=-9, fontSize=11, font="IBM Plex Mono",
                                       color="#1c1838").encode(y=y, x="MAE:Q", text="etiqueta:N")
    return _config(alt.layer(rango, punto, texto).resolve_legend(color="independent"), 30 * len(datos))


def grafico_importancia(tabla):
    datos = tabla.copy()
    datos["etiqueta"] = datos["aumento_MAE"].map(lambda v: f"+{p.num(v, 2)} min")
    base = alt.Chart(datos).encode(
        y=alt.Y("grupo:N", title=None, sort="-x", axis=alt.Axis(ticks=False, domain=False, labelPadding=10,
                                                                labelFontSize=13)),
        x=alt.X("aumento_MAE:Q", title="Cuánto sube el error al desordenar el grupo (min)",
                scale=alt.Scale(domain=[0, datos["aumento_MAE"].max() * 1.3]),
                axis=alt.Axis(tickCount=5, labelExpr="replace(format(datum.value, '.1f'), '.', ',')")))
    barras = base.mark_bar(height=22, cornerRadiusEnd=2, color="#5b4a9b").encode(
        tooltip=[alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("aumento_MAE:Q", title="Aumento (min)", format=".2f")])
    texto = base.mark_text(align="left", dx=8, fontSize=13, font="IBM Plex Mono", color="#1c1838").encode(
        text="etiqueta:N")
    return _config(alt.layer(barras, texto), 64 * len(datos))


def grafico_confusion(matriz):
    filas = matriz.div(matriz.sum(axis=1), axis=0)
    datos = matriz.reset_index(names="real").melt(id_vars="real", var_name="predicho", value_name="vueltas")
    datos["pct"] = [filas.loc[r, c] for r, c in zip(datos["real"], datos["predicho"])]
    datos["etiqueta"] = [f"{n}\n{p.num(100 * q, 0)} %" for n, q in zip(datos["vueltas"], datos["pct"])]
    orden = ["adelantado", "a tiempo", "atrasado"]
    base = alt.Chart(datos).encode(
        x=alt.X("predicho:N", title="Clase predicha", sort=orden, axis=alt.Axis(labelAngle=0, orient="top",
                                                                               ticks=False, domain=False)),
        y=alt.Y("real:N", title="Clase real", sort=orden, axis=alt.Axis(ticks=False, domain=False)))
    celdas = base.mark_rect(stroke="#f7f7f9", strokeWidth=3).encode(
        color=alt.Color("pct:Q", legend=None, scale=alt.Scale(domain=[0, 1], range=["#f1eff7", "#3b2f73"])),
        tooltip=[alt.Tooltip("real:N", title="Real"), alt.Tooltip("predicho:N", title="Predicha"),
                 alt.Tooltip("vueltas:Q", title="Vueltas"), alt.Tooltip("pct:Q", title="% de la fila", format=".0%")])
    texto = base.mark_text(fontSize=14, font="IBM Plex Mono", lineBreak="\n").encode(
        text="etiqueta:N", color=alt.condition(alt.datum.pct > 0.5, alt.value("#ffffff"), alt.value("#1c1838")))
    return _config(alt.layer(celdas, texto), 300)


def tabla_html(tabla, decimales, umbral=None, nota_fila=None):
    encabezado = "".join(f"<th>k = {k}</th>" for k in tabla.columns)
    filas = []
    for nombre, fila in tabla.iterrows():
        celdas = "".join(
            f'<td class="ok">{p.num(v, decimales)}</td>' if umbral is not None and v >= umbral
            else f"<td>{p.num(v, decimales)}</td>" for v in fila)
        filas.append(f'<tr><th class="fila">{nota_fila(nombre) if nota_fila else nombre}</th>{celdas}</tr>')
    return f'<table class="tabla"><thead><tr><th></th>{encabezado}</tr></thead><tbody>{"".join(filas)}</tbody></table>'


def pestana_resultados():
    mejor = r.mejor_por_k()
    mae = r.mae_por_k()
    m9 = mejor.loc[9]
    mejora = 100 * (1 - m9["MAE"] / m9["MAE_B1"])

    seccion(1, "Resumen de resultados")
    html('<p class="fecha-linea">Semana de prueba · 26 ene – 1 feb 2025 · 1.061 vueltas</p>'
         f'<p class="titular" style="--resalte:rgba(91,74,155,0.18)">A mitad de la vuelta (k = 9), '
         f'<strong>{m9["modelo"]}</strong> predice la desviación en el último punto de control con un error medio de '
         f'<strong>{p.num(m9["MAE"], 2)} min</strong>, un {p.num(mejora, 0)} % menos que la mejor línea base.</p>')
    html('<div class="cifras">'
         f'<div class="cifra"><div class="rotulo">Mejor modelo en los cinco cortes</div>'
         f'<div class="valor estado">{m9["modelo"]}</div></div>'
         f'<div class="cifra"><div class="rotulo">Error medio (MAE) en k = 9</div>'
         f'<div class="valor">{p.num(m9["MAE"], 2)}<small>min</small></div></div>'
         f'<div class="cifra"><div class="rotulo">Vueltas con error ≤ 5 min</div>'
         f'<div class="valor">{p.num(m9["pct_err_5"], 1)}<small>%</small></div></div>'
         f'<div class="cifra"><div class="rotulo">Error medio en k = 15</div>'
         f'<div class="valor">{p.num(mejor.loc[15, "MAE"], 2)}<small>min</small></div></div>'
         '</div>')
    html(f'<p class="pie">R² en k = 9: {p.num(m9["R2"], 2)}. Mejor línea base (mediana por hora y día): '
         f'{p.num(m9["MAE_B1"], 2)} min. Hiperparámetros elegidos con validación temporal de ventana creciente; '
         'la semana de prueba se evaluó una sola vez.</p>')

    seccion(2, "El error según el punto de corte")
    html('<p class="entradilla">Observar más puntos de la vuelta reduce el error solo cuando quedan pocos tramos: '
         'entre k = 4 y 9 el error se mantiene casi igual y baja con claridad en k = 12 y 15.</p>')
    st.altair_chart(grafico_mae_k(mae), width="stretch", theme=None)
    html('<p class="pie">Líneas continuas: modelos; líneas discontinuas: líneas base. B1 usa solo la hora de inicio y '
         'el día; B2 supone que el atraso final es igual al último atraso observado.</p>')

    seccion(3, "Comparación de todos los modelos")
    k = st.select_slider("Punto de corte (k) para comparar", options=list(p.CORTES), value=9, key="k_resultados")
    html(f'<p class="entradilla">Doce regresores y cuatro líneas base ordenados por su error en prueba con k = {k}. '
         'La línea fina es el intervalo de confianza del 95 %, estimado remuestreando días completos.</p>')
    st.altair_chart(grafico_comparacion(r.intervalos(), k), width="stretch", theme=None)
    html('<p class="pie">Los intervalos son anchos porque la prueba tiene solo 7 días. Gradient Boosting supera a las '
         'líneas base y a Ridge en todos los cortes (Wilcoxon, p < 0,001); frente a Random Forest la diferencia no es '
         'significativa por día.</p>')

    seccion(4, "Qué explica las predicciones")
    html('<p class="entradilla">El horario (hora de inicio y día de la semana) es el grupo de variables que más pesa; '
         'el GPS casi no aporta.</p>')
    st.altair_chart(grafico_importancia(r.importancia_grupos()), width="stretch", theme=None)
    html('<p class="pie">Importancia por permutación en prueba con k = 9: cuánto sube el error si se desordenan al azar '
         'los valores de cada grupo.</p>')
    html('<p class="subtitulo">Ablación: error al agregar cada grupo de variables (MAE en prueba, min)</p>')
    html(tabla_html(r.ablacion(), 2))
    html('<p class="pie">Gradient Boosting con los mismos hiperparámetros. El atraso observado reduce el error de forma '
         'apreciable solo desde k = 12; el vehículo y la flota restan entre 0,04 y 0,07 min.</p>')

    seccion(5, "Clasificación: atrasado, a tiempo o adelantado")
    c9 = r.clasificacion_k9().loc["Gradient Boosting"]
    html('<p class="entradilla">El mismo problema planteado como tres estados (atrasado ≥ 5 min, adelantado ≥ 3 min de '
         'adelanto). Gradient Boosting es también el mejor clasificador en k = 9.</p>')
    html('<div class="cifras">'
         f'<div class="cifra"><div class="rotulo">Exactitud</div><div class="valor">{p.num(c9["exactitud"], 3)}</div></div>'
         f'<div class="cifra"><div class="rotulo">F1 macro</div><div class="valor">{p.num(c9["F1_macro"], 3)}</div></div>'
         f'<div class="cifra"><div class="rotulo">F1 de la clase atrasado</div>'
         f'<div class="valor">{p.num(c9["F1_atrasado"], 3)}</div></div>'
         f'<div class="cifra"><div class="rotulo">AUC-PR de atrasado</div>'
         f'<div class="valor">{p.num(c9["AUC_PR_atrasado"], 3)}</div></div>'
         '</div>')
    st.write("")
    st.altair_chart(grafico_confusion(r.matriz_confusion()), width="stretch", theme=None)
    html('<p class="pie">Matriz de confusión en prueba (k = 9): número de vueltas y porcentaje de cada clase real. '
         'La mayor confusión ocurre entre «a tiempo» y «atrasado». Discretizar la predicción en minutos da casi el '
         'mismo resultado (exactitud 0,796).</p>')

    seccion(6, "Métricas frente a la meta de 0,80")
    html('<p class="entradilla">Las métricas en escala 0–1 del mejor modelo de cada enfoque. En negrita, las que '
         'superan la meta de 0,80 planteada por los docentes.</p>')
    html(tabla_html(r.meta_80(), 3, umbral=0.80))
    html('<p class="pie">A mitad de la vuelta la superan el porcentaje de vueltas con error ≤ 5 min y el AUC-PR; '
         'con k = 15 la superan casi todas. El límite lo pone la información disponible, no el preprocesamiento.</p>')

    seccion(7, "Hipótesis")
    items = "".join(
        f'<div class="hipotesis"><div class="codigo">{codigo}</div><div><p class="enunciado">{enunciado}</p>'
        f'<p class="veredicto {clase}">{veredicto}</p><p class="evidencia">{evidencia}</p></div></div>'
        for codigo, enunciado, veredicto, clase, evidencia in r.HIPOTESIS)
    html(f'<div class="lista-hipotesis">{items}</div>')


@st.cache_data(show_spinner=False)
def procesar_archivo(contenido, k):
    vueltas = p.leer_archivo_times(io.BytesIO(contenido))
    return p.predecir_vueltas(vueltas, modelos()[k]), p.dias_de_entrenamiento(vueltas)


def _hora(minutos):
    return f"{int(minutos // 60):02d}:{int(round(minutos % 60)):02d}"


def tabla_predicciones(x, k):
    t = x.sort_values(["fecha", "hora_inicio", "vehiculo"])
    return pd.DataFrame({
        "Fecha": t["fecha"].map(nombre_fecha),
        "Bus": t["vehiculo"],
        "Vuelta": t["lap"].astype(int),
        "Salida": (t["hora_inicio"] * 60).map(_hora),
        f"Atraso en k={k}": t["atr_ultimo"],
        "Predicción": t["pred"],
        "Estado": t["estado"].str.capitalize(),
        "Real": t["y"],
        "Error": t["error"],
    })


def grafico_pred_real(x):
    datos = x[x["y"].notna()][["pred", "y", "vehiculo", "lap", "fecha", "error"]].copy()
    datos["fecha"] = datos["fecha"].map(nombre_fecha)
    lim = [np.floor(min(datos["pred"].min(), datos["y"].min())) - 2, np.ceil(max(datos["pred"].max(), datos["y"].max())) + 2]
    lim = [max(lim[0], -25), min(lim[1], 45)]
    eje = dict(scale=alt.Scale(domain=lim, nice=False), axis=alt.Axis(labelExpr="format(datum.value, 'd')"))
    diagonal = alt.Chart(pd.DataFrame({"a": lim, "b": lim})).mark_line(color="#6c6883", strokeDash=[4, 4],
                                                                     strokeWidth=1.5, clip=True).encode(x="a:Q", y="b:Q")
    banda = alt.Chart(pd.DataFrame({"a": lim, "lo": [lim[0] - 5, lim[1] - 5], "hi": [lim[0] + 5, lim[1] + 5]})
                      ).mark_area(color="#ecebf4", opacity=0.9, clip=True).encode(x="a:Q", y="lo:Q", y2="hi:Q")
    puntos = alt.Chart(datos).mark_circle(size=34, opacity=0.55, color="#5b4a9b", clip=True).encode(
        x=alt.X("pred:Q", title="Predicción en el punto 18 (min)", **eje),
        y=alt.Y("y:Q", title="Valor real en el punto 18 (min)", **eje),
        tooltip=[alt.Tooltip("fecha:N", title="Fecha"), alt.Tooltip("vehiculo:N", title="Bus"),
                 alt.Tooltip("lap:Q", title="Vuelta"), alt.Tooltip("pred:Q", title="Predicción (min)", format=".1f"),
                 alt.Tooltip("y:Q", title="Real (min)", format=".0f"),
                 alt.Tooltip("error:Q", title="Error (min)", format=".1f")])
    nota = alt.Chart(pd.DataFrame({"x": [lim[0] + 1], "y": [lim[1] - 1.5], "t": ["franja: error de 5 min o menos"]})
                     ).mark_text(align="left", fontSize=11, color="#6c6883").encode(x="x:Q", y="y:Q", text="t:N")
    return _config(alt.layer(banda, diagonal, puntos, nota).properties(width=520), 460)


def usar_archivo(nombre, contenido):
    """Valida el archivo al confirmarlo; si es válido pasa a ser el archivo en uso (uno a la vez)."""
    st.session_state["n_cargador"] += 1  # vacía el cargador: el archivo pasa a mostrarse solo en la franja
    try:
        vueltas = p.leer_archivo_times(io.BytesIO(contenido))
    except p.ArchivoInvalido as e:
        st.session_state["error_archivo"] = f"<code>{nombre}</code>: {e}"
        return
    st.session_state.pop("error_archivo", None)
    st.session_state["archivo_activo"] = {"nombre": nombre, "contenido": contenido, "vueltas": len(vueltas),
                                          "dias": vueltas["fecha"].nunique()}


def quitar_archivo():
    st.session_state.pop("archivo_activo", None)
    st.session_state.pop("error_archivo", None)
    st.session_state["n_cargador"] += 1  # cambia la clave del cargador para vaciarlo


def pestana_archivo():
    seccion(1, "Archivo de entrada")
    html('<p class="entradilla">Sube un archivo CSV con el mismo formato de <em>Times.csv</em> (una fila por vuelta, '
         'con las listas de los 18 puntos de control) y el modelo predecirá la desviación en el punto 18 de cada '
         'vuelta.</p>')
    html('<p class="pie">Columnas obligatorias: <code>id</code>, <code>date</code>, <code>vehicle_id_id</code>, '
         '<code>lap</code>, <code>start_time</code>, <code>end_time</code> y <code>difference_time</code>. '
         'Las vueltas que aún no llegan al punto 18 también se predicen. Conviene incluir todas las vueltas del día: '
         'el modelo usa el atraso de los otros buses como contexto.</p>')
    ruta_ejemplo = p.DIR_DEMO / "datos" / "ejemplo_times.csv"
    st.session_state.setdefault("n_cargador", 0)
    activo = st.session_state.get("archivo_activo")

    # 1) Archivo en uso, con opción de quitarlo
    if activo:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(
            f'<div class="en-uso"><div class="rotulo">Archivo en uso</div><div class="nombre">{activo["nombre"]}</div>'
            f'<div class="detalle">{activo["vueltas"]:,} vueltas · {activo["dias"]} días</div></div>'.replace(
                f'{activo["vueltas"]:,}', f'{activo["vueltas"]:,}'.replace(",", ".")),
            unsafe_allow_html=True)
        c2.button("Quitar", icon=":material/close:", on_click=quitar_archivo, width="stretch")
        st.write("")

    # 2) Carga de un archivo (uno a la vez) y confirmación explícita
    subido = st.file_uploader("Subir otro archivo CSV (formato Times.csv)" if activo else
                              "Archivo CSV con formato Times.csv", type=["csv"],
                              key=f"archivo_times_{st.session_state['n_cargador']}")
    pendiente = subido is not None and (not activo or activo["contenido"] != subido.getvalue())
    if pendiente:
        html(f'<div class="nota"><div class="rotulo">Archivo cargado · pendiente de confirmar</div>'
             f'<p><code>{subido.name}</code>. Pulsa «Usar archivo» para procesarlo'
             f'{" y reemplazar el actual" if activo else ""}.</p></div>')
        st.button("Usar archivo", icon=":material/check:", type="primary", on_click=usar_archivo,
                  args=(subido.name, subido.getvalue()), width="content")
    if st.session_state.get("error_archivo"):
        html(f'<div class="nota"><div class="rotulo">No se pudo usar el archivo</div>'
             f'<p>{st.session_state["error_archivo"]}</p></div>')

    # 3) Archivo de ejemplo
    if ruta_ejemplo.exists():
        c1, c2 = st.columns(2, vertical_alignment="center")
        c1.button("Usar archivo de ejemplo", icon=":material/description:", width="stretch",
                  on_click=usar_archivo, args=("ejemplo_times.csv (semana de prueba)", ruta_ejemplo.read_bytes()))
        c2.download_button("Descargar archivo de ejemplo", ruta_ejemplo.read_bytes(), file_name="ejemplo_times.csv",
                           mime="text/csv", icon=":material/download:", width="stretch")

    seccion(2, "Punto de corte")
    k = st.select_slider("Hasta qué punto de control se conoce cada vuelta (k)", options=list(p.CORTES), value=9,
                         key="k_archivo")

    if not activo:
        html('<p class="pie">Sube un archivo y confírmalo, o usa el archivo de ejemplo, para ver las predicciones.</p>')
        return
    contenido, origen = activo["contenido"], activo["nombre"]
    with st.spinner("Calculando predicciones…"):
        x, dias_vistos = procesar_archivo(contenido, k)

    con_real = x["y"].notna()
    conteo = x["estado"].value_counts()
    seccion(3, "Resumen")
    html(f'<p class="fecha-linea">{origen} · {len(x)} vueltas · {x["fecha"].nunique()} días · k = {k}</p>')
    html('<div class="cifras">' + "".join(
        f'<div class="cifra"><div class="rotulo">{rotulo}</div><div class="valor">{valor}</div></div>'
        for rotulo, valor in [
            ("Vueltas procesadas", f"{len(x):,}".replace(",", ".")),
            ("Atrasadas previstas", f'{conteo.get("atrasado", 0)}<small>{p.num(100 * conteo.get("atrasado", 0) / len(x), 0)} %</small>'),
            ("A tiempo previstas", f'{conteo.get("a tiempo", 0)}<small>{p.num(100 * conteo.get("a tiempo", 0) / len(x), 0)} %</small>'),
            ("Adelantadas previstas", f'{conteo.get("adelantado", 0)}<small>{p.num(100 * conteo.get("adelantado", 0) / len(x), 0)} %</small>'),
        ]) + "</div>")
    if con_real.any():
        mae = x.loc[con_real, "error"].mean()
        dentro = 100 * (x.loc[con_real, "error"] <= 5).mean()
        html(f'<p class="entradilla" style="margin-top:22px">En las <strong>{con_real.sum()}</strong> vueltas que tienen '
             f'valor real en el punto 18, el modelo se equivocó en promedio <strong>{p.num(mae, 2)} min</strong>, y en el '
             f'<strong>{p.num(dentro, 0)} %</strong> de ellas el error fue de 5 min o menos.</p>')
    if (~con_real).any():
        html(f'<p class="pie">{(~con_real).sum()} vueltas no tienen un valor real válido en el punto 18 (en curso, sin '
             'registro o con más de 60 min de desviación): se predicen, pero no entran en el cálculo del error.</p>')
    if dias_vistos:
        html('<div class="nota"><div class="rotulo">Atención</div><p>El archivo incluye días que el modelo usó para '
             f'entrenar ({", ".join(nombre_fecha(d) for d in dias_vistos)}). En esas vueltas el error será menor que '
             'con datos realmente nuevos.</p></div>')
    html('<p class="pie">El archivo no trae GPS, así que esas variables se completan con valores típicos; en el informe '
         'el GPS no mejoraba el error (H4).</p>')

    seccion(4, "Predicciones por vuelta")
    tabla = tabla_predicciones(x, k)
    html('<p class="pie">Minutos respecto al horario en el punto 18 (− adelanto, + atraso). «Atraso en k» es el último '
         'atraso registrado hasta el punto de corte; «—» indica que no hay dato. Haz clic en una columna para '
         'ordenar.</p>')
    vista = tabla.copy()
    for c in (f"Atraso en k={k}", "Real", "Error"):
        decimales = 1 if c == "Error" else 0
        vista[c] = vista[c].map(lambda v: "—" if pd.isna(v) else p.num(v, decimales))
    st.dataframe(vista.style.format(decimal=",", precision=1, subset=["Predicción"]), hide_index=True,
                 width="stretch", height=420,
                 column_config={c: st.column_config.TextColumn(c) for c in (f"Atraso en k={k}", "Real", "Error")})
    st.download_button("Descargar predicciones (CSV)", tabla.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"predicciones_k{k}.csv", mime="text/csv", icon=":material/download:")

    if con_real.any():
        seccion(5, "Predicción frente al valor real")
        html('<p class="entradilla">Cada punto es una vuelta. Si la predicción fuera perfecta, todos los puntos caerían '
             'sobre la diagonal; la franja marca un error de 5 min o menos.</p>')
        st.altair_chart(grafico_pred_real(x), theme=None)


def en_construccion(texto):
    html(f'<div class="nota" style="border-color:#5b4a9b"><div class="rotulo" style="color:#5b4a9b">En construcción'
         f'</div><p>{texto}</p></div>')


def main():
    html(ESTILOS)
    html('<div class="cabecera">'
         '<p class="antetitulo">Aprendizaje de Máquina (1INF02) · PUCP 2026-2 · Grupo 5</p>'
         '<h1>Predicción del adelanto y atraso de buses urbanos en su último punto de control</h1>'
         '<p class="bajada">Sistema de transporte público de Cusco</p>'
         '</div>')
    try:
        paquetes = modelos()
        curvas, variables = semana()
    except FileNotFoundError:
        st.error("No se encontraron los modelos. Ejecuta primero `python entrenar_modelos.py` desde la carpeta demo/.")
        st.stop()

    vueltas = variables[variables["k"] == p.CORTES[0]][["id", "fecha", "vehiculo", "lap"]]
    seleccion = controles(vueltas)

    t1, t2, t3 = st.tabs(["Predecir una vuelta", "Probar con otros datos", "Resultados del proyecto"])
    with t1:
        pestana_vuelta(paquetes, curvas, variables, *seleccion)
    with t2:
        pestana_archivo()
    with t3:
        pestana_resultados()


main()
