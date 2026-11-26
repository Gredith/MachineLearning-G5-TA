"""Demo interactiva: predicción del adelanto o atraso de un bus en su último punto de control.

Desde la carpeta demo/ (antes: python demo/entrenar_modelos.py desde la raíz):
    python -m streamlit run app.py
"""
import html as html_lib
import io
import json
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import prediccion as p
import resultados as r
from textos import HIPOTESIS, IDIOMAS, VEREDICTO, Traductor

st.set_page_config(page_title="Bus delay prediction" if st.query_params.get("idioma", "").lower() == "en"
                   else "Predicción de atraso de buses", page_icon=":material/directions_bus:", layout="wide",
                   initial_sidebar_state="expanded")

FUENTE = "Atkinson Hyperlegible Next"
TEMAS = {"sistema": ("System", ":material/brightness_auto:"), "claro": ("Light", ":material/light_mode:"),
         "oscuro": ("Dark", ":material/dark_mode:")}

# Paleta índigo y ámbar: índigo para lo observado y ámbar para la predicción, en versión clara y oscura.
PALETAS = {
    "light": dict(papel="#f7f7f9", papel2="#ebe9f3", tinta="#1c1838", tinta2="#403b5c", tinta3="#6c6883",
                  regla="#d6d3e3", acento="#5b4a9b", acento_claro="#b3aad6", naranja="#c4741d", naranja_texto="#8a4f10",
                  verde="#5f9a78", verde_texto="#2f7a57", gris="#8a8984", gris_claro="#a9a6b8", zona="#ecebf4",
                  grid="#e4e2ec", dominio="#c3bfd3", celda_min="#f1eff7", celda_max="#3b2f73", texto_celda="#ffffff",
                  rf="#2f8f6a"),
    "dark": dict(papel="#14121c", papel2="#211e2e", tinta="#ecebf4", tinta2="#c9c5dc", tinta3="#9792ad",
                 regla="#383449", acento="#a99be0", acento_claro="#6c5fa8", naranja="#e39a4c", naranja_texto="#efb57a",
                 verde="#6fbf94", verde_texto="#86d3ab", gris="#8f8c99", gris_claro="#6f6a85", zona="#262236",
                 grid="#2c2940", dominio="#4a4560", celda_min="#221f31", celda_max="#a99be0", texto_celda="#14121c",
                 rf="#5fbf91"),
}
PAL = PALETAS["light"]
t = Traductor("es")


# ----------------------------------------------------------------------------------------------------------------
# Datos en caché
# ----------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def modelos():
    return p.cargar_modelos()


@st.cache_data(show_spinner=False)
def semana():
    return p.cargar_semana_prueba()


@st.cache_data(show_spinner=False)
def procesar_archivo(contenido, k):
    vueltas = p.leer_archivo_times(io.BytesIO(contenido))
    return p.predecir_vueltas(vueltas, modelos()[k]), p.dias_de_entrenamiento(vueltas)


# ----------------------------------------------------------------------------------------------------------------
# Utilidades de presentación
# ----------------------------------------------------------------------------------------------------------------
def html(texto):
    st.markdown(texto, unsafe_allow_html=True)


def en_linea(texto):
    """Convierte **negrita** y `código` de los textos a HTML."""
    texto = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto)
    texto = re.sub(r"\*(.+?)\*", r"<em>\1</em>", texto)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", texto)


def ayuda(texto):
    return f'<span class="ayuda" title="{html_lib.escape(texto, quote=True)}">?</span>'


def seccion(numero, clave):
    html(f'<div class="seccion"><span class="num">{numero:02d}</span><span class="nombre">{t(clave)}</span></div>')


def cifras(items, vertical=False):
    """Cifras en fila (o en columna): lista de (rótulo, valor en HTML, ayuda o None[, detalle])."""
    celdas = []
    for rotulo, valor, nota, *detalle in items:
        extra = f'<div class="detalle">{detalle[0]}</div>' if detalle and detalle[0] else ""
        celdas.append(f'<div class="cifra"><div class="rotulo">{rotulo}{ayuda(nota) if nota else ""}</div>'
                      f'<div class="valor">{valor}</div>{extra}</div>')
    html(f'<div class="cifras{" vertical" if vertical else ""}">{"".join(celdas)}</div>')


def nota(rotulo, texto):
    html(f'<div class="nota"><div class="rotulo">{rotulo}</div><p>{en_linea(texto)}</p></div>')


def pie(texto):
    html(f'<p class="pie">{en_linea(texto)}</p>')


def minutos(v, decimales=1):
    return f'{t.num(v, decimales)}<small>min</small>'


def _config(grafico, alto):
    return grafico.properties(height=alto).configure(font=FUENTE, background="transparent").configure_view(
        stroke=None).configure_axis(labelColor=PAL["tinta2"], titleColor=PAL["tinta2"], gridColor=PAL["grid"],
                                    domainColor=PAL["dominio"], tickColor=PAL["dominio"], titleFontWeight=500,
                                    labelFontSize=12, titleFontSize=12, labelLimit=300).configure_legend(
        labelColor=PAL["tinta"], labelFontSize=12, labelLimit=320)


def mostrar(grafico, **kwargs):
    st.altair_chart(grafico, theme=None, **({"width": "stretch"} | kwargs))


# ----------------------------------------------------------------------------------------------------------------
# Idioma y tema
# ----------------------------------------------------------------------------------------------------------------
PESTANAS = ("pestana_vuelta", "pestana_archivo", "pestana_resultados", "pestana_acerca")


def _mantener_pestana():
    """Al cambiar de idioma cambian los nombres de las pestañas: se conserva la que estaba abierta."""
    anterior = Traductor(st.session_state.get("_idioma_anterior", "es"))
    nuevo = Traductor(st.session_state["idioma"])
    nombres = [anterior(c) for c in PESTANAS]
    if st.session_state.get("pestana") in nombres:
        st.session_state["pestana"] = nuevo(PESTANAS[nombres.index(st.session_state["pestana"])])
    st.session_state["_idioma_anterior"] = st.session_state["idioma"]


def preferencias():
    """Idioma y tema, fijos en la esquina superior izquierda y guardados en la URL (sobreviven a la recarga del tema)."""
    with st.container(key="config", horizontal=True, gap="small", width="content"):
        idioma = st.segmented_control("Idioma · Language", list(IDIOMAS), format_func=IDIOMAS.get, key="idioma",
                                      default="es", required=True, bind="query-params", on_change=_mantener_pestana,
                                      label_visibility="collapsed")
        # El tema se guarda a mano en la URL: enlazarlo al widget escribiría el ícono en vez del código.
        if "tema" not in st.session_state:
            st.session_state["tema"] = st.query_params.get("tema") if st.query_params.get("tema") in TEMAS else "sistema"
        tema = st.segmented_control("Tema · Theme", list(TEMAS), format_func=lambda m: TEMAS[m][1], key="tema",
                                    required=True, label_visibility="collapsed")
    st.query_params["tema"] = tema
    st.session_state.setdefault("_idioma_anterior", idioma)

    # Streamlit guarda el tema elegido en el navegador; si no coincide con el pedido, se actualiza y se recarga.
    deseado = json.dumps(json.dumps(TEMAS[tema][0]))
    components.html(f"""<script>
        const clave = 'stActiveTheme-' + window.parent.location.pathname + '-v2';
        const ls = window.parent.localStorage;
        if ((ls.getItem(clave) || '"System"') !== {deseado}) {{ ls.setItem(clave, {deseado}); window.parent.location.reload(); }}
        </script>""", height=0)
    return Traductor(idioma), PALETAS["dark" if st.context.theme.type == "dark" else "light"]


def estilos():
    variables = "; ".join(f"--{k.replace('_', '-')}: {v}" for k, v in PAL.items() if isinstance(v, str))
    html(f"<style>:root {{ {variables}; }}</style>" + ESTILOS)


# Atkinson Hyperlegible Next: tipografía diseñada para máxima legibilidad (letras y números bien diferenciados).
ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;500;600;700&family=Atkinson+Hyperlegible+Mono:wght@400;500;700&display=swap');
:root { --fuente: 'Atkinson Hyperlegible Next', system-ui, sans-serif; --mono: 'Atkinson Hyperlegible Mono', ui-monospace, monospace; }
.stApp { background: var(--papel); color: var(--tinta); font-family: var(--fuente); }
.stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp [data-baseweb] { font-family: var(--fuente); }
.block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: var(--papel); }
[data-testid="stMainMenu"], #MainMenu, [data-testid="stStatusWidget"] { display: none !important; }
div[data-testid="stElementContainer"]:has(iframe[height="0"]) { display: none; }

/* Configuración global: esquina superior izquierda del área principal, sobre la cabecera de Streamlit. Se ancla
   al mismo contenedor que la cabecera (no a la ventana) para quedar siempre a la derecha de la barra lateral, y
   deja libre la esquina derecha, donde Streamlit Community Cloud pone sus botones (compartir, GitHub, menú). */
.st-key-config { position: absolute; top: 0.7rem; left: 1.2rem !important; right: auto; width: auto !important;
                 z-index: 999991; }
/* Con la barra lateral cerrada, Streamlit pone a la izquierda el botón para abrirla: se deja su espacio. */
.stApp:has([data-testid="stExpandSidebarButton"]) .st-key-config { left: 3.8rem !important; }
.st-key-config [data-testid="stButtonGroup"] button { min-height: 32px; padding: 0 10px; }

section[data-testid="stSidebar"] { background: var(--papel2); border-right: 1px solid var(--regla); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.stApp .control-titulo { font: 700 0.78rem var(--fuente); letter-spacing: 0.08em; text-transform: uppercase;
                         color: var(--tinta); border-bottom: 1px solid var(--tinta); padding-bottom: 6px; margin-bottom: 8px; }

.stApp .cabecera { border-bottom: 2px solid var(--tinta); padding: 0 0 12px 0; margin-bottom: 4px; }
.stApp .antetitulo { font: 600 0.78rem var(--fuente); letter-spacing: 0.06em; text-transform: uppercase;
                     color: var(--acento); margin: 0 0 6px 0; }
.stApp .cabecera h1 { font: 700 1.7rem/1.2 var(--fuente); letter-spacing: -0.01em; color: var(--tinta); margin: 0; padding: 0; }

.stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid var(--regla); }
.stTabs [data-baseweb="tab"] { padding: 8px 0; }
.stTabs [data-baseweb="tab"] p { font: 600 0.92rem var(--fuente); }

.stApp .seccion { border-top: 1px solid var(--regla); margin: 26px 0 10px 0; padding-top: 10px;
                  display: flex; align-items: baseline; gap: 12px; }
.stApp .seccion .num { font: 700 0.9rem var(--mono); color: var(--acento); }
.stApp .seccion .nombre { font: 700 1rem var(--fuente); letter-spacing: 0.04em; text-transform: uppercase; color: var(--tinta); }
.stApp .subtitulo { font: 700 0.95rem var(--fuente); color: var(--tinta); margin: 4px 0 2px 0; }
.stApp .entradilla { font: 400 1.08rem/1.5 var(--fuente); color: var(--tinta); margin: 2px 0 10px 0; }
.stApp .fecha-linea { font: 500 0.88rem var(--fuente); color: var(--tinta3); margin: 0 0 4px 0; }
.stApp .titular { font: 400 1.5rem/1.3 var(--fuente); color: var(--tinta); margin: 12px 0 8px 0; }
.stApp .titular strong { font-weight: 700; box-shadow: inset 0 -0.32em 0 var(--resalte); }

.stApp .cifras { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--regla);
                 border-bottom: 1px solid var(--regla); margin-top: 6px; }
.stApp .cifras.vertical { grid-template-columns: 1fr; border-bottom: 0; }
.stApp .cifras.vertical .cifra { border-left: 0; padding-left: 0; border-bottom: 1px solid var(--regla); }
.stApp .cifra { padding: 12px 28px 12px 0; position: relative; }
.stApp .cifra + .cifra { border-left: 1px solid var(--regla); padding-left: 16px; }
.stApp .cifra .ayuda { position: absolute; top: 12px; right: 8px; margin: 0; }
.stApp .cifra .rotulo { font: 600 0.8rem var(--fuente); color: var(--tinta3); }
.stApp .cifra .valor { font: 700 1.85rem var(--fuente); font-variant-numeric: tabular-nums; color: var(--tinta);
                       margin-top: 2px; }
.stApp .cifra .valor small { font-size: 0.95rem; font-weight: 500; color: var(--tinta3); margin-left: 5px; }
.stApp .cifra .valor.texto { font-size: 1.35rem; display: flex; align-items: center; gap: 10px; padding-top: 4px; }
.stApp .cifra .valor i { width: 11px; height: 11px; display: inline-block; border-radius: 2px; }
.stApp .cifra .detalle { font-size: 0.85rem; color: var(--tinta3); margin-top: 2px; }

.stApp .ayuda { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px;
                margin-left: 6px; border: 1px solid var(--tinta3); border-radius: 50%; font: 700 0.65rem var(--fuente);
                color: var(--tinta3); cursor: help; vertical-align: 1px; }
.stApp .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; align-items: center; }
.stApp .chip { font: 600 0.82rem var(--fuente); color: var(--naranja-texto); padding: 3px 10px; border: 1px solid var(--naranja);
               cursor: help; }

.stApp .nota { border-left: 3px solid var(--naranja); padding: 2px 0 2px 14px; margin: 8px 0 10px 0; }
.stApp .nota .rotulo { font: 700 0.75rem var(--fuente); letter-spacing: 0.06em; text-transform: uppercase;
                       color: var(--naranja-texto); }
.stApp .nota p { margin: 1px 0 0 0; color: var(--tinta2); font-size: 1rem; }
.stApp .pie { font: 0.9rem/1.5 var(--fuente); color: var(--tinta3); margin: 4px 0 6px 0; }

.stApp .tabla { width: 100%; border-collapse: collapse; font-size: 0.97rem; margin: 4px 0 6px 0; }
.stApp .tabla th, .stApp .tabla td { border: 0; border-bottom: 1px solid var(--regla); padding: 8px 10px;
                                     background: transparent; color: var(--tinta2); }
.stApp .tabla thead th { font: 700 0.82rem var(--fuente); color: var(--tinta3); text-align: right;
                         border-bottom: 2px solid var(--tinta); }
.stApp .tabla th.fila { text-align: left; font-weight: 600; color: var(--tinta); }
.stApp .tabla td { text-align: right; font-variant-numeric: tabular-nums; }
.stApp .tabla td.ok { color: var(--acento); font-weight: 700; }
.stApp .tabla.texto td, .stApp .tabla.texto thead th { text-align: left; }
.stApp .tabla.texto td.clave { font-weight: 700; color: var(--tinta); white-space: nowrap; }
.stApp .tabla.texto td.cod { font: 700 1rem var(--mono); color: var(--acento); width: 48px; }
.stApp .tabla.texto td.num { color: var(--tinta); white-space: nowrap; font-variant-numeric: tabular-nums; }
.stApp .veredicto { font: 700 0.85rem var(--fuente); white-space: nowrap; }
.stApp .veredicto.si { color: var(--verde-texto); }
.stApp .veredicto.parcial { color: var(--naranja-texto); }
.stApp .veredicto.no { color: var(--tinta3); }

.stApp .ficha { display: grid; grid-template-columns: 200px 1fr; gap: 0 24px; border-top: 1px solid var(--regla); }
.stApp .ficha > div { padding: 9px 0; border-bottom: 1px solid var(--regla); font-size: 0.98rem; color: var(--tinta2); }
.stApp .ficha .clave { font-weight: 700; color: var(--tinta); }
.stApp ul.limitaciones { margin: 0; padding-left: 1.1rem; color: var(--tinta2); }
.stApp ul.limitaciones li { margin: 4px 0; font-size: 0.98rem; }

.stApp .en-uso { border-left: 3px solid var(--acento); background: var(--papel2); padding: 12px 16px; }
.stApp .en-uso .rotulo { font: 700 0.75rem var(--fuente); letter-spacing: 0.06em; text-transform: uppercase; color: var(--acento); }
.stApp .en-uso .nombre { font: 700 1.05rem var(--mono); color: var(--tinta); margin-top: 2px; }
.stApp .en-uso .detalle { font-size: 0.9rem; color: var(--tinta3); margin-top: 2px; }

.stButton button, .stDownloadButton button { border-radius: 0; border: 1px solid var(--tinta); background: transparent;
    color: var(--tinta); font: 600 0.9rem var(--fuente); }
.stButton button:hover, .stDownloadButton button:hover { background: var(--tinta); color: var(--papel); border-color: var(--tinta); }
.stApp .stButton button[kind="primary"] { background: var(--tinta); color: var(--papel); padding: 0.5rem 1.4rem; }
.stApp .stButton button[kind="primary"]:hover { background: var(--acento); border-color: var(--acento); }
.stApp .stButton button p { white-space: nowrap; }
.stApp [data-testid="stFileUploaderDropzone"] { border-radius: 0; border: 1px dashed var(--tinta3); }
.stApp [data-testid="stFileUploaderDropzone"] button { border-radius: 0; }
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stDataFrame"] > div { border-radius: 0; }
.stApp [data-testid="stExpander"] details { border-radius: 0; border-color: var(--regla); }
.stApp [data-testid="stExpander"] summary p { font-size: 0.95rem; color: var(--tinta2); }
.stApp div[data-baseweb="select"], .stApp div[data-baseweb="select"] div { border-radius: 0 !important; }
.stApp code { font-family: var(--mono); font-size: 0.88em; background: var(--papel2); color: var(--tinta);
              padding: 1px 4px; border-radius: 0; }
@media (max-width: 760px) {
    .stApp .cabecera { padding-right: 0; padding-top: 44px; }
    .stApp .cifras { grid-template-columns: repeat(2, 1fr); }
    .stApp .cifra:nth-child(3) { border-left: 0; padding-left: 0; }
    .stApp .cifra:nth-child(n+3) { border-top: 1px solid var(--regla); }
    .stApp .ficha { grid-template-columns: 1fr; }
    .stApp .cabecera h1 { font-size: 1.4rem; }
}
</style>
"""


def resalte(estado):
    return {"atrasado": PAL["naranja"], "a tiempo": PAL["verde"], "adelantado": PAL["acento"]}[estado] + "40"


# ----------------------------------------------------------------------------------------------------------------
# Pestaña 1: predecir una vuelta
# ----------------------------------------------------------------------------------------------------------------
def elegir_al_azar(vueltas):
    fila = vueltas.sample(1).iloc[0]
    st.session_state.update(fecha=pd.Timestamp(fila["fecha"]), vehiculo=fila["vehiculo"], lap=int(fila["lap"]))


def seleccion_vuelta(vueltas):
    """Barra lateral, debajo de la configuración: vuelta y punto de corte de la pestaña «Predecir una vuelta»."""
    fechas = sorted(pd.to_datetime(vueltas["fecha"].unique()))
    st.session_state.setdefault("fecha", fechas[0])
    with st.sidebar:
        html(f'<div class="control-titulo espacio">{t("sel_titulo")}</div>')
        st.caption(t("sel_ayuda"))
        fecha = st.selectbox(t("dia"), fechas, key="fecha", format_func=t.fecha)
        del_dia = vueltas[vueltas["fecha"] == fecha]
        vehiculos = sorted(del_dia["vehiculo"].unique(), key=lambda v: (len(v), v))
        if st.session_state.get("vehiculo") not in vehiculos:
            st.session_state["vehiculo"] = vehiculos[0]
        c1, c2 = st.columns(2)
        vehiculo = c1.selectbox(t("bus"), vehiculos, key="vehiculo")
        laps = sorted(del_dia.loc[del_dia["vehiculo"] == vehiculo, "lap"].astype(int))
        if st.session_state.get("lap") not in laps:
            st.session_state["lap"] = laps[0]
        lap = c2.selectbox(t("vuelta"), laps, key="lap")
        st.button(t("al_azar"), icon=":material/shuffle:", on_click=elegir_al_azar, args=(vueltas,), width="stretch")
        st.session_state.setdefault("k_vuelta", 9)
        k = st.select_slider(t("k_control"), options=list(p.CORTES), key="k_vuelta", help=t("k_ayuda"))
    id_vuelta = del_dia.loc[(del_dia["vehiculo"] == vehiculo) & (del_dia["lap"] == lap), "id"].iloc[0]
    return id_vuelta, fecha, vehiculo, lap, k


def titular(pred, estado, paquete):
    m = f"{abs(round(pred))} min"
    base = {"atrasado": t("titular_atraso", m=m), "adelantado": t("titular_adelanto", m=m)}.get(
        estado, t("titular_tiempo"))
    lo, hi = p.rango(pred, paquete)
    if lo >= 0:
        rango = t("rango_atraso", a=lo, b=hi)
    elif hi <= 0:
        rango = t("rango_adelanto", a=-hi, b=-lo)
    else:
        rango = t("rango_mixto", a=-lo, b=hi)
    return f"{base} ({rango})."


def texto_senal(codigo, valores):
    if codigo == "franja":
        return t("chip_franja", franja=valores["franja"], v=t.num(valores["v"]))
    if codigo == "dia":
        return t("chip_dia", dia=t.dia(valores["dia"]).capitalize(), v=t.num(valores["v"]))
    if codigo == "faltantes":
        return t("chip_faltante") if valores["n"] == 1 else t("chip_faltantes", n=valores["n"])
    return t("chip_sin_flota")


def notas_faltantes(curva, k):
    """Explica los puntos sin registro hasta k, distinguiendo cómo los trata el modelo."""
    dif = curva[[f"dif_{j}" for j in range(1, k + 1)]].to_numpy(dtype=float)
    validos = np.flatnonzero(~np.isnan(dif))
    faltan = np.flatnonzero(np.isnan(dif))
    if not len(faltan):
        return []
    if not len(validos):
        return [t("falta_todo", k=k)]
    internos = [i + 1 for i in faltan if validos[0] < i < validos[-1]]
    externos = [i + 1 for i in faltan if not validos[0] < i < validos[-1]]
    notas = []
    if internos:
        notas.append(t("falta_interno_1" if len(internos) == 1 else "falta_interno_n", p=t.lista(internos)))
    if externos:
        notas.append(t("falta_externo_1" if len(externos) == 1 else "falta_externo_n", p=t.lista(externos)))
    return notas


def grafico_curva(curva, k, pred, mae):
    puntos = np.arange(1, 19)
    dif = curva[[f"dif_{j}" for j in puntos]].to_numpy(dtype=float)
    s_obs, s_hueco, s_despues, s_pred, s_real = (t("serie_obs"), t("serie_hueco"), t("serie_despues"),
                                                 t("serie_pred"), t("serie_real"))
    obs = pd.DataFrame({"punto": puntos[:k], "minutos": dif[:k], "serie": s_obs, "tramo": "observado"})
    capas = [
        alt.Chart(pd.DataFrame({"y0": [-p.UMBRAL_ADELANTO], "y1": [p.UMBRAL_ATRASO]})).mark_rect(
            color=PAL["zona"], opacity=0.9).encode(y="y0:Q", y2="y1:Q"),
        alt.Chart(pd.DataFrame({"t": [t("zona_tiempo")], "x": [1], "y": [4.4]})).mark_text(
            align="left", color=PAL["tinta3"], fontSize=11).encode(x="x:Q", y="y:Q", text="t:N"),
        alt.Chart(pd.DataFrame({"x": [k]})).mark_rule(strokeDash=[4, 4], color=PAL["gris"]).encode(x="x:Q"),
    ]

    # Puntos sin registro entre dos observados: el modelo los completa por interpolación (tramo punteado, marcador hueco).
    huecos = []
    validos = np.flatnonzero(~np.isnan(dif[:k]))
    if len(validos) >= 2:
        completo = np.interp(np.arange(k), validos, dif[:k][validos])
        for i in [i for i in range(validos[0], validos[-1]) if np.isnan(dif[i])]:
            a = max(j for j in validos if j < i)
            b = min(j for j in validos if j > i)
            huecos.append(pd.DataFrame({"punto": puntos[a:b + 1], "minutos": completo[a:b + 1], "serie": s_hueco,
                                        "tramo": f"hueco_{a}"}))
    huecos = pd.concat(huecos, ignore_index=True).drop_duplicates() if huecos else pd.DataFrame(
        columns=["punto", "minutos", "serie", "tramo"])
    despues = pd.DataFrame({"punto": puntos[k - 1:], "minutos": dif[k - 1:], "serie": s_despues, "tramo": "despues"})
    lineas = pd.concat([obs, huecos, despues.dropna()], ignore_index=True)
    lineas["minutos"] = lineas["minutos"].astype(float)
    llenas = lineas[(lineas["serie"] == s_obs) | ((lineas["serie"] == s_despues) & (lineas["punto"] != k))].dropna()
    marcas = pd.concat([llenas, pd.DataFrame({"punto": [18, 18], "minutos": [pred, dif[-1]], "serie": [s_pred, s_real]})],
                       ignore_index=True)
    huecas = huecos[np.isnan(dif[huecos["punto"].astype(int) - 1])] if len(huecos) else huecos

    estilo = {s_obs: (PAL["acento"], "circle"), s_hueco: (PAL["acento_claro"], "circle"),
              s_despues: (PAL["gris"], "circle"), s_pred: (PAL["naranja"], "circle"), s_real: (PAL["tinta"], "diamond")}
    presentes = set(lineas["serie"]) | set(marcas["serie"])
    dominio = [s for s in estilo if s in presentes]
    color = alt.Color("serie:N", title=None, scale=alt.Scale(domain=dominio, range=[estilo[s][0] for s in dominio]),
                      legend=alt.Legend(orient="top"))
    forma = alt.Shape("serie:N", scale=alt.Scale(domain=dominio, range=[estilo[s][1] for s in dominio]), legend=None)
    tooltip = [alt.Tooltip("serie:N", title=t("tt_serie")), alt.Tooltip("punto:Q", title=t("tt_punto"), format="d"),
               alt.Tooltip("minutos:Q", title=t("tt_minutos"), format=".1f")]
    eje_x = alt.X("punto:Q", title=t("eje_punto"), scale=alt.Scale(domain=[0.5, 18.5], nice=False),
                  axis=alt.Axis(values=list(range(1, 19)), format="d", grid=False))
    valores = np.r_[dif[~np.isnan(dif)], pred - mae, pred + mae, -p.UMBRAL_ADELANTO, p.UMBRAL_ATRASO]
    y_min, y_max = np.floor(valores.min()) - 2, np.ceil(valores.max()) + 2
    eje_y = alt.Y("minutos:Q", title=t("eje_minutos"), scale=alt.Scale(domain=[y_min, y_max], nice=False))

    capas.append(alt.Chart(lineas).mark_line(strokeWidth=2).encode(
        x=eje_x, y=eje_y, color=color, detail="tramo:N",
        strokeDash=alt.condition(alt.datum.serie == s_obs, alt.value([1, 0]), alt.value([5, 4]))))
    if len(huecas):
        capas.append(alt.Chart(huecas).mark_point(filled=False, size=90, strokeWidth=2).encode(
            x=eje_x, y=eje_y, color=color, tooltip=tooltip))
    capas.append(alt.Chart(pd.DataFrame({"punto": [18], "lo": [pred - mae], "hi": [pred + mae]})).mark_rule(
        color=PAL["naranja"], strokeWidth=2).encode(
        x="punto:Q", y="lo:Q", y2="hi:Q", tooltip=[alt.Tooltip("lo:Q", title=t("tt_desde"), format=".1f"),
                                                   alt.Tooltip("hi:Q", title=t("tt_hasta"), format=".1f")]))
    capas.append(alt.Chart(marcas).mark_point(filled=True, size=110, stroke=PAL["papel"], strokeWidth=2).encode(
        x=eje_x, y=eje_y, color=color, shape=forma, tooltip=tooltip))
    capas.append(alt.Chart(pd.DataFrame({"x": [k], "y": [y_min + 0.6], "t": [t("ahora", k=k)]})).mark_text(
        align="left", dx=5, color=PAL["tinta3"], fontSize=11).encode(x="x:Q", y="y:Q", text="t:N"))
    return _config(alt.layer(*capas), 300)


def grafico_por_k(datos, k, real):
    datos = datos.assign(lo=datos["pred"] - datos["mae"], hi=datos["pred"] + datos["mae"],
                         actual=datos["k"] == k)
    x = alt.X("k:Q", title=t("eje_k"), scale=alt.Scale(domain=[3, 16], nice=False),
              axis=alt.Axis(values=list(p.CORTES), format="d", grid=False))
    valores = np.r_[datos["lo"], datos["hi"], real]
    dominio = [np.floor(valores.min()) - 1.5, np.ceil(valores.max()) + 1.5]
    y = alt.Y("pred:Q", title=t("eje_pred"), scale=alt.Scale(domain=dominio, nice=False),
              axis=alt.Axis(labelExpr=t.eje(0)))
    tooltip = [alt.Tooltip("k:Q", title="k", format="d"), alt.Tooltip("pred:Q", title=t("tt_pred"), format=".1f"),
               alt.Tooltip("lo:Q", title=t("tt_desde"), format=".1f"), alt.Tooltip("hi:Q", title=t("tt_hasta"), format=".1f")]
    banda = alt.Chart(datos).mark_area(color=PAL["naranja"], opacity=0.16).encode(x=x, y="lo:Q", y2="hi:Q")
    linea = alt.Chart(datos).mark_line(color=PAL["naranja"], strokeWidth=2).encode(x=x, y=y)
    puntos = alt.Chart(datos).mark_point(filled=True, color=PAL["naranja"], stroke=PAL["papel"], strokeWidth=2).encode(
        x=x, y=y, tooltip=tooltip, size=alt.condition(alt.datum.actual, alt.value(230), alt.value(80)))
    anillo = alt.Chart(datos[datos["actual"]]).mark_point(size=520, color=PAL["naranja"], strokeWidth=1.5).encode(x=x, y=y)
    linea_real = alt.Chart(pd.DataFrame({"y": [real]})).mark_rule(color=PAL["tinta"], strokeDash=[5, 4]).encode(y="y:Q")
    etiqueta = alt.Chart(pd.DataFrame({"x": [10.5], "y": [real], "t": [f'{t("real_linea")}: {t.num(real, 0)} min']})
                         ).mark_text(align="center", dy=-9, color=PAL["tinta"], fontSize=11).encode(
        x="x:Q", y="y:Q", text="t:N")
    return _config(alt.layer(banda, linea_real, etiqueta, linea, anillo, puntos), 220)


def grafico_explicacion(efectos):
    corto = {"hor": "imp_Horario", "atr": "imp_Atraso observado", "ctx": "imp_Vehículo y contexto", "gps": "imp_GPS"}
    datos = pd.DataFrame({"grupo": [t(corto[g]) for g in efectos.index], "efecto": efectos.to_numpy()})
    atr, ade = t("hacia_atraso"), t("hacia_adelanto")
    datos["sentido"] = np.where(datos["efecto"] >= 0, atr, ade)
    datos["etiqueta"] = datos["efecto"].map(lambda v: f'{"+" if v >= 0 else "−"}{t.num(abs(v))}')
    lim = max(1.0, datos["efecto"].abs().max() * 1.6)
    base = alt.Chart(datos).encode(
        y=alt.Y("grupo:N", title=None, sort=list(datos["grupo"]),
                axis=alt.Axis(labelFontSize=13, labelPadding=12, ticks=False, domain=False)),
        x=alt.X("efecto:Q", title=t("eje_efecto"), scale=alt.Scale(domain=[-lim, lim]),
                axis=alt.Axis(tickCount=5, labelExpr=t.eje(1))))
    barras = base.mark_bar(height=22).encode(
        color=alt.Color("sentido:N", title=None, legend=alt.Legend(orient="bottom", direction="horizontal"),
                        scale=alt.Scale(domain=[atr, ade], range=[PAL["naranja"], PAL["acento"]])),
        tooltip=[alt.Tooltip("grupo:N", title=t("tt_grupo")), alt.Tooltip("efecto:Q", title=t("tt_efecto"), format="+.2f")])
    texto = base.mark_text(dx=alt.expr(alt.expr.if_(alt.datum.efecto >= 0, 8, -8)), color=PAL["tinta"], fontSize=13,
                           font=FUENTE, align=alt.expr(alt.expr.if_(alt.datum.efecto >= 0, "left", "right"))
                           ).encode(text="etiqueta:N")
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=PAL["tinta3"]).encode(x="x:Q")
    return _config(alt.layer(cero, barras, texto), 50 * len(datos))


def pestana_vuelta(paquetes, curvas, variables, vueltas):
    id_vuelta, fecha, vehiculo, lap, k = seleccion_vuelta(vueltas)
    paquete = paquetes[k]
    fila = variables[(variables["k"] == k) & (variables["id"] == id_vuelta)]
    curva = curvas[curvas["id"] == id_vuelta].iloc[0]
    pred = float(p.predecir(paquete, fila)[0])
    real = float(fila["y"].iloc[0])
    estado = p.clasificar(pred)
    metricas = paquete["metricas_prueba"]
    mae = metricas["MAE"]
    hora = fila["hora_inicio"].iloc[0]

    # Mensaje principal, sugerencia y avisos.
    html(f'<p class="titular" style="--resalte:{resalte(estado)}">{en_linea(titular(pred, estado, paquete))}</p>')
    senales = p.senales(paquete, fila)
    titulo = html_lib.escape(t("chips_ayuda"), quote=True)
    chips = "".join(f'<span class="chip" title="{titulo}">{texto_senal(c, v)}</span>' for c, v in senales)
    html(f'<div class="nota"><div class="rotulo">{t("sugerencia")}</div><p>{t(f"sug_{estado}")}</p></div>'
         + (f'<div class="chips">{chips}</div>' if chips else ""))

    # Cifras y recorrido de la vuelta en una misma fila.
    izquierda, derecha = st.columns([1, 2.7], gap="medium")
    with izquierda:
        cifras([
            (t("cifra_pred"), minutos(pred), t("ayuda_pred"), t("pred_detalle", mae=t.num(mae), k=k)),
            (t("cifra_real"), minutos(real, 0), t("ayuda_real"), t("real_detalle", estado=t.estado(p.clasificar(real)))),
            (t("cifra_error"), minutos(abs(real - pred)), t("ayuda_error")),
        ], vertical=True)
        html(f'<p class="pie">{t("salida", hora=f"{int(hora):02d}:{round(hora % 1 * 60):02d}")}</p>')
    with derecha:
        html(f'<p class="subtitulo">{t("sec_evolucion")}</p>')
        mostrar(grafico_curva(curva, k, pred, mae))
        faltantes = notas_faltantes(curva, k)
        if faltantes:
            with st.expander(t("faltantes_titulo")):
                for texto in faltantes:
                    st.markdown(texto)

    # Debajo: la misma vuelta con cada k y qué grupo de variables pesó más.
    c_k, c_explicacion = st.columns(2, gap="large")
    with c_k:
        html(f'<p class="subtitulo">{t("sec_por_k")}</p>')
        mostrar(grafico_por_k(p.predecir_por_k(paquetes, variables, id_vuelta), k, real))
    with c_explicacion:
        html(f'<p class="subtitulo">{t("sec_explicacion")}</p>')
        mostrar(grafico_explicacion(p.explicar(paquete, fila)))
        with st.expander(t("como_se_calcula")):
            st.markdown(t("explicacion_metodo"))


# ----------------------------------------------------------------------------------------------------------------
# Pestaña 2: probar con otros datos
# ----------------------------------------------------------------------------------------------------------------
def _hora(minutos_dia):
    return f"{int(minutos_dia // 60):02d}:{int(round(minutos_dia % 60)):02d}"


def tabla_predicciones(x, k):
    tabla = x.sort_values(["fecha", "hora_inicio", "vehiculo"])
    return pd.DataFrame({
        t("col_fecha"): tabla["fecha"].map(t.fecha),
        t("col_bus"): tabla["vehiculo"],
        t("col_vuelta"): tabla["lap"].astype(int),
        t("col_salida"): (tabla["hora_inicio"] * 60).map(_hora),
        t("col_atraso_k", k=k): tabla["atr_ultimo"],
        t("col_pred"): tabla["pred"],
        t("col_estado"): tabla["estado"].map(t.estado).str.capitalize(),
        t("col_real"): tabla["y"],
        t("col_error"): tabla["error"],
    })


def grafico_pred_real(x):
    datos = x[x["y"].notna()][["pred", "y", "vehiculo", "lap", "fecha", "error"]].copy()
    datos["fecha"] = datos["fecha"].map(t.fecha)
    lim = [np.floor(min(datos["pred"].min(), datos["y"].min())) - 2, np.ceil(max(datos["pred"].max(), datos["y"].max())) + 2]
    lim = [max(lim[0], -25), min(lim[1], 45)]
    eje = dict(scale=alt.Scale(domain=lim, nice=False), axis=alt.Axis(labelExpr="format(datum.value, 'd')"))
    diagonal = alt.Chart(pd.DataFrame({"a": lim, "b": lim})).mark_line(
        color=PAL["tinta3"], strokeDash=[4, 4], strokeWidth=1.5, clip=True).encode(x="a:Q", y="b:Q")
    banda = alt.Chart(pd.DataFrame({"a": lim, "lo": [lim[0] - 5, lim[1] - 5], "hi": [lim[0] + 5, lim[1] + 5]})).mark_area(
        color=PAL["zona"], opacity=0.9, clip=True).encode(x="a:Q", y="lo:Q", y2="hi:Q")
    puntos = alt.Chart(datos).mark_circle(size=34, opacity=0.55, color=PAL["acento"], clip=True).encode(
        x=alt.X("pred:Q", title=t("eje_pred18"), **eje), y=alt.Y("y:Q", title=t("eje_real18"), **eje),
        tooltip=[alt.Tooltip("fecha:N", title=t("tt_fecha")), alt.Tooltip("vehiculo:N", title=t("tt_bus")),
                 alt.Tooltip("lap:Q", title=t("tt_vuelta")), alt.Tooltip("pred:Q", title=t("tt_pred"), format=".1f"),
                 alt.Tooltip("y:Q", title=t("tt_real"), format=".0f"), alt.Tooltip("error:Q", title=t("tt_error"), format=".1f")])
    texto = alt.Chart(pd.DataFrame({"x": [lim[0] + 1], "y": [lim[1] - 1.5], "t": [t("franja_5")]})).mark_text(
        align="left", fontSize=11, color=PAL["tinta3"]).encode(x="x:Q", y="y:Q", text="t:N")
    return _config(alt.layer(banda, diagonal, puntos, texto).properties(width=520), 460)


def usar_archivo(nombre, contenido):
    """Valida el archivo al confirmarlo; si es válido pasa a ser el archivo en uso (uno a la vez)."""
    st.session_state["n_cargador"] += 1  # vacía el cargador: el archivo pasa a mostrarse solo en la franja
    try:
        vueltas = p.leer_archivo_times(io.BytesIO(contenido))
    except p.ArchivoInvalido as e:
        st.session_state["error_archivo"] = (nombre, e.clave, e.datos)
        return
    st.session_state.pop("error_archivo", None)
    st.session_state["archivo_activo"] = {"nombre": nombre, "contenido": contenido, "vueltas": len(vueltas),
                                          "dias": vueltas["fecha"].nunique()}


def quitar_archivo():
    st.session_state.pop("archivo_activo", None)
    st.session_state.pop("error_archivo", None)
    st.session_state["n_cargador"] += 1


def pestana_archivo():
    seccion(1, "sec_archivo")
    html(f'<p class="entradilla">{en_linea(t("archivo_entradilla"))}</p>')
    with st.expander(t("formato_titulo"), icon=":material/info:"):
        st.markdown(t("formato_texto"))
    ruta_ejemplo = p.DIR_DEMO / "datos" / "ejemplo_times.csv"
    st.session_state.setdefault("n_cargador", 0)
    activo = st.session_state.get("archivo_activo")

    if activo:
        nombre = t("nombre_ejemplo") if activo["nombre"] == "__ejemplo__" else activo["nombre"]
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f'<div class="en-uso"><div class="rotulo">{t("en_uso")}</div><div class="nombre">{nombre}</div>'
                    f'<div class="detalle">{t("vueltas_dias", v=t.entero(activo["vueltas"]), d=activo["dias"])}</div></div>',
                    unsafe_allow_html=True)
        c2.button(t("quitar"), icon=":material/close:", on_click=quitar_archivo, width="stretch")

    subido = st.file_uploader(t("subir_otro") if activo else t("subir"), type=["csv"],
                              key=f"archivo_times_{st.session_state['n_cargador']}")
    if subido is not None and (not activo or activo["contenido"] != subido.getvalue()):
        nota(t("pendiente"), t("pendiente_texto", nombre=subido.name, reemplazo=t("reemplazo") if activo else ""))
        st.button(t("usar"), icon=":material/check:", type="primary", on_click=usar_archivo,
                  args=(subido.name, subido.getvalue()), width="content")
    if st.session_state.get("error_archivo"):
        nombre, clave, datos = st.session_state["error_archivo"]
        nota(t("no_se_pudo"), f"`{nombre}`: {t(clave, **datos)}")

    if ruta_ejemplo.exists():
        c1, c2 = st.columns(2, vertical_alignment="center")
        c1.button(t("usar_ejemplo"), icon=":material/description:", width="stretch",
                  on_click=usar_archivo, args=("__ejemplo__", ruta_ejemplo.read_bytes()))
        c2.download_button(t("descargar_ejemplo"), ruta_ejemplo.read_bytes(), file_name="ejemplo_times.csv",
                           mime="text/csv", icon=":material/download:", width="stretch")

    seccion(2, "sec_corte")
    k = st.select_slider(t("k_archivo"), options=list(p.CORTES), value=9, key="k_archivo")
    if not activo:
        pie(t("sin_archivo"))
        return
    origen = t("nombre_ejemplo") if activo["nombre"] == "__ejemplo__" else activo["nombre"]
    with st.spinner(t("calculando")):
        x, dias_vistos = procesar_archivo(activo["contenido"], k)

    con_real = x["y"].notna()
    conteo = x["estado"].value_counts()
    seccion(3, "sec_resumen_archivo")
    html(f'<p class="fecha-linea">{t("linea_archivo", origen=origen, v=t.entero(len(x)), d=x["fecha"].nunique(), k=k)}</p>')

    def previstas(estado):
        n = conteo.get(estado, 0)
        return f'{t.entero(n)}<small>{t.num(100 * n / len(x), 0)} %</small>'

    cifras([(t("procesadas"), t.entero(len(x)), None), (t("prev_atrasadas"), previstas("atrasado"), None),
            (t("prev_tiempo"), previstas("a tiempo"), None), (t("prev_adelantadas"), previstas("adelantado"), None)])
    if con_real.any():
        errores = x.loc[con_real, "error"]
        html(f'<p class="entradilla" style="margin-top:18px">{en_linea(t("error_archivo", n=t.entero(con_real.sum()), mae=t.num(errores.mean(), 2), pct=t.num(100 * (errores <= 5).mean(), 0)))}</p>')
    if dias_vistos:
        nota(t("atencion"), t("dias_vistos", dias=t.lista(t.fecha(d) for d in dias_vistos)))
    with st.expander(t("detalles")):
        if (~con_real).any():
            st.markdown(t("sin_real", n=t.entero((~con_real).sum())))
        st.markdown(t("sin_gps"))

    seccion(4, "sec_tabla")
    tabla = tabla_predicciones(x, k)
    pie(t("pie_tabla"))
    vista = tabla.copy()
    texto_cols = (t("col_atraso_k", k=k), t("col_real"), t("col_error"))
    for c in texto_cols:
        vista[c] = vista[c].map(lambda v, d=(1 if c == t("col_error") else 0): "—" if pd.isna(v) else t.num(v, d))
    vista[t("col_pred")] = vista[t("col_pred")].map(lambda v: t.num(v, 1))
    st.dataframe(vista, hide_index=True, width="stretch", height=400)
    sep, dec = (";", ",") if t.es else (",", ".")
    st.download_button(t("descargar_pred"), tabla.to_csv(index=False, sep=sep, decimal=dec).encode("utf-8-sig"),
                       file_name=f"predicciones_k{k}.csv", mime="text/csv", icon=":material/download:")

    if con_real.any():
        seccion(5, "sec_pred_real")
        html(f'<p class="entradilla">{t("pred_real_entradilla")}</p>')
        mostrar(grafico_pred_real(x), width="content")


# ----------------------------------------------------------------------------------------------------------------
# Pestaña 3: resultados del proyecto
# ----------------------------------------------------------------------------------------------------------------
COLOR_MODELO = ("Gradient Boosting", "Random Forest", "Ridge", "B1 mediana hora-día", "B2 persistencia")


def colores_modelo():
    return [PAL["acento"], PAL["rf"], PAL["naranja"], PAL["tinta3"], PAL["gris_claro"]]


def etiquetas_finales(datos, columna, separacion):
    """Posición vertical de las etiquetas al final de cada línea, separadas para que no se monten."""
    posiciones = []
    for v in datos[columna]:
        posiciones.append(max(v, posiciones[-1] + separacion) if posiciones else v)
    return posiciones


def grafico_mae_k(tabla):
    datos = tabla.loc[list(COLOR_MODELO)].reset_index().melt(id_vars="modelo", var_name="k", value_name="MAE")
    datos["k"] = datos["k"].astype(int)
    datos["base"] = datos["modelo"].isin(r.LINEAS_BASE)
    datos["nombre"] = datos["modelo"].map(t.modelo)
    color = alt.Color("modelo:N", legend=None, scale=alt.Scale(domain=list(COLOR_MODELO), range=colores_modelo()))
    x = alt.X("k:Q", title=t("eje_k"), scale=alt.Scale(domain=[3.4, 18.5], nice=False),
              axis=alt.Axis(values=list(p.CORTES), format="d", grid=False))
    y = alt.Y("MAE:Q", title=t("eje_mae"), scale=alt.Scale(domain=[1.8, 5.2], nice=False),
              axis=alt.Axis(labelExpr=t.eje(1)))
    tooltip = [alt.Tooltip("nombre:N", title=t("tt_modelo")), alt.Tooltip("k:Q", title="k", format="d"),
               alt.Tooltip("MAE:Q", title=t("tt_mae"), format=".2f")]
    lineas = alt.Chart(datos).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color, detail="modelo:N",
        strokeDash=alt.condition(alt.datum.base, alt.value([5, 4]), alt.value([1, 0])))
    puntos = alt.Chart(datos).mark_point(filled=True, size=70, stroke=PAL["papel"], strokeWidth=2).encode(
        x=x, y=y, color=color, tooltip=tooltip)
    finales = datos[datos["k"] == 15].sort_values("MAE").copy()
    finales["y_etiqueta"] = etiquetas_finales(finales, "MAE", 0.13)
    etiquetas = alt.Chart(finales).mark_text(align="left", dx=10, fontSize=12).encode(
        x=x, y=alt.Y("y_etiqueta:Q"), text="nombre:N", color=color)
    return _config(alt.layer(lineas, puntos, etiquetas), 360)


def grafico_comparacion(ic, k):
    datos = ic[ic["k"] == k].copy()
    datos["nombre"] = datos["modelo"].map(t.modelo)
    mejor, otros, base = t("mejor_modelo"), t("otros_modelos"), t("linea_base")
    datos["grupo"] = np.select([datos["modelo"] == "Gradient Boosting", datos["modelo"].isin(r.LINEAS_BASE)],
                               [mejor, base], otros)
    datos = datos.sort_values("MAE")
    escala = alt.Scale(domain=[mejor, otros, base], range=[PAL["acento"], PAL["tinta2"], PAL["gris_claro"]])
    y = alt.Y("nombre:N", title=None, sort=list(datos["nombre"]), axis=alt.Axis(ticks=False, domain=False, labelPadding=10))
    x = alt.X("MAE:Q", title=t("eje_mae_ic"), axis=alt.Axis(labelExpr=t.eje(1)))
    tooltip = [alt.Tooltip("nombre:N", title=t("tt_modelo")), alt.Tooltip("MAE:Q", title=t("tt_mae"), format=".2f"),
               alt.Tooltip("IC_inf:Q", title=t("tt_ic_desde"), format=".2f"),
               alt.Tooltip("IC_sup:Q", title=t("tt_ic_hasta"), format=".2f")]
    rango = alt.Chart(datos).mark_rule(strokeWidth=2, opacity=0.45).encode(
        y=y, x="IC_inf:Q", x2="IC_sup:Q", tooltip=tooltip, color=alt.Color("grupo:N", legend=None, scale=escala))
    punto = alt.Chart(datos).mark_point(filled=True, opacity=1, stroke=PAL["papel"], strokeWidth=2).encode(
        y=y, x=x, tooltip=tooltip, color=alt.Color("grupo:N", title=None, legend=alt.Legend(orient="top"), scale=escala),
        size=alt.condition(alt.datum.grupo == mejor, alt.value(150), alt.value(80)))
    datos["etiqueta"] = datos["MAE"].map(lambda v: t.num(v, 2))
    texto = alt.Chart(datos).mark_text(align="left", dx=8, dy=-9, fontSize=11, font=FUENTE,
                                       color=PAL["tinta"]).encode(y=y, x="MAE:Q", text="etiqueta:N")
    return _config(alt.layer(rango, punto, texto).resolve_legend(color="independent"), 30 * len(datos))


def grafico_importancia(tabla):
    datos = tabla.assign(grupo=tabla["grupo"].map(lambda g: t(f"imp_{g}")))
    datos["etiqueta"] = datos["aumento_MAE"].map(lambda v: f"+{t.num(v, 2)}")
    base = alt.Chart(datos).encode(
        y=alt.Y("grupo:N", title=None, sort="-x", axis=alt.Axis(ticks=False, domain=False, labelPadding=10)),
        x=alt.X("aumento_MAE:Q", title=t("eje_importancia"), scale=alt.Scale(domain=[0, datos["aumento_MAE"].max() * 1.35]),
                axis=alt.Axis(tickCount=4, labelExpr=t.eje(1))))
    barras = base.mark_bar(height=22, color=PAL["acento"]).encode(
        tooltip=[alt.Tooltip("grupo:N", title=t("tt_grupo")), alt.Tooltip("aumento_MAE:Q", title=t("tt_aumento"), format=".2f")])
    texto = base.mark_text(align="left", dx=6, fontSize=12, font=FUENTE, color=PAL["tinta"]).encode(text="etiqueta:N")
    return _config(alt.layer(barras, texto), 230)


def grafico_ablacion(tabla):
    datos = tabla.reset_index().melt(id_vars="variables", var_name="k", value_name="MAE")
    datos["k"] = datos["k"].astype(int)
    orden = list(tabla.index)
    datos["nombre"] = datos["variables"].map(lambda v: t(f"abl_{v}"))
    nombres = [t(f"abl_{v}") for v in orden]
    colores = [PAL["gris_claro"], PAL["acento_claro"], PAL["acento"], PAL["naranja"], PAL["tinta3"]][:len(orden)]
    color = alt.Color("nombre:N", legend=None, scale=alt.Scale(domain=nombres, range=colores))
    x = alt.X("k:Q", title=t("eje_k"), scale=alt.Scale(domain=[3.4, 19.5], nice=False),
              axis=alt.Axis(values=list(p.CORTES), format="d", grid=False))
    y = alt.Y("MAE:Q", title=t("eje_mae"), scale=alt.Scale(domain=[1.95, 2.85], nice=False),
              axis=alt.Axis(labelExpr=t.eje(1)))
    tooltip = [alt.Tooltip("nombre:N", title=t("tt_grupo")), alt.Tooltip("k:Q", title="k", format="d"),
               alt.Tooltip("MAE:Q", title=t("tt_mae"), format=".2f")]
    discontinua = alt.condition(alt.datum.variables == orden[-1], alt.value([5, 4]), alt.value([1, 0]))
    lineas = alt.Chart(datos).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, strokeDash=discontinua)
    puntos = alt.Chart(datos).mark_point(filled=True, size=50).encode(x=x, y=y, color=color, tooltip=tooltip)
    finales = datos[datos["k"] == 15].sort_values("MAE").copy()
    finales["y_etiqueta"] = etiquetas_finales(finales, "MAE", 0.045)
    etiquetas = alt.Chart(finales).mark_text(align="left", dx=10, fontSize=12).encode(
        x=x, y=alt.Y("y_etiqueta:Q"), text="nombre:N", color=color)
    return _config(alt.layer(lineas, puntos, etiquetas), 300)


def grafico_confusion(matriz):
    filas = matriz.div(matriz.sum(axis=1), axis=0)
    datos = matriz.reset_index(names="real").melt(id_vars="real", var_name="predicho", value_name="vueltas")
    datos["pct"] = [filas.loc[a, b] for a, b in zip(datos["real"], datos["predicho"])]
    datos["etiqueta"] = [f"{n}\n{t.num(100 * q, 0)} %" for n, q in zip(datos["vueltas"], datos["pct"])]
    datos["real"] = datos["real"].map(t.estado)
    datos["predicho"] = datos["predicho"].map(t.estado)
    orden = [t.estado(e) for e in ("adelantado", "a tiempo", "atrasado")]
    base = alt.Chart(datos).encode(
        x=alt.X("predicho:N", title=t("eje_predicha"), sort=orden,
                axis=alt.Axis(labelAngle=0, orient="top", ticks=False, domain=False)),
        y=alt.Y("real:N", title=t("eje_real_clase"), sort=orden, axis=alt.Axis(ticks=False, domain=False)))
    celdas = base.mark_rect(stroke=PAL["papel"], strokeWidth=3).encode(
        color=alt.Color("pct:Q", legend=None, scale=alt.Scale(domain=[0, 1], range=[PAL["celda_min"], PAL["celda_max"]])),
        tooltip=[alt.Tooltip("real:N", title=t("tt_real_clase")), alt.Tooltip("predicho:N", title=t("tt_predicha")),
                 alt.Tooltip("vueltas:Q", title=t("tt_vueltas")), alt.Tooltip("pct:Q", title=t("tt_pct_fila"), format=".0%")])
    texto = base.mark_text(fontSize=14, font=FUENTE, lineBreak="\n").encode(
        text="etiqueta:N", color=alt.condition(alt.datum.pct > 0.5, alt.value(PAL["texto_celda"]), alt.value(PAL["tinta"])))
    return _config(alt.layer(celdas, texto), 280)


def tabla_html(tabla, decimales, umbral=None):
    encabezado = "".join(f"<th>k = {k}</th>" for k in tabla.columns)
    filas = []
    for nombre, fila in tabla.iterrows():
        celdas = "".join(f'<td class="ok">{t.num(v, decimales)}</td>' if umbral is not None and v >= umbral
                         else f"<td>{t.num(v, decimales)}</td>" for v in fila)
        filas.append(f'<tr><th class="fila">{nombre}</th>{celdas}</tr>')
    return f'<table class="tabla"><thead><tr><th></th>{encabezado}</tr></thead><tbody>{"".join(filas)}</tbody></table>'


def tabla_texto(columnas, filas):
    """Tabla de texto: filas como listas de (contenido HTML, clase CSS)."""
    encabezado = "".join(f"<th>{c}</th>" for c in columnas)
    cuerpo = "".join("<tr>" + "".join(f'<td class="{clase}">{valor}</td>' for valor, clase in fila) + "</tr>"
                     for fila in filas)
    return f'<table class="tabla texto"><thead><tr>{encabezado}</tr></thead><tbody>{cuerpo}</tbody></table>'


def pestana_resultados():
    mejor = r.mejor_por_k()
    m9 = mejor.loc[9]
    mejora = 100 * (1 - m9["MAE"] / m9["MAE_B1"])

    seccion(1, "sec_resumen")
    html(f'<p class="fecha-linea">{t("linea_prueba", n=t.entero(1061))}</p>'
         f'<p class="titular" style="--resalte:{PAL["acento"]}33">'
         f'{en_linea(t("resumen_titular", modelo=m9["modelo"], mae=t.num(m9["MAE"], 2), mejora=t.num(mejora, 0)))}</p>')
    cifras([
        (t("cifra_mejor"), f'<span class="valor texto">{m9["modelo"]}</span>', t("ayuda_mejor")),
        (t("cifra_mae9"), minutos(m9["MAE"], 2), t("ayuda_mae", v=t.num(m9["MAE"], 2))),
        (t("cifra_dentro5"), f'{t.num(m9["pct_err_5"], 1)}<small>%</small>', t("ayuda_dentro5", v=t.num(m9["pct_err_5"], 0))),
        (t("cifra_r2"), t.num(m9["R2"], 2), t("ayuda_r2", v=t.num(100 * m9["R2"], 0))),
    ])

    seccion(2, "sec_error_k")
    html(f'<p class="entradilla">{t("error_k_entradilla")}</p>')
    mostrar(grafico_mae_k(r.mae_por_k()))
    pie(t("pie_error_k"))

    seccion(3, "sec_comparacion")
    k = st.select_slider(t("k_comparar"), options=list(p.CORTES), value=9, key="k_resultados")
    html(f'<p class="entradilla">{t("comparacion_entradilla", k=k)}</p>')
    mostrar(grafico_comparacion(r.intervalos(), k))
    pie(t("pie_comparacion"))

    seccion(4, "sec_que_explica")
    html(f'<p class="entradilla">{t("que_explica_entradilla")}</p>')
    html(f'<p class="subtitulo">{t("importancia_titulo")}</p>')
    mostrar(grafico_importancia(r.importancia_grupos()))
    html(f'<p class="subtitulo">{t("ablacion_titulo")}</p>')
    mostrar(grafico_ablacion(r.ablacion()))
    pie(t("pie_importancia"))

    seccion(5, "sec_clasificacion")
    c9 = r.clasificacion_k9().loc["Gradient Boosting"]
    html(f'<p class="entradilla">{en_linea(t("clasificacion_entradilla", v=t.num(10 * c9["exactitud"], 0)))}</p>')
    cifras([
        (t("cifra_exactitud"), t.num(c9["exactitud"], 3), t("ayuda_exactitud")),
        (t("cifra_f1"), t.num(c9["F1_macro"], 3), t("ayuda_f1")),
        (t("cifra_f1_atr"), t.num(c9["F1_atrasado"], 3), t("ayuda_f1_atr")),
        (t("cifra_aucpr"), t.num(c9["AUC_PR_atrasado"], 3), t("ayuda_aucpr")),
    ])
    st.write("")
    mostrar(grafico_confusion(r.matriz_confusion()))
    pie(t("pie_confusion"))

    seccion(6, "sec_meta")
    html(f'<p class="entradilla">{t("meta_entradilla")}</p>')
    meta = r.meta_80()
    meta.index = [t(f"meta_{fila}") for fila in meta.index]
    html(tabla_html(meta, 3, umbral=0.80))
    pie(t("pie_meta"))

    seccion(7, "sec_hipotesis")
    veredictos = VEREDICTO[t.idioma]
    filas = [[(codigo, "cod"), (enunciado, ""), (f'<span class="veredicto {clase}">{veredictos[clase]}</span>', ""),
              (evidencia, "")] for codigo, enunciado, clase, evidencia in HIPOTESIS[t.idioma]]
    html(tabla_texto(["", t("col_hipotesis"), t("col_resultado"), t("col_evidencia")], filas))


# ----------------------------------------------------------------------------------------------------------------
# Pestaña 4: acerca del modelo
# ----------------------------------------------------------------------------------------------------------------
GPS_REGISTROS, GPS_FUERA_DE_OPERACION = 1_630_431, 47_921  # Monitoring.csv completo y filas con lap = 0 (informe E2)


def grafico_embudo(pasos):
    """Cuántos registros quedan tras cada paso de la limpieza; la etiqueta indica cuántos se quitaron."""
    datos = pd.DataFrame(pasos, columns=["paso", "quedan", "quitados"])
    datos["tipo"] = ["extremo"] + ["paso"] * (len(datos) - 2) + ["extremo"]
    datos["etiqueta"] = [t.entero(q) + (f"  (−{t.entero(d)})" if d else "") for q, d in zip(datos["quedan"], datos["quitados"])]
    base = alt.Chart(datos).encode(
        y=alt.Y("paso:N", title=None, sort=list(datos["paso"]), axis=alt.Axis(ticks=False, domain=False, labelPadding=10)),
        x=alt.X("quedan:Q", title=None, axis=None, scale=alt.Scale(domain=[0, datos["quedan"].max() * 1.8])))
    barras = base.mark_bar(height=20).encode(
        color=alt.Color("tipo:N", legend=None, scale=alt.Scale(domain=["extremo", "paso"],
                                                               range=[PAL["acento"], PAL["acento_claro"]])),
        tooltip=[alt.Tooltip("paso:N", title=t("tt_paso")), alt.Tooltip("quedan:Q", title=t("tt_cantidad"), format=",")])
    texto = base.mark_text(align="left", dx=6, fontSize=12, font=FUENTE, color=PAL["tinta"]).encode(text="etiqueta:N")
    return _config(alt.layer(barras, texto), 38 * len(datos))


def pestana_acerca():
    filtros = r.filtros()
    gps = r.limpieza_gps()
    seccion(1, "sec_datos")
    html(f'<p class="entradilla">{t("datos_entradilla")}</p>')
    total = int(filtros["Vueltas en Times"])
    validas = int(filtros["Con punto 18 válido (sin regla)"])
    usadas = int(filtros["Con regla de 60 min"])
    en_operacion = int(gps["Puntos GPS en operación"])
    limpios = int(gps["Puntos GPS tras la limpieza"])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html(f'<p class="subtitulo">{t("embudo_vueltas")}</p>')
        mostrar(grafico_embudo([(t("e_registradas"), total, 0), (t("e_sin18"), validas, total - validas),
                                (t("e_extremas"), usadas, validas - usadas), (t("e_usadas"), usadas, 0)]))
    with c2:
        html(f'<p class="subtitulo">{t("embudo_gps")}</p>')
        sin_fuera = GPS_REGISTROS - GPS_FUERA_DE_OPERACION
        mostrar(grafico_embudo([(t("e_gps_total"), GPS_REGISTROS, 0), (t("e_gps_fuera"), sin_fuera, GPS_FUERA_DE_OPERACION),
                                (t("e_gps_rep"), en_operacion, sin_fuera - en_operacion),
                                (t("e_gps_saltos"), limpios, en_operacion - limpios), (t("e_gps_usados"), limpios, 0)]))
    pie(t("pie_embudo"))
    html(f'<p class="subtitulo">{t("decisiones_titulo")}</p>')
    html('<div class="ficha">' + "".join(f'<div class="clave">{a}</div><div>{b}</div>' for a, b in t("decisiones"))
         + "</div>")

    seccion(2, "sec_modelo")
    html('<div class="ficha">' + "".join(f'<div class="clave">{a}</div><div>{b}</div>' for a, b in t("modelo_items"))
         + "</div>")

    seccion(3, "sec_metricas")
    m9 = r.mejor_por_k().loc[9]
    c9 = r.clasificacion_k9().loc["Gradient Boosting"]
    valores = {"mae": f'{t.num(m9["MAE"], 2)} min', "dentro5": f'{t.num(m9["pct_err_5"], 1)} %',
               "r2": t.num(m9["R2"], 3), "exactitud": t.num(c9["exactitud"], 3), "f1": t.num(c9["F1_macro"], 3),
               "aucpr": t.num(c9["AUC_PR_atrasado"], 3)}
    filas = [[(nombre, "clave"), (valores[clave], "num"), (significado, "")] for nombre, clave, significado in t("metricas")]
    html(tabla_texto([t("col_metrica"), t("col_valor"), t("col_significa")], filas))

    seccion(4, "sec_limitaciones")
    html('<ul class="limitaciones">' + "".join(f"<li>{x}</li>" for x in t("limitaciones")) + "</ul>")


# ----------------------------------------------------------------------------------------------------------------
def main():
    global t, PAL
    t, PAL = preferencias()
    estilos()
    html(f'<div class="cabecera"><p class="antetitulo">{t("antetitulo")}</p><h1>{t("titulo")}</h1></div>')
    try:
        paquetes = modelos()
        curvas, variables = semana()
    except FileNotFoundError:
        st.error(t("sin_modelos"))
        st.stop()

    # Solo se ejecuta la pestaña abierta; así la barra lateral (selección de la vuelta) aparece solo en la primera.
    vueltas = variables[variables["k"] == p.CORTES[0]][["id", "fecha", "vehiculo", "lap"]]
    pestanas = st.tabs([t(c) for c in PESTANAS], key="pestana", on_change="rerun", bind="query-params")
    contenido = (lambda: pestana_vuelta(paquetes, curvas, variables, vueltas), pestana_archivo, pestana_resultados,
                 pestana_acerca)
    for pestana, mostrar_pestana in zip(pestanas, contenido):
        if pestana.open:
            with pestana:
                mostrar_pestana()


main()
