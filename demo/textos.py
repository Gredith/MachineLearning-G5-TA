"""Textos de la demo en español (por defecto) e inglés.

Cada texto se pide con una clave: t("clave", valor=...). Los valores entre llaves se completan con format.
"""
import pandas as pd

IDIOMAS = {"es": "ES", "en": "EN"}

TEXTOS = {
    "es": {
        # Cabecera y controles
        "antetitulo": "Aprendizaje de Máquina (1INF02) · PUCP 2026-2 · Grupo 5",
        "titulo": "Predicción del adelanto y atraso de los buses urbanos de Cusco en su último punto de control",
        "idioma": "Idioma",
        "tema": "Tema",
        "tema_sistema": "Sistema",
        "tema_claro": "Claro",
        "tema_oscuro": "Oscuro",
        "sin_modelos": "No se encontraron los modelos. Ejecuta primero `python demo/entrenar_modelos.py` desde la raíz "
                       "del repositorio.",
        "pestana_vuelta": "Predecir una vuelta",
        "pestana_archivo": "Probar con otros datos",
        "pestana_resultados": "Resultados del proyecto",
        "pestana_acerca": "Acerca del modelo",

        # Estados
        "atrasado": "atrasado",
        "a tiempo": "a tiempo",
        "adelantado": "adelantado",

        # Pestaña 1: selección
        "sel_titulo": "Selección de la vuelta",
        "sel_ayuda": "Semana de prueba (26 ene – 1 feb 2025): vueltas que el modelo no vio al entrenar.",
        "dia": "Día",
        "bus": "Bus",
        "vuelta": "Vuelta del día",
        "al_azar": "Vuelta al azar",
        "k_control": "Punto de control observado (k)",
        "k_ayuda": "Hasta qué punto de control ya se conoce la vuelta. El modelo solo usa lo ocurrido antes de ese "
                   "instante.",

        # Pestaña 1: resultado
        "titular_atraso": "El bus llegará al punto 18 con unos **{m} de atraso**",
        "titular_adelanto": "El bus llegará al punto 18 con unos **{m} de adelanto**",
        "titular_tiempo": "El bus llegará al punto 18 **a tiempo**",
        "rango_atraso": "entre {a} y {b} min de atraso",
        "rango_adelanto": "entre {a} y {b} min de adelanto",
        "rango_mixto": "entre {a} min de adelanto y {b} min de atraso",
        "sugerencia": "Sugerencia",
        "sug_atrasado": "Regular la salida del siguiente bus de la ruta y avisar a los usuarios del retraso.",
        "sug_a tiempo": "No hace falta intervenir; basta con seguir el recorrido.",
        "sug_adelantado": "Retener el bus unos minutos en un punto de control para no adelantarse al horario.",
        "cifra_pred": "Predicción en el punto 18",
        "cifra_real": "Valor real",
        "cifra_error": "Error de esta predicción",
        "ayuda_pred": "Minutos respecto al horario en el último punto de control: negativo = adelanto, positivo = atraso.",
        "ayuda_real": "Lo que realmente registró el sistema en el punto 18 (el modelo no lo conocía).",
        "ayuda_error": "Diferencia, en minutos, entre la predicción y el valor real.",
        "chip_franja": "Franja {franja}: error medio {v} min",
        "chip_dia": "{dia}: error medio {v} min",
        "chip_faltantes": "{n} registros faltantes completados",
        "chip_faltante": "1 registro faltante completado",
        "chip_sin_flota": "Sin otros buses en los 30 min previos",
        "chips_ayuda": "Situaciones en las que el modelo suele equivocarse más.",

        # Pestaña 1: gráfico de la vuelta
        "sec_evolucion": "Evolución del atraso en la vuelta",
        "serie_obs": "Observado hasta k",
        "serie_hueco": "Sin registro (completado por el modelo)",
        "serie_despues": "Lo que pasó después",
        "serie_pred": "Predicción del modelo",
        "serie_real": "Valor real",
        "zona_tiempo": "zona «a tiempo» (−3 a +5 min)",
        "ahora": "ahora (punto {k})",
        "eje_punto": "Punto de control",
        "eje_minutos": "Minutos respecto al horario (− adelanto, + atraso)",
        "tt_serie": "Serie",
        "tt_punto": "Punto de control",
        "tt_minutos": "Minutos (− adelanto, + atraso)",
        "tt_desde": "Desde (min)",
        "tt_hasta": "Hasta (min)",
        "faltantes_titulo": "Puntos sin registro en esta vuelta",
        "falta_todo": "Esta vuelta no tiene registros entre los puntos 1 y {k}; el modelo usa valores típicos.",
        "falta_interno_1": "El punto {p} no tiene registro: el modelo lo completa uniendo los puntos vecinos "
                           "(círculo hueco sobre la línea punteada).",
        "falta_interno_n": "Los puntos {p} no tienen registro: el modelo los completa uniendo los puntos vecinos "
                           "(círculos huecos sobre la línea punteada).",
        "falta_externo_1": "El punto {p} no tiene registro y no queda entre dos puntos registrados: el modelo usa un "
                           "valor típico del entrenamiento.",
        "falta_externo_n": "Los puntos {p} no tienen registro y no quedan entre dos puntos registrados: el modelo usa "
                           "valores típicos del entrenamiento.",
        "y": "y",

        # Pestaña 1: predicción según k
        "sec_por_k": "Predicción según el punto de corte",
        "salida": "Salida programada: {hora} h",
        "pred_detalle": "± {mae} min (error típico con k = {k})",
        "real_detalle": "Estado real: {estado}",
        "eje_k": "Punto de control observado (k)",
        "eje_pred": "Predicción en el punto 18 (min)",
        "pred_k": "Predicción con cada k",
        "banda_k": "± error medio con ese k",
        "real_linea": "Valor real",
        "k_actual": "k elegido",

        # Pestaña 1: explicación
        "sec_explicacion": "Qué pesó en esta predicción",
        "eje_efecto": "Cambio en la predicción (min)",
        "hacia_atraso": "Empuja hacia atraso",
        "hacia_adelanto": "Empuja hacia adelanto",
        "tt_grupo": "Grupo",
        "tt_efecto": "Efecto (min)",
        "como_se_calcula": "Cómo se calcula",
        "explicacion_metodo": "Cada barra compara la predicción con la que daría el modelo si ese grupo de datos se "
                              "reemplazara por un valor típico del entrenamiento. Por eso un grupo puede mover la "
                              "predicción de una vuelta aunque, en promedio, aporte poco (como el GPS). Los efectos no "
                              "suman exactamente la predicción.",
        "grupo_hor": "Horario (hora de inicio y día)",
        "grupo_atr": "Atraso observado hasta k",
        "grupo_ctx": "Vehículo y otros buses",
        "grupo_gps": "GPS",

        # Pestaña 2
        "sec_archivo": "Archivo de entrada",
        "archivo_entradilla": "Sube un CSV con el formato de *Times.csv* y el modelo predecirá el punto 18 de cada vuelta.",
        "formato_titulo": "Formato del archivo",
        "formato_texto": "Columnas obligatorias: `id`, `date`, `vehicle_id_id`, `lap`, `start_time`, `end_time` y "
                         "`difference_time`, con las listas de los 18 puntos de control entre llaves, como en "
                         "*Times.csv*. Las vueltas que aún no llegan al punto 18 también se predicen. Conviene incluir "
                         "todas las vueltas del día: el modelo usa el atraso de los otros buses como contexto.",
        "en_uso": "Archivo en uso",
        "vueltas_dias": "{v} vueltas · {d} días",
        "quitar": "Quitar",
        "subir_otro": "Subir otro archivo CSV (formato Times.csv)",
        "subir": "Archivo CSV con formato Times.csv",
        "pendiente": "Archivo cargado · pendiente de confirmar",
        "pendiente_texto": "`{nombre}`. Pulsa «Usar archivo» para procesarlo{reemplazo}.",
        "reemplazo": " y reemplazar el actual",
        "usar": "Usar archivo",
        "no_se_pudo": "No se pudo usar el archivo",
        "usar_ejemplo": "Usar archivo de ejemplo",
        "descargar_ejemplo": "Descargar archivo de ejemplo",
        "nombre_ejemplo": "ejemplo_times.csv (semana de prueba)",
        "sec_corte": "Punto de corte",
        "k_archivo": "Hasta qué punto de control se conoce cada vuelta (k)",
        "sin_archivo": "Sube un archivo y confírmalo, o usa el archivo de ejemplo, para ver las predicciones.",
        "calculando": "Calculando predicciones…",
        "sec_resumen_archivo": "Resumen",
        "linea_archivo": "{origen} · {v} vueltas · {d} días · k = {k}",
        "procesadas": "Vueltas procesadas",
        "prev_atrasadas": "Atrasadas previstas",
        "prev_tiempo": "A tiempo previstas",
        "prev_adelantadas": "Adelantadas previstas",
        "error_archivo": "En las **{n}** vueltas con valor real en el punto 18, el modelo se equivocó en promedio "
                         "**{mae} min**, y en el **{pct} %** el error fue de 5 min o menos.",
        "detalles": "Detalles del cálculo",
        "sin_real": "{n} vueltas no tienen un valor real válido en el punto 18 (en curso, sin registro o con más de "
                    "60 min de desviación): se predicen, pero no entran en el cálculo del error.",
        "sin_gps": "El archivo no trae GPS, así que esas variables se completan con valores típicos; en el informe el "
                   "GPS no mejoraba el error (H4).",
        "atencion": "Atención",
        "dias_vistos": "El archivo incluye días que el modelo usó para entrenar ({dias}). En esas vueltas el error será "
                       "menor que con datos realmente nuevos.",
        "sec_tabla": "Predicciones por vuelta",
        "pie_tabla": "Minutos respecto al horario en el punto 18 (− adelanto, + atraso). «—» indica que no hay dato.",
        "col_fecha": "Fecha",
        "col_bus": "Bus",
        "col_vuelta": "Vuelta",
        "col_salida": "Salida",
        "col_atraso_k": "Atraso en k={k}",
        "col_pred": "Predicción",
        "col_estado": "Estado",
        "col_real": "Real",
        "col_error": "Error",
        "descargar_pred": "Descargar predicciones (CSV)",
        "sec_pred_real": "Predicción frente al valor real",
        "pred_real_entradilla": "Cada punto es una vuelta: cuanto más cerca de la diagonal, mejor la predicción.",
        "eje_pred18": "Predicción en el punto 18 (min)",
        "eje_real18": "Valor real en el punto 18 (min)",
        "franja_5": "franja: error de 5 min o menos",
        "tt_fecha": "Fecha",
        "tt_bus": "Bus",
        "tt_vuelta": "Vuelta",
        "tt_pred": "Predicción (min)",
        "tt_real": "Real (min)",
        "tt_error": "Error (min)",
        "error_lectura": "No se pudo leer el archivo como CSV.",
        "error_columnas": "Faltan columnas obligatorias: {faltan}.",
        "error_listas": "Las columnas start_time, end_time y difference_time deben tener 18 valores entre llaves, "
                        "como en Times.csv: {{05:45:00,05:47:00,...}}.",
        "error_vacio": "El archivo no tiene filas.",

        # Pestaña 3
        "sec_resumen": "Resumen de resultados",
        "linea_prueba": "Semana de prueba · 26 ene – 1 feb 2025 · {n} vueltas",
        "resumen_titular": "A mitad de la vuelta (k = 9), **{modelo}** predice la desviación en el último punto de "
                           "control con un error medio de **{mae} min**, un {mejora} % menos que la mejor línea base.",
        "cifra_mejor": "Mejor modelo",
        "cifra_mae9": "Error medio (MAE) en k = 9",
        "cifra_dentro5": "Vueltas con error ≤ 5 min",
        "cifra_r2": "R² en k = 9",
        "ayuda_mejor": "El de menor error en validación en los cinco cortes.",
        "ayuda_mae": "MAE: en promedio, la predicción se aleja {v} min de lo que realmente pasó.",
        "ayuda_dentro5": "De cada 100 vueltas, en {v} el error fue de 5 min o menos.",
        "ayuda_r2": "R²: el modelo explica el {v} % de la variación del atraso final entre vueltas (1 sería perfecto).",
        "sec_error_k": "El error según el punto de corte",
        "error_k_entradilla": "El error casi no cambia entre k = 4 y 9 y baja con claridad en k = 12 y 15.",
        "eje_mae": "Error absoluto medio en prueba (min)",
        "pie_error_k": "Líneas continuas: modelos; discontinuas: líneas base. B1 usa solo la hora y el día; B2 supone que "
                       "el atraso final es el último atraso observado.",
        "tt_modelo": "Modelo",
        "tt_mae": "MAE (min)",
        "sec_comparacion": "Comparación de todos los modelos",
        "k_comparar": "Punto de corte (k) para comparar",
        "comparacion_entradilla": "Doce modelos y cuatro líneas base ordenados por su error con k = {k}.",
        "eje_mae_ic": "Error absoluto medio en prueba (min), con IC 95 %",
        "mejor_modelo": "Mejor modelo",
        "otros_modelos": "Otros modelos",
        "linea_base": "Línea base",
        "tt_ic_desde": "IC 95 % desde",
        "tt_ic_hasta": "IC 95 % hasta",
        "pie_comparacion": "La línea fina es el intervalo de confianza del 95 % (remuestreo por días). Es ancho porque la "
                           "prueba tiene 7 días; aun así, Gradient Boosting supera a las líneas base y a Ridge en todos "
                           "los cortes (Wilcoxon, p < 0,001).",
        "sec_que_explica": "Qué explica las predicciones",
        "que_explica_entradilla": "El horario es lo que más pesa; el atraso observado solo ayuda desde k = 12 y el GPS "
                                  "casi no aporta.",
        "importancia_titulo": "Importancia por grupo (k = 9)",
        "eje_importancia": "Cuánto sube el error al desordenar el grupo (min)",
        "tt_aumento": "Aumento (min)",
        "ablacion_titulo": "Error al agregar cada grupo de variables",
        "pie_importancia": "Izquierda: importancia por permutación en prueba. Derecha: ablación acumulativa con Gradient "
                           "Boosting y los mismos hiperparámetros.",
        "abl_Horario": "Solo horario",
        "abl_+ Atraso": "+ atraso observado",
        "abl_+ Vehículo/flota": "+ vehículo y flota",
        "abl_+ GPS (completo)": "+ GPS (completo)",
        "abl_Completo sin vehículo/flota": "Completo sin vehículo y flota",
        "imp_Horario": "Horario",
        "imp_Atraso observado": "Atraso observado",
        "imp_Vehículo y contexto": "Vehículo y flota",
        "imp_GPS": "GPS",
        "sec_clasificacion": "Clasificación: atrasado, a tiempo o adelantado",
        "clasificacion_entradilla": "El mismo problema en tres estados. Con k = 9, Gradient Boosting acierta el estado "
                                    "en **{v} de cada 10** vueltas.",
        "cifra_exactitud": "Exactitud",
        "cifra_f1": "F1 macro",
        "cifra_f1_atr": "F1 de atrasado",
        "cifra_aucpr": "AUC-PR de atrasado",
        "ayuda_exactitud": "Proporción de vueltas cuyo estado se predijo bien.",
        "ayuda_f1": "Promedio del equilibrio entre aciertos y omisiones de las tres clases; no se deja engañar por la "
                    "clase más frecuente.",
        "ayuda_f1_atr": "Qué tan bien detecta las vueltas atrasadas sin dar falsas alarmas.",
        "ayuda_aucpr": "Calidad del ordenamiento por riesgo de atraso (1 sería perfecto; al azar sería ~0,28).",
        "eje_predicha": "Clase predicha",
        "eje_real_clase": "Clase real",
        "tt_real_clase": "Real",
        "tt_predicha": "Predicha",
        "tt_vueltas": "Vueltas",
        "tt_pct_fila": "% de la fila",
        "pie_confusion": "Número de vueltas y porcentaje de cada clase real. La mayor confusión es entre «a tiempo» y "
                         "«atrasado». Convertir la predicción en minutos a estados da casi lo mismo (exactitud 0,796).",
        "sec_meta": "Métricas frente a la meta de 0,80",
        "meta_entradilla": "En negrita, las métricas que superan la meta de los docentes.",
        "meta_R² (P1)": "R² (regresión)",
        "meta_Vueltas con error ≤ 5 min (P1)": "Vueltas con error ≤ 5 min (regresión)",
        "meta_Exactitud (P2)": "Exactitud (clasificación)",
        "meta_F1 macro (P2)": "F1 macro (clasificación)",
        "meta_AUC-PR atrasado (P2)": "AUC-PR de atrasado (clasificación)",
        "pie_meta": "A mitad de la vuelta la superan el error ≤ 5 min y el AUC-PR; con k = 15, casi todas. El límite lo "
                    "pone la información disponible, no el preprocesamiento.",
        "sec_hipotesis": "Hipótesis",
        "col_hipotesis": "Hipótesis",
        "col_resultado": "Resultado",
        "col_evidencia": "Evidencia",

        # Pestaña 4
        "sec_datos": "Datos y limpieza",
        "datos_entradilla": "De cada decisión de limpieza se comprobó con los datos que no quitara información real.",
        "embudo_vueltas": "Vueltas (Times.csv)",
        "embudo_gps": "Registros GPS (Monitoring.csv)",
        "e_registradas": "Vueltas registradas",
        "e_sin18": "Sin valor válido en el punto 18",
        "e_extremas": "Desviación mayor a 60 min",
        "e_usadas": "Vueltas usadas",
        "e_gps_total": "Registros GPS",
        "e_gps_fuera": "Bus fuera de operación",
        "e_gps_rep": "Filas repetidas",
        "e_gps_saltos": "Saltos imposibles (> 144 km/h)",
        "e_gps_usados": "Registros usados",
        "tt_paso": "Paso",
        "tt_cantidad": "Cantidad",
        "pie_embudo": "Las 4.146 vueltas usadas se dividen por fecha: 3.085 para entrenar (6 al 25 ene) y 1.061 para la "
                      "prueba final (26 ene al 1 feb).",
        "decisiones_titulo": "Por qué cada decisión",
        "decisiones": [
            ("Sin punto 18", "Sin valor real no hay respuesta que aprender ni comparar: esas vueltas no se usan."),
            ("Más de 60 min", "Solo 24 puntos caen entre 40 y 60 min y 138 superan los 120 (hasta −1052): son errores "
                              "de registro. La regla de Tukey habría borrado 210 vueltas con atrasos reales."),
            ("Faltantes intermedios", "Un punto sin registro entre dos registrados se completa uniendo sus vecinos; si "
                                      "no hay vecinos, se usa la mediana del entrenamiento."),
            ("GPS", "Se quitan los puntos a los que se llega y de los que se sale a más de 144 km/h. Las vueltas que "
                    "empiezan lejos de la ruta o con horario irregular se marcan con una variable, no se borran."),
        ],
        "sec_modelo": "El modelo",
        "modelo_items": [
            ("Tipo de aprendizaje", "Supervisado: cada vuelta histórica trae su respuesta real en el punto 18."),
            ("Modelo", "Gradient Boosting: muchos árboles de decisión pequeños, cada uno corrige los errores del "
                       "anterior. Fue el de menor error entre 12 algoritmos."),
            ("Un modelo por cada k", "Con más puntos observados cambian las variables disponibles, así que cada punto de "
                                     "corte tiene su propio modelo."),
            ("Entrenamiento", "Días del 6 al 25 de enero, con validación por bloques de días en orden. La semana del 26 "
                              "ene al 1 feb se evaluó una sola vez."),
            ("Variables", "Horario programado, atraso observado en los puntos 1 a k, vehículo, vuelta anterior del "
                          "mismo bus, atraso de otros buses en los 30 min previos y GPS."),
            ("Sin fuga de información", "Nada posterior al instante de predicción entra al modelo."),
            ("Herramientas", "Python con scikit-learn (modelos y pipelines), pandas y numpy (datos), scipy "
                             "(pruebas estadísticas) y Streamlit (esta demo)."),
        ],
        "sec_metricas": "Cómo leer las métricas",
        "col_metrica": "Métrica",
        "col_valor": "Valor (k = 9)",
        "col_significa": "Qué significa",
        "metricas": [
            ("MAE", "mae", "En promedio, la predicción se aleja esos minutos de lo que realmente pasó. Menos es mejor."),
            ("Error ≤ 5 min", "dentro5", "Porcentaje de vueltas en que la predicción acertó con 5 min de margen o menos."),
            ("R²", "r2", "Qué parte de la variación del atraso final entre vueltas explica el modelo (1 = todo)."),
            ("Exactitud", "exactitud", "Porcentaje de vueltas cuyo estado (atrasado, a tiempo, adelantado) se acertó."),
            ("F1 macro", "f1", "Promedio del F1 de las tres clases: penaliza ignorar las clases poco frecuentes."),
            ("AUC-PR de atrasado", "aucpr", "Qué tan bien ordena las vueltas por riesgo de atraso (1 = perfecto)."),
        ],
        "sec_limitaciones": "Limitaciones",
        "limitaciones": [
            "Solo 27 días de enero de 2025 y una ruta: no se sabe cómo se comporta en otras temporadas.",
            "Las vueltas con desviaciones de más de 60 min (errores de registro) no se pueden predecir.",
            "Que start_time sea el horario programado es un supuesto respaldado por los datos, no confirmado.",
            "Los puntos de control no tienen coordenadas, así que el GPS solo se relaciona por tiempo.",
        ],
    },

    "en": {
        "antetitulo": "Machine Learning (1INF02) · PUCP 2026-2 · Group 5",
        "titulo": "Predicting how early or late Cusco's urban buses reach their last control point",
        "idioma": "Language",
        "tema": "Theme",
        "tema_sistema": "System",
        "tema_claro": "Light",
        "tema_oscuro": "Dark",
        "sin_modelos": "Models not found. First run `python demo/entrenar_modelos.py` from the repository root.",
        "pestana_vuelta": "Predict a trip",
        "pestana_archivo": "Try other data",
        "pestana_resultados": "Project results",
        "pestana_acerca": "About the model",

        "atrasado": "late",
        "a tiempo": "on time",
        "adelantado": "early",

        "sel_titulo": "Trip selection",
        "sel_ayuda": "Test week (Jan 26 – Feb 1, 2025): trips the model did not see during training.",
        "dia": "Day",
        "bus": "Bus",
        "vuelta": "Trip of the day",
        "al_azar": "Random trip",
        "k_control": "Observed control point (k)",
        "k_ayuda": "Up to which control point the trip is already known. The model only uses what happened before that "
                   "moment.",

        "titular_atraso": "The bus will reach point 18 about **{m} late**",
        "titular_adelanto": "The bus will reach point 18 about **{m} early**",
        "titular_tiempo": "The bus will reach point 18 **on time**",
        "rango_atraso": "between {a} and {b} min late",
        "rango_adelanto": "between {a} and {b} min early",
        "rango_mixto": "between {a} min early and {b} min late",
        "sugerencia": "Suggestion",
        "sug_atrasado": "Adjust the departure of the next bus on the route and tell passengers about the delay.",
        "sug_a tiempo": "No action needed; just keep following the trip.",
        "sug_adelantado": "Hold the bus a few minutes at a control point so it does not run ahead of schedule.",
        "cifra_pred": "Prediction at point 18",
        "cifra_real": "Actual value",
        "cifra_error": "Error of this prediction",
        "ayuda_pred": "Minutes relative to the schedule at the last control point: negative = early, positive = late.",
        "ayuda_real": "What the system actually recorded at point 18 (unknown to the model).",
        "ayuda_error": "Difference, in minutes, between the prediction and the actual value.",
        "chip_franja": "{franja} slot: mean error {v} min",
        "chip_dia": "{dia}: mean error {v} min",
        "chip_faltantes": "{n} missing records filled in",
        "chip_faltante": "1 missing record filled in",
        "chip_sin_flota": "No other buses in the previous 30 min",
        "chips_ayuda": "Situations where the model tends to make larger errors.",

        "sec_evolucion": "Delay along the trip",
        "serie_obs": "Observed up to k",
        "serie_hueco": "Missing (filled in by the model)",
        "serie_despues": "What happened next",
        "serie_pred": "Model prediction",
        "serie_real": "Actual value",
        "zona_tiempo": "on-time zone (−3 to +5 min)",
        "ahora": "now (point {k})",
        "eje_punto": "Control point",
        "eje_minutos": "Minutes vs. schedule (− early, + late)",
        "tt_serie": "Series",
        "tt_punto": "Control point",
        "tt_minutos": "Minutes (− early, + late)",
        "tt_desde": "From (min)",
        "tt_hasta": "To (min)",
        "faltantes_titulo": "Missing records in this trip",
        "falta_todo": "This trip has no records between points 1 and {k}; the model uses typical values.",
        "falta_interno_1": "Point {p} has no record: the model fills it in by joining the neighboring points (hollow "
                           "circle on the dotted line).",
        "falta_interno_n": "Points {p} have no record: the model fills them in by joining the neighboring points "
                           "(hollow circles on the dotted line).",
        "falta_externo_1": "Point {p} has no record and is not between two recorded points: the model uses a typical "
                           "training value.",
        "falta_externo_n": "Points {p} have no record and are not between two recorded points: the model uses typical "
                           "training values.",
        "y": "and",

        "sec_por_k": "Prediction by cut point",
        "salida": "Scheduled departure: {hora}",
        "pred_detalle": "± {mae} min (typical error with k = {k})",
        "real_detalle": "Actual status: {estado}",
        "eje_k": "Observed control point (k)",
        "eje_pred": "Prediction at point 18 (min)",
        "pred_k": "Prediction with each k",
        "banda_k": "± mean error with that k",
        "real_linea": "Actual value",
        "k_actual": "selected k",

        "sec_explicacion": "What drove this prediction",
        "eje_efecto": "Change in the prediction (min)",
        "hacia_atraso": "Pushes toward late",
        "hacia_adelanto": "Pushes toward early",
        "tt_grupo": "Group",
        "tt_efecto": "Effect (min)",
        "como_se_calcula": "How it is computed",
        "explicacion_metodo": "Each bar compares the prediction with the one the model would give if that group of data "
                              "were replaced by a typical training value. That is why a group can move one trip's "
                              "prediction even if, on average, it adds little (like GPS). The effects do not add up "
                              "exactly to the prediction.",
        "grupo_hor": "Schedule (start time and day)",
        "grupo_atr": "Delay observed up to k",
        "grupo_ctx": "Vehicle and other buses",
        "grupo_gps": "GPS",

        "sec_archivo": "Input file",
        "archivo_entradilla": "Upload a CSV with the *Times.csv* format and the model will predict point 18 for each trip.",
        "formato_titulo": "File format",
        "formato_texto": "Required columns: `id`, `date`, `vehicle_id_id`, `lap`, `start_time`, `end_time` and "
                         "`difference_time`, with the lists of the 18 control points in braces, as in *Times.csv*. "
                         "Trips that have not reached point 18 yet are also predicted. Include all the trips of the "
                         "day: the model uses the delay of the other buses as context.",
        "en_uso": "File in use",
        "vueltas_dias": "{v} trips · {d} days",
        "quitar": "Remove",
        "subir_otro": "Upload another CSV file (Times.csv format)",
        "subir": "CSV file in Times.csv format",
        "pendiente": "File uploaded · waiting for confirmation",
        "pendiente_texto": "`{nombre}`. Press “Use file” to process it{reemplazo}.",
        "reemplazo": " and replace the current one",
        "usar": "Use file",
        "no_se_pudo": "The file could not be used",
        "usar_ejemplo": "Use example file",
        "descargar_ejemplo": "Download example file",
        "nombre_ejemplo": "ejemplo_times.csv (test week)",
        "sec_corte": "Cut point",
        "k_archivo": "Up to which control point each trip is known (k)",
        "sin_archivo": "Upload a file and confirm it, or use the example file, to see the predictions.",
        "calculando": "Computing predictions…",
        "sec_resumen_archivo": "Summary",
        "linea_archivo": "{origen} · {v} trips · {d} days · k = {k}",
        "procesadas": "Trips processed",
        "prev_atrasadas": "Predicted late",
        "prev_tiempo": "Predicted on time",
        "prev_adelantadas": "Predicted early",
        "error_archivo": "On the **{n}** trips with an actual value at point 18, the model's mean error was "
                         "**{mae} min**, and **{pct}%** had an error of 5 min or less.",
        "detalles": "Calculation details",
        "sin_real": "{n} trips have no valid actual value at point 18 (in progress, not recorded or more than 60 min "
                    "off): they are predicted but not counted in the error.",
        "sin_gps": "The file has no GPS, so those variables are filled with typical values; in the report GPS did not "
                   "improve the error (H4).",
        "atencion": "Note",
        "dias_vistos": "The file includes days the model used for training ({dias}). On those trips the error will be "
                       "lower than with truly new data.",
        "sec_tabla": "Predictions by trip",
        "pie_tabla": "Minutes vs. schedule at point 18 (− early, + late). “—” means no data.",
        "col_fecha": "Date",
        "col_bus": "Bus",
        "col_vuelta": "Trip",
        "col_salida": "Departure",
        "col_atraso_k": "Delay at k={k}",
        "col_pred": "Prediction",
        "col_estado": "Status",
        "col_real": "Actual",
        "col_error": "Error",
        "descargar_pred": "Download predictions (CSV)",
        "sec_pred_real": "Prediction vs. actual value",
        "pred_real_entradilla": "Each dot is a trip: the closer to the diagonal, the better the prediction.",
        "eje_pred18": "Prediction at point 18 (min)",
        "eje_real18": "Actual value at point 18 (min)",
        "franja_5": "band: error of 5 min or less",
        "tt_fecha": "Date",
        "tt_bus": "Bus",
        "tt_vuelta": "Trip",
        "tt_pred": "Prediction (min)",
        "tt_real": "Actual (min)",
        "tt_error": "Error (min)",
        "error_lectura": "The file could not be read as a CSV.",
        "error_columnas": "Missing required columns: {faltan}.",
        "error_listas": "The start_time, end_time and difference_time columns must hold 18 values in braces, as in "
                        "Times.csv: {{05:45:00,05:47:00,...}}.",
        "error_vacio": "The file has no rows.",

        "sec_resumen": "Summary of results",
        "linea_prueba": "Test week · Jan 26 – Feb 1, 2025 · {n} trips",
        "resumen_titular": "At mid-trip (k = 9), **{modelo}** predicts the deviation at the last control point with a "
                           "mean error of **{mae} min**, {mejora}% less than the best baseline.",
        "cifra_mejor": "Best model",
        "cifra_mae9": "Mean error (MAE) at k = 9",
        "cifra_dentro5": "Trips with error ≤ 5 min",
        "cifra_r2": "R² at k = 9",
        "ayuda_mejor": "Lowest validation error at all five cut points.",
        "ayuda_mae": "MAE: on average, the prediction is {v} min away from what actually happened.",
        "ayuda_dentro5": "Out of 100 trips, {v} had an error of 5 min or less.",
        "ayuda_r2": "R²: the model explains {v}% of the variation in the final delay between trips (1 would be perfect).",
        "sec_error_k": "Error by cut point",
        "error_k_entradilla": "The error barely changes between k = 4 and 9 and drops clearly at k = 12 and 15.",
        "eje_mae": "Mean absolute error on test (min)",
        "pie_error_k": "Solid lines: models; dashed: baselines. B1 uses only the hour and the day; B2 assumes the final "
                       "delay equals the last observed delay.",
        "tt_modelo": "Model",
        "tt_mae": "MAE (min)",
        "sec_comparacion": "All models compared",
        "k_comparar": "Cut point (k) to compare",
        "comparacion_entradilla": "Twelve models and four baselines sorted by their error with k = {k}.",
        "eje_mae_ic": "Mean absolute error on test (min), with 95% CI",
        "mejor_modelo": "Best model",
        "otros_modelos": "Other models",
        "linea_base": "Baseline",
        "tt_ic_desde": "95% CI from",
        "tt_ic_hasta": "95% CI to",
        "pie_comparacion": "The thin line is the 95% confidence interval (resampling whole days). It is wide because the "
                           "test has 7 days; still, Gradient Boosting beats the baselines and Ridge at every cut point "
                           "(Wilcoxon, p < 0.001).",
        "sec_que_explica": "What drives the predictions",
        "que_explica_entradilla": "The schedule weighs most; the observed delay only helps from k = 12 and GPS adds "
                                  "almost nothing.",
        "importancia_titulo": "Importance by group (k = 9)",
        "eje_importancia": "Increase in error when the group is shuffled (min)",
        "tt_aumento": "Increase (min)",
        "ablacion_titulo": "Error when adding each group of variables",
        "pie_importancia": "Left: permutation importance on test. Right: cumulative ablation with Gradient Boosting and "
                           "the same hyperparameters.",
        "abl_Horario": "Schedule only",
        "abl_+ Atraso": "+ observed delay",
        "abl_+ Vehículo/flota": "+ vehicle and fleet",
        "abl_+ GPS (completo)": "+ GPS (full)",
        "abl_Completo sin vehículo/flota": "Full without vehicle and fleet",
        "imp_Horario": "Schedule",
        "imp_Atraso observado": "Observed delay",
        "imp_Vehículo y contexto": "Vehicle and fleet",
        "imp_GPS": "GPS",
        "sec_clasificacion": "Classification: late, on time or early",
        "clasificacion_entradilla": "The same problem as three statuses. With k = 9, Gradient Boosting gets the status "
                                    "right in **{v} out of 10** trips.",
        "cifra_exactitud": "Accuracy",
        "cifra_f1": "Macro F1",
        "cifra_f1_atr": "F1 for late",
        "cifra_aucpr": "PR-AUC for late",
        "ayuda_exactitud": "Share of trips whose status was predicted correctly.",
        "ayuda_f1": "Average balance between hits and misses over the three classes; not fooled by the most frequent "
                    "class.",
        "ayuda_f1_atr": "How well it detects late trips without false alarms.",
        "ayuda_aucpr": "Quality of the ranking by risk of being late (1 would be perfect; random would be ~0.28).",
        "eje_predicha": "Predicted class",
        "eje_real_clase": "Actual class",
        "tt_real_clase": "Actual",
        "tt_predicha": "Predicted",
        "tt_vueltas": "Trips",
        "tt_pct_fila": "% of row",
        "pie_confusion": "Number of trips and share of each actual class. Most confusion is between “on time” and "
                         "“late”. Turning the minutes prediction into statuses gives almost the same (accuracy 0.796).",
        "sec_meta": "Metrics vs. the 0.80 target",
        "meta_entradilla": "In bold, the metrics above the instructors' target.",
        "meta_R² (P1)": "R² (regression)",
        "meta_Vueltas con error ≤ 5 min (P1)": "Trips with error ≤ 5 min (regression)",
        "meta_Exactitud (P2)": "Accuracy (classification)",
        "meta_F1 macro (P2)": "Macro F1 (classification)",
        "meta_AUC-PR atrasado (P2)": "PR-AUC for late (classification)",
        "pie_meta": "At mid-trip, error ≤ 5 min and PR-AUC pass it; with k = 15, almost all do. The limit is the "
                    "available information, not the preprocessing.",
        "sec_hipotesis": "Hypotheses",
        "col_hipotesis": "Hypothesis",
        "col_resultado": "Result",
        "col_evidencia": "Evidence",

        "sec_datos": "Data and cleaning",
        "datos_entradilla": "Each cleaning decision was checked against the data so that no real information was lost.",
        "embudo_vueltas": "Trips (Times.csv)",
        "embudo_gps": "GPS records (Monitoring.csv)",
        "e_registradas": "Recorded trips",
        "e_sin18": "No valid value at point 18",
        "e_extremas": "Deviation above 60 min",
        "e_usadas": "Trips used",
        "e_gps_total": "GPS records",
        "e_gps_fuera": "Bus out of service",
        "e_gps_rep": "Duplicate rows",
        "e_gps_saltos": "Impossible jumps (> 144 km/h)",
        "e_gps_usados": "Records used",
        "tt_paso": "Step",
        "tt_cantidad": "Count",
        "pie_embudo": "The 4,146 trips used are split by date: 3,085 for training (Jan 6–25) and 1,061 for the final "
                      "test (Jan 26 – Feb 1).",
        "decisiones_titulo": "Why each decision",
        "decisiones": [
            ("No point 18", "Without an actual value there is no answer to learn or compare: those trips are not used."),
            ("More than 60 min", "Only 24 points fall between 40 and 60 min while 138 exceed 120 (down to −1052): they "
                                 "are recording errors. Tukey's rule would have removed 210 trips with real delays."),
            ("Missing in between", "A missing point between two recorded ones is filled by joining its neighbors; with "
                                   "no neighbors, the training median is used."),
            ("GPS", "Points reached and left at more than 144 km/h are removed. Trips starting far from the route or with "
                    "irregular schedules are flagged with a variable, not deleted."),
        ],
        "sec_modelo": "The model",
        "modelo_items": [
            ("Type of learning", "Supervised: each past trip comes with its actual answer at point 18."),
            ("Model", "Gradient Boosting: many small decision trees, each one fixing the errors of the previous ones. It "
                      "had the lowest error among 12 algorithms."),
            ("One model per k", "With more observed points the available variables change, so each cut point has its own "
                                "model."),
            ("Training", "Days from Jan 6 to 25, validated on blocks of consecutive days. The week of Jan 26 – Feb 1 was "
                         "evaluated only once."),
            ("Variables", "Scheduled times, delay observed at points 1 to k, vehicle, previous trip of the same bus, "
                          "delay of other buses in the previous 30 min and GPS."),
            ("No information leakage", "Nothing after the prediction moment goes into the model."),
            ("Tools", "Python with scikit-learn (models and pipelines), pandas and numpy (data), scipy (statistical "
                      "tests) and Streamlit (this demo)."),
        ],
        "sec_metricas": "How to read the metrics",
        "col_metrica": "Metric",
        "col_valor": "Value (k = 9)",
        "col_significa": "What it means",
        "metricas": [
            ("MAE", "mae", "On average, the prediction is that many minutes away from what happened. Lower is better."),
            ("Error ≤ 5 min", "dentro5", "Share of trips where the prediction was within 5 min."),
            ("R²", "r2", "How much of the variation in the final delay between trips the model explains (1 = all)."),
            ("Accuracy", "exactitud", "Share of trips whose status (late, on time, early) was right."),
            ("Macro F1", "f1", "Average F1 over the three classes: it penalizes ignoring the rare classes."),
            ("PR-AUC for late", "aucpr", "How well it ranks trips by risk of being late (1 = perfect)."),
        ],
        "sec_limitaciones": "Limitations",
        "limitaciones": [
            "Only 27 days of January 2025 and one route: behavior in other seasons is unknown.",
            "Trips more than 60 min off (recording errors) cannot be predicted.",
            "That start_time is the scheduled time is an assumption backed by the data, not confirmed.",
            "Control points have no coordinates, so GPS is only linked by time.",
        ],
    },
}

DIAS = {"es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
MESES = {"es": {1: "ene", 2: "feb"}, "en": {1: "Jan", 2: "Feb"}}

MODELOS_EN = {"Regresión lineal": "Linear regression", "Regresión lineal (GD)": "Linear regression (GD)",
              "Árbol de decisión": "Decision tree"}
LINEAS_BASE = {"es": {"B0 a tiempo": "B0 · a tiempo", "B1 mediana hora-día": "B1 · mediana hora y día",
                      "B2 persistencia": "B2 · persistencia", "B3 lineal atraso k": "B3 · lineal con atraso k"},
               "en": {"B0 a tiempo": "B0 · on time", "B1 mediana hora-día": "B1 · median by hour and day",
                      "B2 persistencia": "B2 · persistence", "B3 lineal atraso k": "B3 · linear on delay at k"}}

# Hipótesis: (código, enunciado, resultado si / parcial / no, evidencia), como la Tabla X del informe del E2.
HIPOTESIS = {
    "es": [
        ("H1", "El atraso observado mejora frente a usar solo el horario", "parcial", "Desde k = 12: 2,78 → 2,51 min"),
        ("H2", "Los árboles superan a los lineales", "si", "GB y RF < Ridge en los 5 cortes (p < 0,001)"),
        ("H3", "El error baja al observar más puntos", "parcial", "Igual de k = 4 a 9; baja en 12 y 15"),
        ("H4", "El GPS aporta información", "no", "Cambia el MAE en ± 0,01 min"),
        ("H5", "Las pérdidas absoluta y Huber rinden mejor", "parcial", "Absoluta mejor en prueba, no en validación"),
        ("H6", "El contexto de flota ayuda en horas de más atraso", "parcial", "−0,24 min de 16 a 19 h; −0,33 los sábados"),
    ],
    "en": [
        ("H1", "Observed delay improves on using only the schedule", "parcial", "From k = 12: 2.78 → 2.51 min"),
        ("H2", "Tree models beat linear models", "si", "GB and RF < Ridge at all 5 cut points (p < 0.001)"),
        ("H3", "Error drops as more points are observed", "parcial", "Flat from k = 4 to 9; drops at 12 and 15"),
        ("H4", "GPS adds information", "no", "Changes MAE by ± 0.01 min"),
        ("H5", "Absolute and Huber losses do better", "parcial", "Absolute better on test, not on validation"),
        ("H6", "Fleet context helps in the most delayed hours", "parcial", "−0.24 min from 4 to 7 pm; −0.33 on Saturdays"),
    ],
}
VEREDICTO = {"es": {"si": "✓ Se cumple", "parcial": "~ Parcial", "no": "✗ No se cumple"},
             "en": {"si": "✓ Holds", "parcial": "~ Partly", "no": "✗ Does not hold"}}


class Traductor:
    """Textos y formato de números según el idioma elegido."""

    def __init__(self, idioma):
        self.idioma = idioma if idioma in TEXTOS else "es"

    def __call__(self, clave, **valores):
        texto = TEXTOS[self.idioma].get(clave, TEXTOS["es"].get(clave, clave))
        return texto.format(**valores) if isinstance(texto, str) else texto

    @property
    def es(self):
        return self.idioma == "es"

    def num(self, valor, decimales=1):
        texto = f"{valor:.{decimales}f}"
        return texto.replace(".", ",") if self.es else texto

    def entero(self, n):
        texto = f"{int(n):,}"
        return texto.replace(",", ".") if self.es else texto

    def eje(self, decimales=1):
        """Expresión de Vega para las etiquetas de los ejes con el separador decimal del idioma."""
        formato = f"format(datum.value, '.{decimales}f')"
        return f"replace({formato}, '.', ',')" if self.es else formato

    def fecha(self, f):
        f = pd.Timestamp(f)
        dia = DIAS[self.idioma][f.day_of_week]
        mes = MESES[self.idioma][f.month]
        return f"{dia} {f.day} {mes}" if self.es else f"{dia}, {mes} {f.day}"

    def dia(self, indice):
        return DIAS[self.idioma][indice]

    def estado(self, codigo):
        return self(codigo)

    def modelo(self, nombre):
        if nombre in LINEAS_BASE["es"]:
            return LINEAS_BASE[self.idioma][nombre]
        return nombre if self.es else MODELOS_EN.get(nombre, nombre)

    def lista(self, elementos):
        elementos = [str(e) for e in elementos]
        if len(elementos) == 1:
            return elementos[0]
        return ", ".join(elementos[:-1]) + f" {self('y')} " + elementos[-1]
