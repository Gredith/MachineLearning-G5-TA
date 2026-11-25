# Demo interactiva

Aplicación web local hecha con [Streamlit](https://streamlit.io) para la presentación final. Muestra el mejor modelo del proyecto, **Gradient Boosting**, prediciendo el adelanto o atraso de una vuelta en el punto de control 18 con la información disponible hasta el punto k, y resume los resultados del informe del E2.

La demo **no modifica** los notebooks, `src/` ni `results/`: solo los usa.

## Cómo probarla

Requisitos: Python 3.13, `Times.csv` y `Monitoring.csv` en `data/raw/`.

```bash
# 0. Traer la rama
git fetch
git checkout feat/demo

# 1. Entorno (si ya lo tienes, basta con instalar los requisitos)
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # ahora incluye streamlit

# 2. Datos por corte (una sola vez, unos 3 min)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 02_preprocesamiento_variables_por_corte.ipynb
cd ..

# 3. Entrenar y guardar los modelos de la demo (menos de 1 min)
python demo/entrenar_modelos.py

# 4. Abrir la demo (se abre sola en el navegador, en http://localhost:8501)
cd demo
streamlit run app.py
```

- El paso 3 comprueba que el error de cada modelo coincida con el informe (MAE 2,62 min en k = 9) y se detiene si no coincide.
- El paso 2 reescribe las tablas de `results/` con el mismo contenido; si `git status` las marca como modificadas (solo cambian los saltos de línea), se descartan con `git restore results/`.
- Para cerrar la demo: `Ctrl + C` en la terminal.

## Recorrido por la página

La página tiene tres pestañas. A la izquierda, una barra lateral fija con la selección de la vuelta, para no perderla de vista al recorrer las secciones de la pestaña 1.

### 1. Predecir una vuelta

Se elige una vuelta de la **semana de prueba** (26 ene – 1 feb 2025, que el modelo no vio al entrenar) y el punto de corte k. El botón «Vuelta al azar» elige una cualquiera.

| Sección | Qué muestra |
|---|---|
| 01 Resultado de la predicción | Frase en lenguaje simple («…llegará al punto 18 con unos 6 min de atraso»), estado previsto (atrasado / a tiempo / adelantado), valor real y error |
| 02 Evolución del atraso en la vuelta | Atraso en cada punto de control: lo observado hasta k, lo que pasó después, la predicción con su margen de error y el valor real. Los puntos sin registro que el modelo completa se ven como círculos huecos |
| 03 Explicación de la predicción | Cuánto cambia la predicción si el modelo no conociera cada grupo de variables (horario, atraso observado, vehículo y flota, GPS) |
| 04 Confiabilidad del modelo | Error medio con ese k y avisos cuando la vuelta es difícil (franja de 16 a 19 h, sábados, datos faltantes) |
| 05 Metodología del modelo | Resumen del modelo, las variables y la prevención de fuga |

### 2. Probar con otros datos

Se sube un CSV con el **mismo formato de `Times.csv`** y el modelo predice cada vuelta.

1. Subir el archivo → aparece «pendiente de confirmar».
2. Pulsar **«Usar archivo»** → se procesa y queda la franja «Archivo en uso» (un archivo a la vez; «Quitar» lo descarta).
3. Elegir k → resumen, tabla de predicciones (descargable en CSV para Excel) y gráfico de predicción frente al valor real.

También se puede usar el **archivo de ejemplo**, que es la semana de prueba en el formato original (lo genera el paso 3). Con k = 9 da un error de 2,65 min, frente a 2,62 del informe: la diferencia se debe a que el archivo no trae GPS.

### 3. Resultados del proyecto

Las cifras del informe, leídas directamente de `results/tables/` (no necesita los datos crudos):
resumen, error según k frente a las líneas base, comparación de los 12 modelos con IC 95 % (con selector de k), importancia por grupos y ablación, clasificación con matriz de confusión, métricas frente a la meta de 0,80 e hipótesis H1 a H6.

## Decisiones que conviene conocer

- **Mismo modelo que el informe.** `entrenar_modelos.py` reentrena Gradient Boosting con los hiperparámetros de `results/tables/04_hiperparametros.csv` y el mismo entrenamiento (6 a 25 de enero).
- **Explicación por grupos.** Cada barra de «Explicación de la predicción» reemplaza un grupo de variables por un valor típico y mide cuánto cambia la predicción: es la idea de la ablación del informe aplicada a una vuelta. Los efectos no suman exactamente la predicción.
- **Vueltas en curso.** En «Probar con otros datos», las vueltas sin punto 18 también se predicen. Para que `construir_corte` no las descarte se les pone un valor provisional en `dif_18`; su hora real en el punto 18 queda vacía, así que ninguna variable lo usa, y se borra del resultado.
- **Sin GPS en archivos subidos.** Esas variables se completan con valores típicos; en el informe el GPS no mejoraba el error (H4).
- **El archivo de ejemplo no son datos nuevos.** Es la semana de prueba del informe: datos que el modelo no vio al entrenar, pero no posteriores al 1 de febrero.
- **Datos fuera del repositorio.** `demo/modelos/` y `demo/datos/` están en `.gitignore` porque contienen datos del curso; cada uno los regenera con el paso 3.

## Qué revisar

- [ ] La demo corre en tu máquina siguiendo los pasos de arriba.
- [ ] El paso 3 imprime los mismos MAE del informe (2,66 / 2,67 / 2,62 / 2,46 / 2,05).
- [ ] Los textos y las cifras de las tres pestañas son coherentes con el informe.
- [ ] La explicación de la predicción y los avisos se entienden sin conocer el código.
- [ ] Ideas para la exposición: qué vueltas mostrar (sugerencia: una bien predicha y una difícil, por ejemplo un sábado por la tarde).

## Limitaciones conocidas

- Las fuentes se cargan desde Google Fonts: sin internet la página usa fuentes del sistema (funciona igual, pero se ve menos cuidada).
- Los textos del recuadro para subir archivos («Upload», «200MB per file») son de Streamlit y salen en inglés.
- La barra lateral se ve en las tres pestañas, aunque solo se usa en la primera (Streamlit no permite ocultarla según la pestaña).
- En pantallas de celular, el gráfico de explicación queda apretado; en laptop y proyector se ve bien.

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz de las tres pestañas |
| `prediccion.py` | Lógica sin interfaz: cargar modelos, predecir, explicar, mensajes y lectura de archivos |
| `resultados.py` | Lectura de las tablas de `results/tables/` |
| `entrenar_modelos.py` | Entrena y guarda los modelos y genera la semana de prueba y el archivo de ejemplo |
| `.streamlit/config.toml` | Tema de colores y configuración de Streamlit |
