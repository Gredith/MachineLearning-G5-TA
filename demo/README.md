# Demo interactiva

Aplicación web local hecha con [Streamlit](https://streamlit.io) para la presentación final. Muestra el mejor modelo del proyecto, **Gradient Boosting con pérdida de Huber**, prediciendo el adelanto o atraso de una vuelta en el punto de control 18 con la información disponible hasta el punto k, y resume los resultados del informe del E2. Está en español e inglés, con tema claro, oscuro o según el sistema.

## Cómo abrirla

Requisitos: Python 3.13 o 3.14, y `Times.csv` y `Monitoring.csv` en `data/raw/`. Todos los comandos se ejecutan desde la raíz del repositorio.

1. Instalar las dependencias (una sola vez):

   ```bash
   pip install -r requirements.txt
   ```

2. Preparar los datos y entrenar los modelos (una sola vez, menos de 1 min):

   ```bash
   python demo/entrenar_modelos.py
   ```

   Si `data/processed/` está vacío, el script construye los datos por corte desde `data/raw/` con las funciones de `src/` (no hace falta Jupyter). Luego reentrena los modelos y se detiene si el error no coincide con el del notebook 07 (2,62 / 2,59 / 2,58 / 2,40 / 2,00 min para k = 4, 6, 9, 12 y 15).

3. Abrir la demo (se abre en el navegador, en http://localhost:8501):

   ```bash
   cd demo
   python -m streamlit run app.py
   ```

   Para cerrarla: `Ctrl + C` en la terminal.

Se ejecuta desde `demo/` para que Streamlit lea los colores de `demo/.streamlit/config.toml`. Se usa `python -m streamlit` porque en Windows el comando `streamlit` a veces no está en el PATH.

## Qué tiene

Arriba a la izquierda está la configuración global: idioma (ES / EN) y tema (sistema, claro u oscuro). La esquina derecha queda libre para los botones que agrega Streamlit Community Cloud al desplegarla. La barra lateral, con la selección de la vuelta, aparece solo en la pestaña «Predecir una vuelta». El idioma, el tema y la pestaña abierta quedan en la dirección de la página, así que se puede compartir el enlace con la misma vista.

| Pestaña | Qué muestra |
|---|---|
| **Predecir una vuelta** | Se elige una vuelta de la semana de prueba (26 ene – 1 feb 2025, que el modelo no vio) y el punto de corte k. Muestra la predicción en lenguaje simple con su rango y una sugerencia operativa, el estado previsto frente al real, el error típico, avisos breves cuando la vuelta es difícil, la evolución del atraso en los 18 puntos, cómo cambia la predicción con cada k y qué grupo de variables pesó más |
| **Probar con otros datos** | Se sube un CSV con el formato de `Times.csv` (o se usa el archivo de ejemplo) y el modelo predice cada vuelta: resumen, tabla descargable y gráfico de predicción frente al valor real |
| **Resultados del proyecto** | Las cifras del informe, leídas de `results/tables/`: resumen, error según k, comparación de los 12 modelos con IC 95 %, importancia y ablación, clasificación, meta de 0,80 e hipótesis |
| **Acerca del modelo** | Limpieza de los datos (4.707 → 4.146 vueltas y el GPS) y por qué cada decisión, cómo funciona el modelo, cómo leer cada métrica y limitaciones |

Las ayudas (?) junto a las cifras explican cada métrica al pasar el cursor.

## Decisiones que conviene conocer

- **Mejor modelo de las pruebas.** `entrenar_modelos.py` reentrena Gradient Boosting con pérdida de Huber: los hiperparámetros de `results/tables/04_hiperparametros.csv`, el `alpha` de `results/tables/07_huber.csv` y el mismo entrenamiento (6 a 25 de enero). Requiere haber ejecutado el notebook 07. En la semana de prueba supera 0,80 en el porcentaje de vueltas con error ≤ 5 min en los cinco cortes (`results/tables/07_modelo_demo_meta80.csv`). La pestaña «Resultados del proyecto» muestra las cifras del informe, con la pérdida cuadrática.
- **Rango de la predicción.** Es la predicción ± el error medio (MAE) del modelo con ese k en la semana de prueba.
- **Explicación por grupos.** Cada barra reemplaza un grupo de variables por un valor típico y mide cuánto cambia la predicción: es la ablación del informe aplicada a una vuelta. Por eso un grupo puede mover una vuelta concreta aunque en promedio aporte poco (como el GPS).
- **Vueltas en curso.** En «Probar con otros datos», las vueltas sin punto 18 también se predicen; no entran en el cálculo del error.
- **Sin GPS en archivos subidos.** Esas variables se completan con valores típicos; en el informe el GPS no mejoraba el error (H4).
- **El archivo de ejemplo es la semana de prueba** en el formato original: datos que el modelo no vio al entrenar, pero no posteriores al 1 de febrero.
- **Tema.** Streamlit guarda el tema en el navegador; al cambiarlo desde la página, la demo actualiza esa preferencia y se recarga (se mantienen el idioma, el tema y la pestaña, no la vuelta elegida).
- **Datos fuera del repositorio.** `demo/modelos/` y `demo/datos/` están en `.gitignore` porque contienen datos del curso; cada uno los regenera con el paso 2.

## Limitaciones conocidas

- Las fuentes se cargan desde Google Fonts: sin internet se usan las del sistema.
- Algunos textos del recuadro para subir archivos («Upload», «200MB per file») son de Streamlit y salen en inglés.

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz de las cuatro pestañas, configuración de idioma y tema, gráficos; tipografía Atkinson Hyperlegible |
| `textos.py` | Todos los textos en español e inglés y el formato de números y fechas |
| `prediccion.py` | Lógica sin interfaz: cargar modelos, predecir, explicar y leer archivos |
| `resultados.py` | Lectura de las tablas de `results/tables/` |
| `entrenar_modelos.py` | Prepara los datos por corte si faltan, entrena y guarda los modelos y genera la semana de prueba y el archivo de ejemplo |
| `.streamlit/config.toml` | Colores del tema claro y oscuro |
