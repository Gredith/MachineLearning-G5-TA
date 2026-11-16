# Convenciones de trabajo

Reglas comunes para que el equipo trabaje en paralelo sin pisarse y con resultados reproducibles. Si algo no está aquí, se decide en el grupo y se agrega.

## 1. Flujo de trabajo

- `main` siempre está estable: todo lo que hay ahí corre de punta a punta.
- Nadie trabaja directo en `main`. Cada tarea de [TAREAS.md](TAREAS.md) va en su propia rama.
- Ciclo: rama → commits → pull request → revisión de al menos otro integrante → merge con *squash* → borrar la rama.
- Antes de crear una rama, actualizar `main` (`git pull`).

## 2. Ramas

Formato `tipo/descripcion-corta`, en minúsculas y con guiones.

| Tipo | Uso | Ejemplo |
|---|---|---|
| `feat/` | Código o notebook nuevo | `feat/lineas-base` |
| `data/` | Preprocesamiento y construcción de variables | `data/variables-por-corte` |
| `exp/` | Experimentos y ablaciones | `exp/ablacion-gps` |
| `fix/` | Corrección de errores | `fix/rutas-notebook-01` |
| `docs/` | Documentación, paper y slides | `docs/borrador-e2` |

Ramas de corta vida (2 días como máximo): mejor varias ramas pequeñas que una grande.

## 3. Commits

- Español, en imperativo, 72 caracteres como máximo, con el prefijo del tipo de rama:
  `feat: agrega líneas base B0 a B3`, `fix: corrige rutas relativas del notebook 01`.
- Un solo propósito por commit; si el mensaje necesita "y", probablemente son dos commits.
- Nunca se commitean datos (`*.csv`), credenciales, tokens ni salidas pesadas. Revisar `git status` antes de cada commit.

## 4. Pull requests

- Título en el mismo formato que el commit.
- Descripción con tres puntos: qué cambia, cómo se verificó y qué tarea de TAREAS.md cierra.
- Checklist antes de pedir revisión:
  - [ ] El notebook corre con *Restart & Run All* sin errores
  - [ ] No se suben CSV ni archivos pesados
  - [ ] Rutas relativas, semilla 42
  - [ ] Cifras o decisiones nuevas anotadas en CONTEXTO.md y tarea marcada en TAREAS.md
- Quien revisa comprueba que no haya fuga de información (nada calculado con los días de prueba).

## 5. Notebooks

- **Nombre:** `NN_tema.ipynb` con dos dígitos según el orden de ejecución (`02_preprocesamiento_variables_por_corte.ipynb`). No crear copias `_v3` o `_final`: la versión nueva reemplaza a la anterior y el historial queda en git.
- **Un responsable por notebook.** Los `.ipynb` no se fusionan bien; si dos personas necesitan el mismo, se divide en dos notebooks o el código común pasa a `src/`.
- **Estructura:**
  1. Título, objetivo y entradas/salidas en una celda Markdown.
  2. Celda de configuración: importaciones, `SEMILLA = 42`, rutas relativas (`../data/raw/`, `../data/processed/`, `../results/`).
  3. Secciones numeradas con una breve explicación en Markdown antes del código.
  4. Conclusiones o cifras clave al final.
- **Antes de commitear:** *Restart & Run All*. Se dejan salidas ligeras (tablas cortas, gráficos); se limpian impresiones largas de DataFrames o logs.
- Las rutas de Colab o Drive van comentadas, nunca activas por defecto. Sin rutas absolutas.
- Si una función se usa en dos notebooks, se mueve a `src/` con un docstring breve y se importa.

## 6. Datos y reproducibilidad

- Los CSV originales van solo en `data/raw/` y no se versionan. Lo derivado va en `data/processed/` y se regenera con los notebooks.
- Semilla 42 en todo lo aleatorio (`random_state=SEMILLA`).
- Partición temporal fija ([CONTEXTO.md §4](CONTEXTO.md#4-diseño-experimental)). Imputación, escalado y codificación se ajustan solo con entrenamiento; el conjunto de prueba se evalúa una sola vez.
- Una librería nueva se agrega a `requirements.txt` en el mismo PR.
- Los notebooks de experimentos terminan con una celda que imprime las versiones de Python y de las librerías.

## 7. Resultados y documentación

- Figuras en `results/figures/` y tablas en `results/tables/`, con el prefijo del notebook que las genera: `04_mae_vs_k.png`, `04_hiperparametros.csv`.
- Cada decisión metodológica nueva se anota en el [registro de decisiones](CONTEXTO.md#5-registro-de-decisiones) con fecha y motivo.
- Cada tarea terminada se marca en [TAREAS.md](TAREAS.md) en el mismo PR.

## 8. Entregables

- Van en `docs/entregables/E1`, `E2` y `final`, con estos nombres: `Entregable1_G5.pdf`, `Entregable2_G5.pdf`, `Paper_final_G5.pdf`, `Slides_G5.pdf`.
- Se guarda la versión enviada a la plataforma; los borradores se trabajan en Overleaf o Drive.
- Al entregar, se marca el commit con un tag: `git tag e2` o `git tag final`, y `git push --tags`.

## 9. Comandos de referencia

```bash
git switch main
git pull
git switch -c feat/lineas-base
```

```bash
git add notebooks/03_lineas_base.ipynb
git commit -m "feat: agrega líneas base B0 a B3"
git push -u origin feat/lineas-base
```

Luego se abre el pull request en GitHub y, tras el merge, se borra la rama:

```bash
git switch main
git pull
git branch -d feat/lineas-base
```
