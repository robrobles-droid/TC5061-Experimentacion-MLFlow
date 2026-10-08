# TC5061 · Código de experimentación con MLflow

Entrenamiento de un **bosque aleatorio** que predice la **duración (en días) de ensayos clínicos oncológicos**, con cada corrida registrada y comparable en **MLflow Tracking**.

Actividad individual de la Semana 4 del curso TC5061.10 *Operaciones de Aprendizaje Automático* (Tecnológico de Monterrey). Autor: Roberto Robles.

## Contenido del repositorio

```
├── train.py              ← script de entrenamiento parametrizado y registrado en MLflow
├── requirements.txt      ← dependencias con versiones fijas
├── data/
│   └── cancer_trials_2020_2024.csv
├── notebooks/
│   └── 0.01-rr-exploracion-datos.ipynb   ← solo exploración de datos
└── .gitignore
```

El nombre del notebook sigue la convención de nombres de las lecturas del curso (`fase.número-iniciales-descripción`; la fase `0` es exploración).

El archivo de `data/` se incluye sin modificaciones, con la licencia [Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) de su publicación en Kaggle (ver Referencias); sus datos de origen provienen de ClinicalTrials.gov. La limpieza la realiza `train.py`.

## Requisitos

- Python 3.13 (probado con 3.13.9)
- Git

## Instalación

```bash
git clone https://github.com/robrobles-droid/TC5061-Experimentacion-MLFlow.git
cd TC5061-Experimentacion-MLFlow
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Con los valores por defecto:

```bash
python train.py
```

Con hiperparámetros explícitos (comando de la corrida base):

```bash
python train.py --n-estimators 200 --max-depth 10 --min-samples-leaf 5 --semilla 42
```

Ayuda con la descripción de cada argumento:

```bash
python train.py --help
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--n-estimators` | 200 | Número de árboles del bosque |
| `--max-depth` | 10 | Profundidad máxima de cada árbol; `0` = sin límite |
| `--min-samples-leaf` | 5 | Mínimo de registros en cada hoja del árbol |
| `--semilla` | 42 | Semilla aleatoria para la división de datos y el bosque |

Los valores inválidos (por ejemplo, `--n-estimators 0`) se rechazan antes de cargar los datos.

### Corridas del experimento

Se cambia un hiperparámetro a la vez respecto de la corrida base:

```bash
python train.py --n-estimators 200 --max-depth 10 --min-samples-leaf 5 --semilla 42
python train.py --n-estimators 50  --max-depth 10 --min-samples-leaf 5 --semilla 42
python train.py --n-estimators 500 --max-depth 10 --min-samples-leaf 5 --semilla 42
python train.py --n-estimators 200 --max-depth 5  --min-samples-leaf 5 --semilla 42
python train.py --n-estimators 200 --max-depth 0  --min-samples-leaf 5 --semilla 42
python train.py --n-estimators 200 --max-depth 10 --min-samples-leaf 1 --semilla 42
python train.py --n-estimators 200 --max-depth 10 --min-samples-leaf 20 --semilla 42
```

## Ver y comparar las corridas en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir <http://127.0.0.1:5000>, elegir la vista **Model training** y el experimento **`TC5061_Semana4_BosqueAleatorio_DuracionEnsayos`**. Para comparar, seleccionar las corridas y pulsar **Compare**.

Cada corrida registra:

- **Parámetros:** `n_estimators`, `max_depth`, `min_samples_leaf`, `semilla`, `test_size`.
- **Métricas:** MAE, RMSE y R² en prueba, en entrenamiento y del modelo de referencia (predice la mediana).
- **Artefactos:** `evaluacion/metricas.json`, `evaluacion/real_contra_predicho.png` y `registro/entrenamiento.log`.
- **Modelo:** el flujo completo (codificación one-hot + bosque aleatorio) con `mlflow.sklearn.log_model`, su firma y un ejemplo de entrada.

La base de datos de MLflow (`mlflow.db`) y los artefactos (`mlruns/`) se generan localmente al ejecutar `train.py` y están excluidos del repositorio (`.gitignore`): quien ejecute el proyecto obtiene su propio registro de corridas.

## Reproducibilidad

- **Semilla fija** (`--semilla`) en la división de datos y en el bosque: dos ejecuciones con los mismos argumentos producen las mismas métricas.
- **Mismo conjunto de prueba en todas las corridas:** la división 80 % entrenamiento / 20 % prueba depende de la semilla; con `--semilla 42` todas las corridas se evalúan sobre los mismos 1,999 ensayos, de modo que las diferencias entre métricas se deben solo a los hiperparámetros.
- **Versiones fijas** en `requirements.txt`.
- Cada corrida registra el ***commit* de Git** del código que la produjo.

## Referencias

Acar, A. H. (2026). *Global oncology clinical trials (2020–2024)* (Versión 1) [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/abdurrahmanhakanacar/global-oncology-clinical-trials-2020-2024

Anthropic. (2026). *Claude Code* (Modelo Claude Opus 5.5) [Asistente de programación basado en un modelo de lenguaje de gran tamaño], utilizado para generación y depuración de código, revisión de buenas prácticas y apoyo en la redacción de la documentación, revisados y validados por el autor. https://claude.com/claude-code
