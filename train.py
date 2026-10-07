"""Este script predice la duración (días) de ensayos clínicos oncológicos entrenando un modelo de
bosque aleatorio.

Se realizó la migración a script del modelo base del notebook de Semana 3 (TC5061). Es importante señalar
que en este script se  sustituye el modelo de regresión lineal que realizamos la semana pasada por un
modelo de bosque aleatorio porque permite variar hiperparámetros que es parte del requerimiento de la
actividad.

El flujo del proceso es el siguiente: 1. Se cargan datos, 2. Se limpian, 3. Se dividen,
4. Se entrena, 5. Se evalúa. Cada etapa en su propia función conforme a lo solicitado.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- Constantes del proyecto ---
# La ruta se calcula a partir de la ubicación de este archivo,
# así el script funciona desde cualquier carpeta
RAIZ_PROYECTO = Path(__file__).parent
RUTA_DATOS = RAIZ_PROYECTO / "data" / "cancer_trials_2020_2024.csv"
OBJETIVO = "duracion_dias"
CATEGORICAS = ["study_type", "phase", "sponsor_type", "sex"]
TIPOS_INTERVENCION = ["DRUG", "OTHER", "PROCEDURE", "BEHAVIORAL", "BIOLOGICAL",
                      "DEVICE", "RADIATION", "DIAGNOSTIC_TEST"]
PATROCINADORES_MINORITARIOS = ["FED", "UNKNOWN", "INDIV"]
COLUMNAS_EXCLUIDAS = ["nct_id", "study_title", "study_status", "conditions", "interventions",
                      "lead_sponsor", "intervention_types", "start_date", "completion_date",
                      "enrollment"]

# Proporción del conjunto de prueba: fija para que todas las corridas
# se evalúen sobre los mismos registros y sean comparables
TEST_SIZE = 0.2

# --- MLflow: base de datos y artefactos dentro de la carpeta del proyecto ---
EXPERIMENTO = "TC5061_Semana4_BosqueAleatorio_DuracionEnsayos"
TRACKING_URI = f"sqlite:///{RAIZ_PROYECTO / 'mlflow.db'}"
RUTA_ARTEFACTOS = RAIZ_PROYECTO / "mlruns"

# Registro de mensajes: se muestran en pantalla y se guardan en un archivo por corrida
logger = logging.getLogger("train")


# ============================================================
# Argumentos de línea de comandos
# ============================================================

def parsear_argumentos():
    """Define los hiperparámetros y la semilla como argumentos, con valores por defecto.

    Los valores se validan aquí, antes de cargar datos o abrir una corrida en MLflow,
    para que un argumento inválido no deje corridas fallidas en el experimento.
    """
    parser = argparse.ArgumentParser(
        description="Entrena un bosque aleatorio que predice la duración (días) "
                    "de ensayos clínicos oncológicos.",
    )
    parser.add_argument("--n-estimators", type=_entero_minimo(1), default=200, metavar="N",
                        help="Número de árboles del bosque (por defecto: %(default)s)")
    parser.add_argument("--max-depth", type=_entero_minimo(0), default=10, metavar="N",
                        help="Profundidad máxima de cada árbol; 0 = sin límite (por defecto: %(default)s)")
    parser.add_argument("--min-samples-leaf", type=_entero_minimo(1), default=5, metavar="N",
                        help="Mínimo de registros en cada hoja del árbol (por defecto: %(default)s)")
    parser.add_argument("--semilla", type=_entero_minimo(0), default=42, metavar="N",
                        help="Semilla aleatoria para la división de datos y el bosque (por defecto: %(default)s)")
    return parser.parse_args()


def _entero_minimo(minimo):
    """Crea un validador para argparse que solo acepta enteros mayores o iguales a `minimo`."""
    def validar(texto):
        if not texto.isdigit() or int(texto) < minimo:
            raise argparse.ArgumentTypeError(
                f"se esperaba un entero mayor o igual a {minimo}; se recibió '{texto}'")
        return int(texto)
    return validar


def configurar_registro(ruta_archivo):
    """Envía los mensajes a la pantalla y a un archivo, con fecha, hora y nivel."""
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
    logger.handlers.clear()  # evita mensajes repetidos si se configura más de una vez
    logger.setLevel(logging.INFO)
    for manejador in (logging.StreamHandler(), logging.FileHandler(ruta_archivo, encoding="utf-8")):
        manejador.setFormatter(formato)
        logger.addHandler(manejador)


# ============================================================
# Datos
# ============================================================

def cargar_datos(ruta):
    """Lee el CSV original de ensayos clínicos."""
    return pd.read_csv(ruta)


def limpiar_datos(df_original):
    """Aplica las decisiones de limpieza documentadas en el notebook de Semana 3 (sección 3).

    Cada paso devuelve un DataFrame nuevo y no modifica el que recibe (sin efectos
    secundarios), así que el DataFrame original queda intacto.
    """
    df = _crear_variable_objetivo(df_original)
    df = _tratar_valores_atipicos(df)
    df = _tratar_valores_faltantes(df)
    df = _agrupar_categorias_raras(df)
    return df.drop(columns=COLUMNAS_EXCLUIDAS)


def _crear_variable_objetivo(df):
    """Calcula la duración en días entre inicio y fin (fechas AAAA-MM → día 1)."""
    inicio = pd.to_datetime(df["start_date"], format="mixed")
    fin = pd.to_datetime(df["completion_date"], format="mixed")
    return df.assign(**{OBJETIVO: (fin - inicio).dt.days})


def _tratar_valores_atipicos(df):
    """Elimina duraciones de 0 días y transforma los participantes con log10."""
    df_sin_cero = df[df[OBJETIVO] > 0]
    return df_sin_cero.assign(log10_participantes=np.log10(df_sin_cero["enrollment"]))


def _tratar_valores_faltantes(df):
    """Fase sin dato → 'No aplica'; tipo de intervención → una columna binaria por tipo.

    Los tipos poco frecuentes (no listados en TIPOS_INTERVENCION) se suman a OTHER, y los
    ensayos sin tipo de intervención se marcan en la columna interv_sin_dato.
    """
    fase = df["phase"].fillna("No aplica").replace("Not Specified", "No aplica")
    tipos_por_ensayo = (df["intervention_types"]
                        .str.replace(r"\s*;\s*", ";", regex=True)
                        .str.get_dummies(sep=";"))
    tipos_raros = tipos_por_ensayo.drop(columns=TIPOS_INTERVENCION).any(axis=1)

    columnas_binarias = {f"interv_{tipo.lower()}": tipos_por_ensayo[tipo]
                         for tipo in TIPOS_INTERVENCION}
    columnas_binarias["interv_other"] = (tipos_por_ensayo["OTHER"] | tipos_raros).astype(int)
    columnas_binarias["interv_sin_dato"] = df["intervention_types"].isna().astype(int)
    return df.assign(phase=fase, **columnas_binarias)


def _agrupar_categorias_raras(df):
    """Agrupa los patrocinadores poco frecuentes en la categoría MINORITARIOS."""
    return df.assign(sponsor_type=df["sponsor_type"].replace(PATROCINADORES_MINORITARIOS,
                                                             "MINORITARIOS"))


def dividir_datos(df, test_size, semilla):
    """Separa variables predictoras y objetivo, y las divide en entrenamiento y prueba.

    Devuelve un diccionario con nombres (no una tupla por posición), para que nadie
    confunda el orden de los conjuntos al usarlos.
    """
    X = df.drop(columns=OBJETIVO)
    y = df[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=semilla)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# ============================================================
# Modelos
# ============================================================

def construir_modelo(n_estimators, max_depth, min_samples_leaf, semilla):
    """Arma el flujo: codificación one-hot de las categóricas + bosque aleatorio."""
    preprocesamiento = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)],
        remainder="passthrough",  # log10_participantes e interv_* pasan sin cambios
    )
    bosque = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=semilla,
        n_jobs=-1,
    )
    return Pipeline([("preprocesamiento", preprocesamiento), ("bosque", bosque)])


def entrenar_referencia(X_train, y_train):
    """Modelo de referencia: siempre predice la mediana del entrenamiento."""
    return DummyRegressor(strategy="median").fit(X_train, y_train)


def evaluar(modelo, X, y):
    """Calcula MAE, RMSE y R² de un modelo sobre un conjunto de datos."""
    pred = modelo.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),       # error absoluto medio (días)
        "rmse": root_mean_squared_error(y, pred),  # raíz del error cuadrático medio (días)
        "r2": r2_score(y, pred),                   # proporción de la varianza explicada
    }


# ============================================================
# Registro en MLflow
# ============================================================

def configurar_mlflow():
    """Conecta con la base de datos de MLflow del proyecto y activa el experimento."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        # Los artefactos se guardan en la carpeta del proyecto, sin importar desde dónde se ejecute
        mlflow.create_experiment(EXPERIMENTO, artifact_location=RUTA_ARTEFACTOS.as_uri())
    mlflow.set_experiment(EXPERIMENTO)


def registrar_en_mlflow(parametros, resultados, modelo, particion):
    """Registra en la corrida activa: parámetros, métricas, artefactos y el modelo entrenado."""
    # 1) Hiperparámetros y configuración de la corrida
    for nombre, valor in parametros.items():
        mlflow.log_param(nombre, valor)

    # 2) Métricas por conjunto: prueba, entrenamiento (para detectar sobreajuste) y referencia
    for conjunto, metricas in resultados.items():
        for nombre, valor in metricas.items():
            mlflow.log_metric(f"{conjunto}_{nombre}", valor)

    # 3) Artefactos: archivo de métricas y gráfica real contra predicho
    with tempfile.TemporaryDirectory() as carpeta:
        ruta_metricas = Path(carpeta) / "metricas.json"
        ruta_metricas.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
        ruta_grafica = Path(carpeta) / "real_contra_predicho.png"
        _graficar_real_contra_predicho(modelo, particion["X_test"], particion["y_test"], ruta_grafica)
        mlflow.log_artifact(str(ruta_metricas), artifact_path="evaluacion")
        mlflow.log_artifact(str(ruta_grafica), artifact_path="evaluacion")

    # 4) Modelo: el Pipeline completo (codificación + bosque) con su firma de entrada y salida
    X_train = particion["X_train"]
    # Las columnas enteras se declaran como decimales en la firma para que el modelo
    # acepte valores faltantes al predecir (recomendación de MLflow)
    enteras_como_decimales = {col: "float64" for col in X_train.select_dtypes("integer").columns}
    firma = mlflow.models.infer_signature(X_train.astype(enteras_como_decimales),
                                          modelo.predict(X_train))
    # El formato seguro (skops) exige declarar como confiable la estructura de los árboles;
    # es confiable porque el modelo lo entrenamos nosotros en esta misma corrida
    mlflow.sklearn.log_model(modelo, name="modelo", signature=firma,
                             input_example=X_train.head(5),
                             skops_trusted_types=["sklearn.tree._tree.Tree"])


def _graficar_real_contra_predicho(modelo, X, y, ruta):
    """Guarda la gráfica de duración real contra predicha del conjunto de prueba."""
    pred = modelo.predict(X)
    fig = Figure(figsize=(7, 5.5))
    ax = fig.subplots()
    ax.scatter(y, pred, alpha=0.3, s=10)
    limite = max(y.max(), pred.max())
    ax.plot([0, limite], [0, limite], color="red", linestyle="--", label="Predicción perfecta")
    ax.set_xlabel("Duración real (días)")
    ax.set_ylabel("Duración predicha (días)")
    ax.set_title("Real contra predicho (conjunto de prueba)")
    ax.legend()
    fig.savefig(ruta, dpi=100, bbox_inches="tight")


# ============================================================
# Ejecución
# ============================================================

def entrenar_y_evaluar(parametros):
    """Carga y limpia los datos, entrena el bosque y la referencia, y registra todo en MLflow."""
    df = limpiar_datos(cargar_datos(RUTA_DATOS))
    particion = dividir_datos(df, parametros["test_size"], parametros["semilla"])
    logger.info("Registros: entrenamiento = %s | prueba = %s",
                f"{len(particion['X_train']):,}", f"{len(particion['X_test']):,}")

    modelo = construir_modelo(parametros["n_estimators"], parametros["max_depth"],
                              parametros["min_samples_leaf"], parametros["semilla"])
    modelo.fit(particion["X_train"], particion["y_train"])
    referencia = entrenar_referencia(particion["X_train"], particion["y_train"])

    resultados = {
        "prueba": evaluar(modelo, particion["X_test"], particion["y_test"]),
        "entrenamiento": evaluar(modelo, particion["X_train"], particion["y_train"]),
        "referencia": evaluar(referencia, particion["X_test"], particion["y_test"]),
    }
    for conjunto, metricas in resultados.items():
        logger.info("%-13s: MAE = %s | RMSE = %s | R² = %.3f", conjunto,
                    f"{metricas['mae']:,.1f}", f"{metricas['rmse']:,.1f}", metricas["r2"])

    registrar_en_mlflow(parametros, resultados, modelo, particion)
    logger.info("Parámetros, métricas, artefactos y modelo registrados en MLflow")


def main():
    args = parsear_argumentos()
    parametros = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth or None,  # 0 → None (árboles sin límite de profundidad)
        "min_samples_leaf": args.min_samples_leaf,
        "semilla": args.semilla,
        "test_size": TEST_SIZE,
    }
    nombre_corrida = (f"rf_n{parametros['n_estimators']}_prof{parametros['max_depth'] or 'libre'}"
                      f"_hoja{parametros['min_samples_leaf']}")

    configurar_mlflow()
    with mlflow.start_run(run_name=nombre_corrida) as corrida, \
            tempfile.TemporaryDirectory() as carpeta:
        ruta_log = Path(carpeta) / "entrenamiento.log"
        configurar_registro(ruta_log)
        logger.info("Corrida '%s' (run_id = %s) en el experimento '%s'",
                    nombre_corrida, corrida.info.run_id, EXPERIMENTO)
        logger.info("Configuración: %s", parametros)
        try:
            entrenar_y_evaluar(parametros)
        except Exception:
            # El error no se oculta: se registra con su traza completa y se vuelve a lanzar
            logger.exception("La corrida falló; MLflow la marcará como FAILED")
            raise
        finally:
            # El registro se sube a MLflow aunque la corrida falle, para poder diagnosticarla
            mlflow.log_artifact(str(ruta_log), artifact_path="registro")


if __name__ == "__main__":
    main()
