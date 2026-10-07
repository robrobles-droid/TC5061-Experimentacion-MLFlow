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
from pathlib import Path
import numpy as np
import pandas as pd
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
RUTA_DATOS = Path(__file__).parent / "data" / "cancer_trials_2020_2024.csv"
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


# ============================================================
# Argumentos de línea de comandos
# ============================================================

def parsear_argumentos():
    """Define los hiperparámetros y la semilla como argumentos, con valores por defecto."""
    parser = argparse.ArgumentParser(
        description="Entrena un bosque aleatorio que predice la duración (días) "
                    "de ensayos clínicos oncológicos.",
    )
    parser.add_argument("--n-estimators", type=int, default=200, metavar="N",
                        help="Número de árboles del bosque (por defecto: %(default)s)")
    parser.add_argument("--max-depth", type=int, default=10, metavar="N",
                        help="Profundidad máxima de cada árbol; 0 = sin límite (por defecto: %(default)s)")
    parser.add_argument("--min-samples-leaf", type=int, default=5, metavar="N",
                        help="Mínimo de registros en cada hoja del árbol (por defecto: %(default)s)")
    parser.add_argument("--semilla", type=int, default=42, metavar="N",
                        help="Semilla aleatoria para la división de datos y el bosque (por defecto: %(default)s)")
    return parser.parse_args()


# ============================================================
# Datos
# ============================================================

def cargar_datos(ruta):
    """Lee el CSV original de ensayos clínicos."""
    return pd.read_csv(ruta)


def limpiar_datos(df_original):
    """Aplica las decisiones de limpieza documentadas en el script (sección 3).

    No modifica el DataFrame original: trabaja sobre una copia.
    """
    df = df_original.copy()
    df = _crear_variable_objetivo(df)
    df = _tratar_valores_atipicos(df)
    df = _tratar_valores_faltantes(df)
    df = _agrupar_categorias_raras(df)
    return df.drop(columns=COLUMNAS_EXCLUIDAS)


def _crear_variable_objetivo(df):
    """Convierte las fechas y calcula la duración en días (fechas AAAA-MM → día 1)."""
    df["start_date"] = pd.to_datetime(df["start_date"], format="mixed")
    df["completion_date"] = pd.to_datetime(df["completion_date"], format="mixed")
    df[OBJETIVO] = (df["completion_date"] - df["start_date"]).dt.days
    return df


def _tratar_valores_atipicos(df):
    """Elimina duraciones de 0 días y transforma los participantes con log10."""
    df = df[df[OBJETIVO] > 0].copy()
    df["log10_participantes"] = np.log10(df["enrollment"])
    return df


def _tratar_valores_faltantes(df):
    """Fase sin dato → 'No aplica'; tipo de intervención → columnas binarias por tipo."""
    df["phase"] = df["phase"].fillna("No aplica").replace("Not Specified", "No aplica")

    tipos = df["intervention_types"].str.split(";").apply(
        lambda lista: {t.strip() for t in lista} if isinstance(lista, list) else set())
    for tipo in TIPOS_INTERVENCION:
        df[f"interv_{tipo.lower()}"] = tipos.apply(lambda s: int(tipo in s))
    # Tipos poco frecuentes (no listados) se suman a OTHER
    raros = tipos.apply(lambda s: int(bool(s - set(TIPOS_INTERVENCION))))
    df["interv_other"] = df["interv_other"] | raros
    df["interv_sin_dato"] = df["intervention_types"].isna().astype(int)
    return df


def _agrupar_categorias_raras(df):
    """Agrupa los patrocinadores poco frecuentes en la categoría MINORITARIOS."""
    df["sponsor_type"] = df["sponsor_type"].replace(PATROCINADORES_MINORITARIOS, "MINORITARIOS")
    return df


def dividir_datos(df, test_size, semilla):
    """Separa variables predictoras y objetivo, y las divide en entrenamiento y prueba."""
    X = df.drop(columns=OBJETIVO)
    y = df[OBJETIVO]
    return train_test_split(X, y, test_size=test_size, random_state=semilla)


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
# Ejecución
# ============================================================

def main():
    args = parsear_argumentos()
    max_depth = args.max_depth or None  # 0 → None (árboles sin límite de profundidad)
    print(f"Configuración: n_estimators = {args.n_estimators} | max_depth = {max_depth} | "
          f"min_samples_leaf = {args.min_samples_leaf} | semilla = {args.semilla}")

    df = limpiar_datos(cargar_datos(RUTA_DATOS))
    X_train, X_test, y_train, y_test = dividir_datos(df, TEST_SIZE, args.semilla)
    print(f"Registros: entrenamiento = {len(X_train):,} | prueba = {len(X_test):,}")

    modelo = construir_modelo(args.n_estimators, max_depth, args.min_samples_leaf, args.semilla)
    modelo.fit(X_train, y_train)
    referencia = entrenar_referencia(X_train, y_train)

    metricas = evaluar(modelo, X_test, y_test)
    metricas_ref = evaluar(referencia, X_test, y_test)
    print(f"Bosque aleatorio (prueba): MAE = {metricas['mae']:,.1f} | "
          f"RMSE = {metricas['rmse']:,.1f} | R² = {metricas['r2']:.3f}")
    print(f"Referencia (mediana)     : MAE = {metricas_ref['mae']:,.1f} | "
          f"RMSE = {metricas_ref['rmse']:,.1f} | R² = {metricas_ref['r2']:.3f}")


if __name__ == "__main__":
    main()
