from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_linear_regression():
    """
    Construye el modelo de Regresión Lineal.
    """
    return LinearRegression()


def build_random_forest():
    """
    Construye el modelo Random Forest para regresión.
    """
    return RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_models():
    """
    Construye todos los modelos que serán evaluados (sin preprocesamiento).
    """
    models = {
        "Linear Regression": build_linear_regression(),
        "Random Forest": build_random_forest(),
    }

    return models


def build_pipelines():
    """
    Construye un Pipeline por modelo: preprocesamiento + modelo.

    Cada pipeline recibe su propia instancia del preprocesador, de modo
    que al llamar a fit() con los datos de entrenamiento, el imputador,
    el escalador y el codificador aprenden únicamente de ese conjunto.
    Luego, predict() aplica esas mismas transformaciones a validación
    y prueba sin volver a ajustarlas.
    """
    pipelines = {}

    for model_name, model in build_models().items():
        pipelines[model_name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

    return pipelines