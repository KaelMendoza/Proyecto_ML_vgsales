from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import RANDOM_STATE


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
    Construye todos los modelos que serán evaluados.
    """
    models = {
        "Linear Regression": build_linear_regression(),
        "Random Forest": build_random_forest(),
    }

    return models