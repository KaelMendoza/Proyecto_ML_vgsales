from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(y_true, y_pred):
    """
    Calcula las métricas de evaluación para un modelo
    de regresión.

    Retorna:
    - MAE
    - RMSE
    - R²
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }