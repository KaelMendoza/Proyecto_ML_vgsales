import pandas as pd

from src.config import DATA_PATH, TARGET, FEATURES


def load_data():
    """
    Carga el dataset desde la ruta configurada.
    """
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    """
    Realiza la limpieza básica necesaria antes de separar
    las variables predictoras y la variable objetivo.

    - Convierte User_Score a formato numérico.
    - Los valores 'tbd' se convierten en NaN.
    """

    df = df.copy()

    # Convertimos User_Score a numérico.
    # Los valores que no puedan convertirse, como 'tbd',
    # se convierten automáticamente en NaN.
    df["User_Score"] = pd.to_numeric(
        df["User_Score"],
        errors="coerce",
    )

    return df


def prepare_features_target(df):
    """
    Separa las variables predictoras (X) de la variable objetivo (y).

    También elimina las filas donde falta el valor del target.
    """

    # Eliminamos únicamente las filas donde falta el target.
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    return X, y