from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


def split_data(X, y):
    """
    Divide los datos en entrenamiento, validación y prueba.

    80% entrenamiento
    10% validación
    10% prueba
    """

    # Primero separamos entrenamiento del conjunto temporal
    temp_size = TEST_SIZE + VALIDATION_SIZE

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=RANDOM_STATE,
    )

    # Del conjunto temporal obtenemos la proporción correspondiente
    # para que el resultado final sea:
    # 80% entrenamiento
    # 10% validación
    # 10% prueba
    test_ratio = TEST_SIZE / temp_size

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )