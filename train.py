from src.data import (
    load_data,
    clean_data,
    prepare_features_target,
)

from src.split import split_data
from src.models import build_pipelines
from src.evaluation import calculate_metrics


def train_and_evaluate():
    """
    Entrena y evalúa los modelos de Machine Learning.

    Flujo:
    1. Carga y limpieza de datos.
    2. Separación de variables predictoras y objetivo.
    3. División en entrenamiento, validación y prueba.
    4. Construcción de pipelines (preprocesamiento + modelo).
    5. Entrenamiento y evaluación sobre validación.
    6. Selección del mejor modelo.
    7. Evaluación final sobre test.
    """

    # ============================================================
    # 1. CARGA Y PREPARACIÓN DE LOS DATOS
    # ============================================================

    print("=" * 60)
    print("CARGA Y PREPARACIÓN DE LOS DATOS")
    print("=" * 60)

    df = load_data()
    df = clean_data(df)

    X, y = prepare_features_target(df)

    print(f"Dataset: {df.shape}")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    # ============================================================
    # 2. DIVISIÓN DE LOS DATOS
    # ============================================================

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    print("\nDivisión de los datos:")
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Validación:    {X_validation.shape}")
    print(f"Prueba:        {X_test.shape}")

    # ============================================================
    # 3. CONSTRUCCIÓN DE LOS PIPELINES
    # ============================================================

    # Cada pipeline integra el preprocesamiento y el modelo.
    # Al hacer fit() solo con X_train, el preprocesamiento se ajusta
    # únicamente con datos de entrenamiento. Al hacer predict(), las
    # transformaciones aprendidas se aplican a datos nuevos sin reajustarse.
    pipelines = build_pipelines()

    # Diccionario para guardar los resultados de validación.
    validation_results = {}

    # ============================================================
    # 4. ENTRENAMIENTO Y EVALUACIÓN EN VALIDACIÓN
    # ============================================================

    print("\n" + "=" * 60)
    print("RESULTADOS DE VALIDACIÓN")
    print("=" * 60)

    for model_name, pipeline in pipelines.items():

        print(f"\nEntrenando: {model_name}")

        # Entrenamiento únicamente con los datos de entrenamiento
        # (incluye el ajuste del preprocesamiento).
        pipeline.fit(X_train, y_train)

        # Predicciones sobre validación (datos sin transformar).
        y_validation_pred = pipeline.predict(X_validation)

        metrics = calculate_metrics(
            y_validation,
            y_validation_pred,
        )

        validation_results[model_name] = metrics

        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R²:   {metrics['R2']:.4f}")

    # ============================================================
    # 5. SELECCIÓN DEL MEJOR MODELO
    # ============================================================

    # Para regresión:
    # - menor MAE = mejor
    # - menor RMSE = mejor
    # - mayor R² = mejor
    #
    # Utilizamos RMSE como criterio principal.

    best_model_name = min(
        validation_results,
        key=lambda name: validation_results[name]["RMSE"],
    )

    best_pipeline = pipelines[best_model_name]

    print("\n" + "=" * 60)
    print("MEJOR MODELO")
    print("=" * 60)

    print(f"Modelo seleccionado: {best_model_name}")
    print(
        f"RMSE de validación: "
        f"{validation_results[best_model_name]['RMSE']:.4f}"
    )

    # ============================================================
    # 6. EVALUACIÓN FINAL SOBRE TEST
    # ============================================================

    print("\n" + "=" * 60)
    print("EVALUACIÓN FINAL SOBRE TEST")
    print("=" * 60)

    # El conjunto de test NO se utilizó para seleccionar
    # el modelo. Solamente se utiliza aquí para obtener
    # la evaluación final.

    y_test_pred = best_pipeline.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
    )

    print(f"Modelo: {best_model_name}")
    print(f"MAE:    {test_metrics['MAE']:.4f}")
    print(f"RMSE:   {test_metrics['RMSE']:.4f}")
    print(f"R²:     {test_metrics['R2']:.4f}")

    # ============================================================
    # 7. RETORNAR RESULTADOS
    # ============================================================

    return {
        "pipelines": pipelines,
        "validation_results": validation_results,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "test_metrics": test_metrics,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }


if __name__ == "__main__":
    train_and_evaluate()