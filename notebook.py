import marimo

app = marimo.App()


@app.cell
def _():
    from src.data import (
        load_data,
        clean_data,
        prepare_features_target,
    )
    from src.split import split_data
    from src.preprocessing import build_preprocessor
    from src.models import build_models
    from src.evaluation import calculate_metrics

    return (
        load_data,
        clean_data,
        prepare_features_target,
        split_data,
        build_preprocessor,
        build_models,
        calculate_metrics,
    )


@app.cell
def _(load_data, clean_data, prepare_features_target):
    # ============================================================
    # 1. CARGA Y PREPARACIÓN
    # ============================================================

    print("=" * 60)
    print("1. CARGA Y PREPARACIÓN DE LOS DATOS")
    print("=" * 60)

    df = load_data()

    print(f"Dataset original: {df.shape}")

    df = clean_data(df)

    X, y = prepare_features_target(df)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    return X, y


@app.cell
def _(X, y, split_data):
    # ============================================================
    # 2. DIVISIÓN
    # ============================================================

    print("\n" + "=" * 60)
    print("2. DIVISIÓN DE LOS DATOS")
    print("=" * 60)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


@app.cell
def _(
    X_train,
    X_validation,
    X_test,
    build_preprocessor,
):
    # ============================================================
    # 3. PREPROCESSING
    # ============================================================

    print("\n" + "=" * 60)
    print("3. PREPROCESSING")
    print("=" * 60)

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        f"X_train_processed: "
        f"{X_train_processed.shape}"
    )

    print(
        f"X_validation_processed: "
        f"{X_validation_processed.shape}"
    )

    print(
        f"X_test_processed: "
        f"{X_test_processed.shape}"
    )

    return (
        X_train_processed,
        X_validation_processed,
        X_test_processed,
    )


@app.cell
def _(
    X_train_processed,
    X_validation_processed,
    X_test_processed,
):
    # ============================================================
    # 4. COMPROBACIÓN
    # ============================================================

    import numpy as np

    print("\n" + "=" * 60)
    print("4. COMPROBACIÓN DEL PREPROCESSING")
    print("=" * 60)

    print(
        "NaN en entrenamiento:",
        np.isnan(X_train_processed).sum(),
    )

    print(
        "NaN en validación:",
        np.isnan(X_validation_processed).sum(),
    )

    print(
        "NaN en test:",
        np.isnan(X_test_processed).sum(),
    )

    return


@app.cell
def _(
    build_models,
    calculate_metrics,
    X_train_processed,
    X_validation_processed,
    y_train,
    y_validation,
):
    # ============================================================
    # 5. ENTRENAMIENTO Y VALIDACIÓN
    # ============================================================

    print("\n" + "=" * 60)
    print("5. ENTRENAMIENTO Y VALIDACIÓN")
    print("=" * 60)

    models = build_models()

    validation_results = {}

    for model_name, model in models.items():

        print(f"\nEntrenando: {model_name}")

        model.fit(
            X_train_processed,
            y_train,
        )

        y_pred = model.predict(
            X_validation_processed
        )

        metrics = calculate_metrics(
            y_validation,
            y_pred,
        )

        validation_results[model_name] = metrics

        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R²:   {metrics['R2']:.4f}")

    return models, validation_results


@app.cell
def _(models, validation_results):
    # ============================================================
    # 6. MEJOR MODELO
    # ============================================================

    print("\n" + "=" * 60)
    print("6. SELECCIÓN DEL MEJOR MODELO")
    print("=" * 60)

    best_model_name = min(
        validation_results,
        key=lambda name: validation_results[name]["RMSE"],
    )

    best_model = models[best_model_name]

    print(
        f"Mejor modelo: {best_model_name}"
    )

    print(
        f"RMSE: "
        f"{validation_results[best_model_name]['RMSE']:.4f}"
    )

    return best_model, best_model_name


@app.cell
def _(
    best_model,
    best_model_name,
    calculate_metrics,
    X_test_processed,
    y_test,
):
    # ============================================================
    # 7. TEST FINAL
    # ============================================================

    print("\n" + "=" * 60)
    print("7. EVALUACIÓN FINAL SOBRE TEST")
    print("=" * 60)

    y_test_pred = best_model.predict(
        X_test_processed
    )

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
    )

    print(f"Modelo: {best_model_name}")
    print(f"MAE:    {test_metrics['MAE']:.4f}")
    print(f"RMSE:   {test_metrics['RMSE']:.4f}")
    print(f"R²:     {test_metrics['R2']:.4f}")

    return test_metrics


if __name__ == "__main__":
    app.run()