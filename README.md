# Video Games Sales — Machine Learning

Proyecto de Introducción a Machine Learning para predecir las ventas globales de videojuegos utilizando información sobre sus características, críticas y datos de publicación.

## Objetivo

Construir y evaluar modelos de regresión capaces de predecir las ventas globales de videojuegos.

La variable objetivo es:

* `Global_Sales`

Se excluyen `NA_Sales`, `EU_Sales`, `JP_Sales` y `Other_Sales` para evitar data leakage, ya que estas variables representan componentes directos de las ventas globales.

## Dataset

Dataset utilizado:

`Video_Games_Sales_as_at_22_Dec_2016.csv`

El dataset contiene 16,719 registros y 16 variables.

### Variables utilizadas

**Numéricas:**

* `Year_of_Release`
* `Critic_Score`
* `Critic_Count`
* `User_Score`
* `User_Count`

**Categóricas:**

* `Platform`
* `Genre`
* `Publisher`
* `Developer`
* `Rating`

## Preprocesamiento

Las variables numéricas reciben:

* Imputación de valores faltantes mediante la mediana.
* Estandarización con `StandardScaler`.

Las variables categóricas reciben:

* Imputación mediante la categoría más frecuente.
* Codificación One-Hot mediante `OneHotEncoder`.

El preprocesamiento se ajusta únicamente utilizando el conjunto de entrenamiento para evitar data leakage.

## División de los datos

Los datos se dividen de la siguiente manera:

* 80% entrenamiento
* 10% validación
* 10% prueba

Se utiliza `random_state = 42` para garantizar reproducibilidad.

## Modelos

Se evaluaron dos modelos de regresión:

1. Regresión Lineal
2. Random Forest Regressor

Las métricas utilizadas son:

* MAE
* RMSE
* R²

El modelo se selecciona utilizando el RMSE obtenido sobre el conjunto de validación.

## Resultados

### Validación

| Modelo           |    MAE |   RMSE |     R² |
| ---------------- | -----: | -----: | -----: |
| Regresión Lineal | 0.5433 | 2.1831 | 0.1703 |
| Random Forest    | 0.3959 | 2.0116 | 0.2955 |

El modelo seleccionado fue **Random Forest**.

### Evaluación final

Sobre el conjunto de prueba:

| Modelo        |    MAE |   RMSE |     R² |
| ------------- | -----: | -----: | -----: |
| Random Forest | 0.3726 | 1.1573 | 0.4667 |

## Estructura del proyecto

```text
Proyecto_ML_vgsales/
│
├── data/
│   └── Video_Games_Sales_as_at_22_Dec_2016.csv
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   └── split.py
│
├── notebook.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

## Ejecución

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar el entrenamiento

```bash
python train.py
```

### Ejecutar el notebook de Marimo

Para editar y ejecutar el notebook:

```bash
marimo edit notebook.py
```

Para ejecutarlo como aplicación:

```bash
marimo run notebook.py
```

## Reproducibilidad

El proyecto utiliza una semilla fija (`random_state = 42`) para obtener resultados reproducibles.
