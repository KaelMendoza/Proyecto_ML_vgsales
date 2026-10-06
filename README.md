# Video Games Sales — Machine Learning

Proyecto de Introducción a Machine Learning para predecir las ventas globales de videojuegos utilizando información sobre sus características, críticas y datos de publicación.

Dante Tlacaelel Mendoza Alcantar 

GitUser: Kael Mendoza

Commits:

e902fc45d318438b9bc0901a20993109f0919054

2f455ce1f12a1f14b3d333f6c79f383a3179dc7f


Oliver O´Brian Rodríguez Pinedo 

GitUser: azfanatic

Commits:

04a679b3a5ba5142ed62f3a872dbb1ffebca6bcf

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

## Preguntas de reflexión 

¿Qué implementaste y qué decisión técnica tomaste para realizarla?

Dante: Implemente la base de datos y la selección de variables, además la decisión de los modelos de regresión a usar viendo que nos convenia mas, usamos ayuda de agentes de IA para codear.

Oliver: Implementé la función build_pipelines() en src/models.py y refactoricé train.py para entrenar y evaluar pipelines que integran el preprocesamiento con cada modelo, en lugar de transformar los datos manualmente con fit_transform y transform. La decisión técnica principal fue que cada pipeline recibe su propia instancia del preprocesador, para que los modelos no compartan estado y el imputador, el escalador y el codificador aprendan solo de los datos de entrenamiento al llamar a fit(). También conservé build_models() y build_preprocessor() sin cambios para no romper el notebook de marimo, que depende de ellas.


¿Cómo verificaste tu aportación?

Dante: Haciendo pruebas tanto de los .py en vscode conforme se iban codeando y también en marimo para auditar cada parte de mi proceso.

Oliver: Ejecuté python train.py en la rama main (código original) y en mi rama, en la misma máquina, y comparé las métricas. Fueron idénticas: Regresión Lineal en validación (MAE 0.5433, RMSE 2.1831, R² 0.1703) y Random Forest en validación (RMSE 1.9950) y en prueba (MAE 0.3729, RMSE 1.1463, R² 0.4769). Eso confirmó que el cambio reorganiza el código sin alterar los resultados. También noté que estas cifras difieren un poco de las del README (RMSE de prueba 1.1573) y lo atribuí a que requirements.txt no fija versiones y el Random Forest depende de la versión de scikit-learn.


¿Qué observaste o aprendiste al revisar el trabajo de otra persona?

Dante: Este equipo se formo en un día pues ambos no teniamos equipo,  asi que decidimos darnos libertad en el proceso para poder entregar a tiempo el trabajo.

Oliver: Al leer la base del proyecto de Kael vi que separar la configuración, la carga, la división, el preprocesamiento y la evaluación en módulos facilita mucho trabajar en paralelo y modificar una parte sin tocar las demás. También aprendí que un resultado reproducible depende no solo de la semilla, sino de las versiones de las librerías, y que revisar con cuidado revela detalles que no se notan al correr el código, como los nombres de juegos repetidos que pueden quedar en conjuntos distintos.


¿Qué mejorarías en la siguiente versión?

Dante: Me gustaría añadir mas modelos para tener mas puntos de comparación, además de fusionar otra base de datos, la actual cuenta con solo 16k datos, mientras que otra que tengo en cuenta para un futuro cuenta con mas de de 460k, siendo también una base mas actualizada que podría ayudar a mejorar los modelos implementados y a implementar.
Me gustaría mejorar la sincronización con mi compañero para dar un trabajo más aceptable y mejor estructurado.

Oliver: Fijaría las versiones de las dependencias en requirements.txt y actualizaría las cifras del README. Agregaría una línea base (un modelo que predice la mediana) para saber si los modelos aportan algo. Haría la división agrupando por Name, porque hay 5,156 filas con nombre repetido y una división aleatoria puede mezclar versiones del mismo juego entre entrenamiento y prueba. Además, probaría una transformación logarítmica del objetivo, porque Global_Sales tiene una cola muy larga que explica la diferencia entre el RMSE de validación (≈2.0) y el de prueba (≈1.15), y reduciría las 2,153 columnas del one-hot agrupando categorías raras de Publisher y Developer.
