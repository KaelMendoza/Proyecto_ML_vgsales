from pathlib import Path


# Ruta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ruta del dataset
DATA_PATH = PROJECT_ROOT / "data" / "Video_Games_Sales_as_at_22_Dec_2016.csv"


# Variable objetivo
TARGET = "Global_Sales"


# Variables predictoras
FEATURES = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Critic_Score",
    "Critic_Count",
    "User_Score",
    "User_Count",
    "Developer",
    "Rating",
]


# Variables numéricas
NUMERIC_FEATURES = [
    "Year_of_Release",
    "Critic_Score",
    "Critic_Count",
    "User_Score",
    "User_Count",
]


# Variables categóricas
CATEGORICAL_FEATURES = [
    "Platform",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]


# Semilla para reproducibilidad
RANDOM_STATE = 42


# División de los datos
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10