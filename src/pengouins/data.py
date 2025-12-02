"""
Module de chargement et pré-traitement des données pingouins.
"""

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pengouins.registry import load_model, save_model


def load_data(filename: str = "pingouins.csv") -> pd.DataFrame:
    """
    Charge le jeu de données des pingouins.
    - Si le fichier existe déjà localement → lecture directe
    - Sinon → téléchargement via seaborn, suppression de la colonne 'island' (fuite de cible),
      puis sauvegarde dans data/
    """
    filepath = Path("data") / filename

    if filepath.exists():
        logger.info(f"Lecture du fichier existant : {filepath}")
        return pd.read_csv(filepath)

    logger.info("Dataset non trouvé localement → téléchargement depuis seaborn...")
    data = sns.load_dataset("penguins")
    data.drop(columns=["island"], inplace=True)

    os.makedirs("data", exist_ok=True)
    data.to_csv(filepath, index=False)
    logger.success(f"Dataset sauvegardé dans {filepath}")

    return data


def get_X_y(dataframe: pd.DataFrame, target_col: str = "species") -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features (X) de la cible (y).
    Utilise .pop() comme demandé dans le notebook.
    """
    if target_col not in dataframe.columns:
        raise ValueError(f"La colonne cible '{target_col}' est absente du DataFrame.")

    y = dataframe.pop(target_col)   # Retire la colonne et la retourne
    X = dataframe.copy()            # On travaille sur une copie propre

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Découpage train/test avec stratification automatique
    (pour conserver la répartition des espèces dans les deux jeux).
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def preprocess_data(X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Pré-traite les données :
    - Valeurs manquantes → imputation (médiane ou mode)
    - Variables numériques → centrage/réduction
    - Variables catégorielles → One-Hot Encoding (drop first)

    Le pré-processeur est sauvegardé via registry lors du fit,
    et rechargé automatiquement en mode inférence.
    """
    if fit:
        logger.info("Construction et entraînement du pré-processeur...")

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, drop="first"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, make_column_selector(dtype_include="number")),
            ("cat", categorical_pipeline, make_column_selector(dtype_include="object"))
        ])

        X_processed = preprocessor.fit_transform(X)
        save_model(preprocessor, "preprocessor")
        logger.success("Pré-processeur entraîné et sauvegardé.")

    else:
        logger.info("Mode inférence → chargement du pré-processeur existant...")
        preprocessor = load_model("preprocessor")
        X_processed = preprocessor.transform(X)

    # Retour sous forme de DataFrame lisible (optionnel mais très pratique)
    if fit:
        feature_names = (
            preprocessor.named_transformers_["num"]["scaler"].get_feature_names_out().tolist() +
            preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out().tolist()
        )
    else:
        feature_names = (
            preprocessor.named_transformers_["num"].named_steps["scaler"].get_feature_names_out().tolist() +
            preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out().tolist()
        )

    return pd.DataFrame(X_processed, columns=feature_names, index=X.index)