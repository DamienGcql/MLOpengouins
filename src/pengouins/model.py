"""
Fonctions d'entraînement et d'évaluation du modèle.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from loguru import logger
from pengouins.registry import load_model, save_model

# Configuration du logger
logger = logger


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Entraîne un modèle de régression logistique sur les données d'entraînement
    et sauvegarde automatiquement le modèle via le registry.
    """
    logi = LogisticRegression()
    logi.fit(X_train, y_train)
    
    save_model(logi, "logistic_regression_model.pkl")
    logger.info("Modèle entraîné et sauvegardé avec succès.")
    
    return logi


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Évalue le modèle sur les données de test et retourne le score d'exactitude (accuracy).
    Le score est arrondi à 4 décimales pour plus de lisibilité.
    """
    y_pred = model.predict(X_test)
    score = round(accuracy_score(y_test, y_pred), 4)
    
    logger.info(f"Précision du modèle sur le jeu de test : {score}")
    
    return score