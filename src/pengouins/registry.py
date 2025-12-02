"""
Sauvegarde et chargement des modèles avec pickle.
"""

import os
import pickle
from pathlib import Path

from loguru import logger

# Configuration du logger
logger = logger


def save_model(model, filename: str) -> None:
    """
    Sauvegarde un modèle entraîné dans le dossier 'models' au format pickle.
    
    Args:
        model: Le modèle entraîné (préprocesseur, classifieur, etc.)
        filename: Nom du fichier (ex: "logistic_regression_model.pkl")
    """
    filepath = Path("models") / filename
    
    # Création du dossier models s'il n'existe pas
    filepath.parent.mkdir(exist_ok=True)
    
    logger.info(f"Sauvegarde du modèle → {filepath}")
    
    with open(filepath, "wb") as file:
        pickle.dump(model, file)
        
    logger.success(f"Modèle sauvegardé avec succès : {filename}")


def load_model(filename: str):
    """
    Charge un modèle précédemment sauvegardé depuis le dossier 'models'.
    
    Args:
        filename: Nom du fichier à charger (ex: "preprocessor.pkl")
    
    Returns:
        Le modèle chargé et prêt à l'emploi
    """
    filepath = Path("models") / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modèle introuvable : {filepath}")
    
    logger.info(f"Chargement du modèle depuis → {filepath}")
    
    with open(filepath, "rb") as file:
        model = pickle.load(file)
        
    logger.success(f"Modèle chargé avec succès : {filename}")
    
    return model