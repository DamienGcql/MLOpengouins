"""
Load and preprocess data.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split


def load_data(path: str = "data/pingouins.csv") -> pd.DataFrame:

    import os
    import pandas as pd
    import seaborn as sns

# On crée le bon dossier
    os.makedirs("data", exist_ok=True)

    if os.path.exists(path):
        pingouins = pd.read_csv(path)
        print(f"Chargé depuis {path}")
    else:
        print("Première fois → téléchargement depuis seaborn...")
        pingouins = sns.load_dataset("penguins")
        
        # On supprime la colonne island
        pingouins.drop(columns=["island"], inplace=True)
        
        # On sauvegarde pour les prochaines fois
        pingouins.to_csv(path, index=False)
        pingouins = pd.read_csv(path)
        print(f"Sauvegardé dans {path}")

    return pingouins


def get_X_y(
    df: pd.DataFrame, target_column: str, target:bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    pass

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    pass

def preprocess_data(X: pd.DataFrame
                    ,fit = True) -> pd.DataFrame:
    """Preprocess data: handle missing values, encode categorical variables, scale numerical features."""
    pass