# src/pengouins/__init__.py
__version__ = "0.1.0"

# Permet d'importer comme ça : from src.pengouins import data, model, registry
from .data import load_data, get_X_y, split_data, get_preprocessor
from .model import train_model, evaluate_model
from .registry import save_model, load_model