import pickle
from pathlib import Path
import os
import loguru

logger = loguru.logger


def save_model(model, filepath):
    """Saves the model to the specified filepath using pickle."""
    if not os.path.exists("models") : os.mkdir("models")
    model_path = Path("models",filepath)
    logger.info(f"✅ Saving {model.__class__.__name__} to {model_path}")
    with open(model_path,"wb" ) as f : 
        pickle.dump(model,f) 

def load_model(filepath : str):
    """Loads the model from the specified filepath using pickle."""
    model_path = Path("models",filepath)
    logger.info(f"✅ Loading model from {model_path}")
    with open(model_path,"rb") as f : 
        model = pickle.load(f)
    return model