"""
Model registry for saving and loading models.
Supports both local storage and MLflow tracking.
"""

import pickle
import os
import loguru
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from .config import Config

logger = loguru.logger

if Config.use_mlflow():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)


def save_model(model: Any
               , model_name: Optional[str] 
               , filepath: Optional[str] = None
               , X: Optional[Any] = None
               ) -> None:
    """
    Save a trained model to disk or MLflow.
    
    Args:
        model: The trained model to save
        filepath: Path to save the model (used in local mode)
        model_name: Name for the model in MLflow (used in mlflow mode)
    """
    signature = infer_signature(X[:10], model.predict(X[:10])) if X is not None else None
    logger.info(signature)
    if Config.use_mlflow():
        
        print(f"💾 Logging model to MLflow as '{model_name}'...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
            signature=signature
        )
        print(f"✅ Model logged to MLflow successfully!")
        
    else:
        # Local mode: save with pickle
        if filepath is None:
            filepath = Config.get_model_path(model_name or "model")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving model locally to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Model saved successfully!")



def load_model(filepath: Optional[str] = None, model_name: Optional[str] = None, 
               run_id: Optional[str] = None, version: Optional[int] = None) -> Any:
    """
    Load a model from disk or MLflow.
    
    Args:
        filepath: Path to load the model from (used in local mode)
        model_name: Name of the registered model in MLflow
        run_id: Specific run ID to load from MLflow
        version: Specific version to load from MLflow model registry
        
    Returns:
        The loaded model
    """
    if Config.use_mlflow():
        # MLflow mode: load from MLflow
        # Load latest version from model registry
        # Si c'est un tag :
        #model_uri = f"models:/{model_name}/latest"
        # Si c'est un alias :
        model_uri = f"models:/{model_name}@champion"
        
        print(f"📥 Loading latest model '{model_name}' from MLflow...")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully from MLflow!")
        return model
        
    else:
        # Local mode: load with pickle
        if filepath is None:
            filepath = Config.get_model_path(model_name or "model")
        
        print(f"📥 Loading model from {filepath}...")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        return model
