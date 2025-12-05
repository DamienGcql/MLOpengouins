"""
API FastAPI pour la classification de pingouins
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import loguru

from pengouins.registry import load_model
from pengouins.data import preprocess_data

app = FastAPI(
    title="Penguin Classification API",
    description="API pour classifier les espèces de pingouins",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis le front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacer par les origines spécifiques
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = loguru.logger

# Modèles Pydantic pour la validation des données
class PenguinFeatures(BaseModel):
    """Modèle pour les caractéristiques d'un pingouin"""
    bill_length_mm: float = Field(..., description="Longueur du bec en mm", ge=0, le=100)
    bill_depth_mm: float = Field(..., description="Profondeur du bec en mm", ge=0, le=50)
    flipper_length_mm: float = Field(..., description="Longueur de l'aileron en mm", ge=0, le=300)
    body_mass_g: float = Field(..., description="Masse corporelle en grammes", ge=0, le=10000)
    sex: str = Field(..., description="Sexe du pingouin", pattern="^(Male|Female)$")

    class Config:
        json_schema_extra = {
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": "Male"
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    species: str
    probabilities: dict[str, float]
    confidence: float


# Variables globales pour charger les modèles une seule fois
_model = None
_preprocessor = None


def load_models():
    """Charge les modèles et le préprocesseur"""
    global _model, _preprocessor
    if _model is None or _preprocessor is None:
        try:
            logger.info("Chargement des modèles...")
            _model = load_model(model_name="logistic_reg")
            _preprocessor = load_model(model_name="preprocessor")
            logger.info("Modèles chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Impossible de charger les modèles: {str(e)}"
            )
    return _model, _preprocessor


@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage de l'API"""
    load_models()


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Bienvenue sur l'API de classification de pingouins",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Documentation interactive (Swagger UI)",
            "/redoc": "Documentation alternative (ReDoc)",
            "/health": "Vérification de l'état de l'API",
            "/predict": "Faire une prédiction (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Vérifie l'état de l'API et des modèles"""
    try:
        model, preprocessor = load_models()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PenguinFeatures):
    """
    Fait une prédiction sur l'espèce d'un pingouin
    
    Args:
        features: Caractéristiques du pingouin
        
    Returns:
        Prédiction avec l'espèce, les probabilités et la confiance
    """
    try:
        # Charger les modèles
        model, preprocessor = load_models()
        
        # Créer un DataFrame avec les caractéristiques
        data = pd.DataFrame([{
            "bill_length_mm": features.bill_length_mm,
            "bill_depth_mm": features.bill_depth_mm,
            "flipper_length_mm": features.flipper_length_mm,
            "body_mass_g": features.body_mass_g,
            "sex": features.sex
        }])
        
        # Préprocesser les données
        _, X_preproc = preprocess_data(data, preprocessor=preprocessor)
        
        # Faire la prédiction
        prediction = model.predict(X_preproc)[0]
        
        # Obtenir les probabilités
        probabilities = model.predict_proba(X_preproc)[0]
        class_names = model.classes_
        
        # Créer un dictionnaire de probabilités
        prob_dict = {
            str(class_name): float(prob) 
            for class_name, prob in zip(class_names, probabilities)
        }
        
        # Calculer la confiance (probabilité maximale)
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            species=str(prediction),
            probabilities=prob_dict,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)

