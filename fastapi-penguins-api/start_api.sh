#!/bin/bash
# Script pour démarrer l'API FastAPI

cd "$(dirname "$0")/.."

echo "🚀 Démarrage de l'API FastAPI..."
echo "📍 Assurez-vous d'avoir entraîné le modèle avant de démarrer l'API"
echo ""

python -m uvicorn fastapi-penguins-api.main:app --reload --host 0.0.0.0 --port 8888

