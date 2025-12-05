#!/bin/bash
# Script pour démarrer l'application Streamlit

cd "$(dirname "$0")/.."

echo "🚀 Démarrage de l'application Streamlit..."
echo "📍 L'application sera accessible sur http://localhost:8501"
echo "📍 Configurez le port forwarding pour le port 8501 dans VS Code/Cursor"
echo ""

streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

