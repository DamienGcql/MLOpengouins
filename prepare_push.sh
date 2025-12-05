#!/bin/bash
# Script pour préparer le push vers GitHub

echo "🔍 Vérification des fichiers à ajouter..."
echo ""

# Afficher le statut git
git status

echo ""
echo "📋 Fichiers qui seront ajoutés :"
echo "  - frontend/ (application Streamlit)"
echo "  - fastapi-penguins-api/ (API FastAPI)"
echo "  - .streamlit/config.toml (config Streamlit)"
echo "  - requirements.txt (dépendances)"
echo "  - STREAMLIT_DEPLOY.md (guide de déploiement)"
echo "  - README.md (mis à jour)"
echo ""

read -p "Voulez-vous continuer et ajouter ces fichiers ? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "📦 Ajout des fichiers..."
    git add frontend/
    git add fastapi-penguins-api/
    git add .streamlit/
    git add requirements.txt
    git add STREAMLIT_DEPLOY.md
    git add README.md
    git add .gitignore
    
    echo ""
    echo "✅ Fichiers ajoutés !"
    echo ""
    echo "📝 Pour commiter, exécutez :"
    echo "   git commit -m 'Add Streamlit frontend and FastAPI backend'"
    echo ""
    echo "🚀 Pour pousser vers GitHub, exécutez :"
    echo "   git push origin main"
else
    echo "❌ Annulé"
fi

