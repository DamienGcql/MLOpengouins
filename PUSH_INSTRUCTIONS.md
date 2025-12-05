# Instructions pour pousser le code sur GitHub

## 📋 Étape 1 : Ajouter les fichiers

```bash
cd /home/damien.gicquel.35/code/MLOpengouins

# Ajouter les nouveaux fichiers
git add frontend/
git add fastapi-penguins-api/
git add .streamlit/
git add requirements.txt
git add STREAMLIT_DEPLOY.md
git add README.md
git add .gitignore
git add packages.txt
```

## 📝 Étape 2 : Vérifier les fichiers ajoutés

```bash
git status
```

Assurez-vous que `mlflow.db` n'est PAS dans la liste (il doit être ignoré).

## 💾 Étape 3 : Créer un commit

```bash
git commit -m "Add Streamlit frontend and FastAPI backend for penguin classification"
```

Ou avec un message plus détaillé :

```bash
git commit -m "Add Streamlit frontend and FastAPI backend

- Add Streamlit application (frontend/app.py)
- Add FastAPI backend (fastapi-penguins-api/main.py)
- Add Streamlit Cloud configuration
- Update requirements.txt with Streamlit and FastAPI dependencies
- Add deployment documentation"
```

## 🚀 Étape 4 : Pousser vers GitHub

```bash
git push origin main
```

## ✅ Vérification

Après le push, vérifiez sur GitHub que tous les fichiers sont présents :
- `frontend/app.py`
- `fastapi-penguins-api/main.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `STREAMLIT_DEPLOY.md`

## 🔗 Étape 5 : Connecter à Streamlit Cloud

1. Allez sur https://share.streamlit.io/
2. Connectez-vous avec votre compte GitHub
3. Cliquez sur "New app"
4. Sélectionnez le dépôt : `DamienGcql/MLOpengouins`
5. Sélectionnez la branche : `main`
6. **Chemin du fichier principal** : `frontend/app.py`
7. Cliquez sur "Deploy"

Votre application sera disponible à : `https://[nom-app].streamlit.app`

## 📚 Documentation

Consultez [STREAMLIT_DEPLOY.md](./STREAMLIT_DEPLOY.md) pour plus de détails sur le déploiement.

