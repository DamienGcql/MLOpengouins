# MLOps Week - Classification de Pingouins

## 🐧 Application de Classification de Pingouins

Application MLOps complète pour classifier les espèces de pingouins à partir de leurs caractéristiques physiques.

## 🚀 Déploiement sur Streamlit Cloud

L'application est disponible en ligne via Streamlit Cloud. Consultez [STREAMLIT_DEPLOY.md](./STREAMLIT_DEPLOY.md) pour les instructions de déploiement.

## 📁 Structure du Projet

```
MLOpengouins/
├── frontend/              # Application Streamlit
│   ├── app.py            # Application principale
│   └── ...
├── fastapi-penguins-api/  # API FastAPI
│   └── main.py           # Endpoints API
├── src/pengouins/        # Modules ML
│   ├── data.py          # Chargement et preprocessing
│   ├── model.py         # Entraînement du modèle
│   └── registry.py      # Gestion des modèles
└── requirements.txt      # Dépendances Python
```

## 🛠️ Getting Started

### Clone the Repository

```bash
cd ~/code
git clone git@github.com:vivadata/MLOpengouins.git
cd MLOpengouins
```

### Setup Project

```bash
make setup
pip install -r requirements.txt
```

### Lancer l'application Streamlit localement

```bash
streamlit run frontend/app.py
```

### Lancer l'API FastAPI localement

```bash
./fastapi-penguins-api/start_api.sh
```

## 📚 Documentation

- [Guide de déploiement Streamlit](./STREAMLIT_DEPLOY.md)
- [Documentation API](./fastapi-penguins-api/README.md)
- [Troubleshooting](./frontend/TROUBLESHOOTING.md)
- [Guidelines MLOps](./docs/02_Experiment_Tracking_Guidelines.md)

## 🔗 Liens

- **API Backend** : https://penguin-949276358023.europe-west9.run.app
- **Documentation API** : https://penguin-949276358023.europe-west9.run.app/docs

## 📝 Next Steps

For detailed guidelines and project instructions, please read [Guidelines.md](./docs/02_Experiment_Tracking_Guidelines.md).
