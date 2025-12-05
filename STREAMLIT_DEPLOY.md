# Déploiement sur Streamlit Cloud

## Prérequis

1. Un compte GitHub avec le code poussé sur un dépôt
2. Un compte Streamlit Cloud (gratuit) : https://streamlit.io/cloud

## Étapes de déploiement

### 1. Préparer le dépôt GitHub

Assurez-vous que votre dépôt contient :
- ✅ `frontend/app.py` - L'application Streamlit
- ✅ `requirements.txt` - Les dépendances Python
- ✅ `.streamlit/config.toml` - Configuration Streamlit (optionnel)

### 2. Se connecter à Streamlit Cloud

1. Allez sur https://share.streamlit.io/
2. Connectez-vous avec votre compte GitHub
3. Autorisez Streamlit Cloud à accéder à vos dépôts GitHub

### 3. Déployer l'application

1. Cliquez sur "New app"
2. Sélectionnez votre dépôt : `MLOpengouins`
3. Sélectionnez la branche : `main` (ou la branche de votre choix)
4. **Chemin du fichier principal** : `frontend/app.py`
5. **URL de l'application** : Laissez Streamlit générer automatiquement ou choisissez un nom personnalisé
6. Cliquez sur "Deploy"

### 4. Configuration (optionnel)

Si vous avez besoin de variables d'environnement ou de secrets :
1. Dans les paramètres de l'application sur Streamlit Cloud
2. Allez dans "Secrets"
3. Ajoutez vos variables d'environnement au format TOML :
   ```toml
   [api]
   url = "https://penguin-949276358023.europe-west9.run.app"
   ```

### 5. Accéder à l'application

Une fois déployée, votre application sera accessible à :
```
https://[votre-nom-app].streamlit.app
```

## Structure des fichiers pour Streamlit Cloud

```
MLOpengouins/
├── frontend/
│   └── app.py              # Application Streamlit principale
├── requirements.txt         # Dépendances Python
├── .streamlit/
│   └── config.toml         # Configuration Streamlit (optionnel)
└── README.md               # Documentation
```

## Notes importantes

- **Chemin du fichier** : Streamlit Cloud doit pointer vers `frontend/app.py`
- **Requirements.txt** : Doit être à la racine du dépôt
- **API Backend** : L'URL de l'API est codée en dur dans `app.py` mais peut être modifiée via la sidebar
- **Premier déploiement** : Peut prendre quelques minutes
- **Mises à jour** : Les commits sur la branche principale déclenchent automatiquement un redéploiement

## Dépannage

### L'application ne démarre pas
- Vérifiez les logs dans Streamlit Cloud
- Vérifiez que `requirements.txt` contient toutes les dépendances
- Vérifiez que le chemin vers `app.py` est correct

### Erreurs d'import
- Assurez-vous que toutes les dépendances sont dans `requirements.txt`
- Vérifiez que les chemins d'import sont corrects

### L'API ne répond pas
- Vérifiez que l'URL de l'API est accessible publiquement
- Vérifiez les logs de l'application pour les erreurs de connexion

