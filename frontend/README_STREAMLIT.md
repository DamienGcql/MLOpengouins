# Application Streamlit - Classification de Pingouins

## Description

Application web interactive créée avec Streamlit pour classifier les espèces de pingouins à partir de leurs caractéristiques physiques.

## Prérequis

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. S'assurer que le backend API est accessible (déployé ou local)

## Démarrage de l'application

### Méthode 1 : Script bash
```bash
./frontend/start_streamlit.sh
```

### Méthode 2 : Commande Streamlit directe
```bash
cd /home/damien.gicquel.35/code/MLOpengouins
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

L'application sera accessible sur le port **8501**.

## Configuration du Port Forwarding dans VS Code/Cursor

Pour accéder à l'application depuis votre navigateur local via la VM :

1. **Port forwarding automatique** :
   - VS Code/Cursor détecte automatiquement le port 8501 grâce à `.vscode/settings.json`
   - Une notification apparaît : "Port 8501 is being forwarded"
   - Cliquez sur "Open in Browser" ou accédez à `http://localhost:8501`

2. **Port forwarding manuel** :
   - Ouvrez la palette de commandes : `Ctrl+Shift+P` (ou `Cmd+Shift+P`)
   - Tapez "Forward a Port"
   - Entrez le port : `8501`
   - VS Code/Cursor créera le tunnel automatiquement

3. **Accéder à l'application** :
   - Une fois le port forwarding actif, ouvrez dans votre navigateur :
   ```
   http://localhost:8501
   ```

## Fonctionnalités

- 📝 **Formulaire interactif** : Saisissez les caractéristiques du pingouin
- 🔮 **Prédiction en temps réel** : Obtenez une prédiction instantanée
- 📊 **Visualisations** : Graphiques en barres des probabilités par espèce
- 🎯 **Métriques** : Affichage de la confiance de la prédiction
- 🔍 **Test de connexion** : Vérifiez la connexion à l'API depuis l'interface
- ⚡ **Exemples rapides** : Boutons pour charger rapidement des exemples

## Configuration de l'API

L'URL de l'API peut être modifiée dans la barre latérale (sidebar) de l'application.

Par défaut, l'application utilise : `https://penguin-949276358023.europe-west9.run.app`

## Partage de l'application

Streamlit permet de partager facilement l'application :

1. **Streamlit Cloud** (recommandé) :
   - Connectez votre dépôt GitHub
   - Déployez automatiquement sur Streamlit Cloud
   - URL publique gratuite

2. **Autres options** :
   - Docker
   - Serveur dédié
   - Cloud providers (AWS, GCP, Azure)

## Notes

- Le serveur écoute sur `0.0.0.0:8501` pour accepter les connexions depuis l'extérieur
- L'application communique avec l'API backend via HTTP/HTTPS
- Les données sont envoyées en JSON lors des prédictions

