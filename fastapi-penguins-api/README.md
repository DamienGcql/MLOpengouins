# API FastAPI - Classification de Pingouins

## Description

API REST pour classifier les espèces de pingouins à partir de leurs caractéristiques physiques.

## Prérequis

1. Avoir entraîné le modèle au préalable :
   ```bash
   cd /home/damien.gicquel.35/code/MLOpengouins
   python main.py
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Démarrage de l'API

### Méthode 1 : Script bash
```bash
./fastapi-penguins-api/start_api.sh
```

### Méthode 2 : Commande Python directe
```bash
cd /home/damien.gicquel.35/code/MLOpengouins
python -m uvicorn fastapi-penguins-api.main:app --reload --host 0.0.0.0 --port 8888
```

L'API sera accessible sur : `http://localhost:8888`

## Endpoints disponibles

- `GET /` : Page d'accueil avec informations sur l'API
- `GET /health` : Vérification de l'état de l'API et des modèles
- `POST /predict` : Faire une prédiction sur l'espèce d'un pingouin
- `GET /docs` : Documentation interactive (Swagger UI)
- `GET /redoc` : Documentation alternative (ReDoc)

## Utilisation du front-end

1. Ouvrir le fichier `frontend/index.html` dans un navigateur web
2. Remplir le formulaire avec les caractéristiques du pingouin
3. Cliquer sur "Prédire l'espèce"
4. Le résultat s'affichera avec :
   - L'espèce prédite
   - Les probabilités pour chaque espèce
   - Le niveau de confiance de la prédiction

## Exemple de requête API

```bash
curl -X POST "http://localhost:8888/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "Male"
  }'
```

## Réponse exemple

```json
{
  "species": "Adelie",
  "probabilities": {
    "Adelie": 0.95,
    "Chinstrap": 0.03,
    "Gentoo": 0.02
  },
  "confidence": 0.95
}
```

