# Front-end - Classification de Pingouins

## Description

Interface web pour interagir avec l'API de classification de pingouins.

## Démarrage du serveur front-end

### Méthode 1 : Script bash
```bash
./frontend/start_frontend.sh
```

### Méthode 2 : Commande Python directe
```bash
cd frontend
python3 server.py
```

Le serveur démarrera sur le port **3000** par défaut.

## Configuration du Port Forwarding dans VS Code/Cursor

Pour accéder au front-end depuis votre navigateur local via la VM :

1. **Ouvrir la palette de commandes** :
   - `Ctrl+Shift+P` (ou `Cmd+Shift+P` sur Mac)

2. **Chercher "Ports"** :
   - Tapez "Ports: Focus on Ports View" ou "Forward a Port"

3. **Ajouter un port forwarding** :
   - Cliquez sur "Forward a Port"
   - Entrez le port : `3000`
   - Sélectionnez "TCP"

4. **Ou utiliser le fichier de configuration** :
   Créez/modifiez `.vscode/settings.json` dans votre workspace :
   ```json
   {
     "remote.portsAttributes": {
       "3000": {
         "label": "Front-end Penguin Classification",
         "onAutoForward": "notify"
       }
     }
   }
   ```

5. **Accéder au front-end** :
   - Une fois le port forwarding configuré, VS Code/Cursor affichera une notification
   - Cliquez sur "Open in Browser" ou accédez à `http://localhost:3000` dans votre navigateur

## Configuration alternative : Port forwarding manuel

Si vous préférez configurer le port forwarding manuellement via SSH :

```bash
ssh -L 3000:localhost:3000 user@34.155.59.112
```

Puis accédez à `http://localhost:3000` dans votre navigateur.

## Notes

- Le serveur écoute sur `0.0.0.0:3000` pour accepter les connexions depuis l'extérieur
- Le front-end est configuré pour communiquer avec l'API déployée sur `https://penguin-949276358023.europe-west9.run.app`
- Vous pouvez modifier l'URL de l'API directement dans l'interface web si nécessaire

