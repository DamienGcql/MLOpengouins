# Guide de dépannage - Application Streamlit

## Problème : Impossible de se connecter à l'interface web

### 1. Vérifier l'installation des dépendances

Exécutez le script de vérification :
```bash
python3 frontend/check_dependencies.py
```

Si Streamlit n'est pas installé :
```bash
pip install -r requirements.txt
```

### 2. Vérifier que Streamlit démarre correctement

Testez le démarrage :
```bash
cd /home/damien.gicquel.35/code/MLOpengouins
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

Vous devriez voir :
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://0.0.0.0:8501
```

### 3. Vérifier le port forwarding dans VS Code/Cursor

1. **Vérifier que le port 8501 est forwardé** :
   - Ouvrez la vue "Ports" dans VS Code/Cursor (onglet en bas)
   - Le port 8501 devrait apparaître avec le label "Streamlit - Penguin Classification"

2. **Si le port n'est pas forwardé automatiquement** :
   - Cliquez sur "Forward a Port" dans la vue Ports
   - Entrez `8501`
   - Sélectionnez "TCP"

3. **Accéder à l'application** :
   - Cliquez sur le globe ou "Open in Browser" à côté du port 8501
   - Ou accédez manuellement à `http://localhost:8501`

### 4. Vérifier les erreurs dans la console

Si Streamlit démarre mais que vous ne pouvez pas accéder :

1. **Vérifier les logs Streamlit** dans le terminal où vous avez lancé l'application
2. **Vérifier les erreurs dans la console du navigateur** (F12)
3. **Vérifier que le firewall n'bloque pas le port 8501**

### 5. Problèmes courants

#### Erreur : "Address already in use"
Le port 8501 est déjà utilisé. Solutions :
```bash
# Trouver le processus qui utilise le port
lsof -i :8501

# Ou utiliser un autre port
streamlit run frontend/app.py --server.port 8502 --server.address 0.0.0.0
```

#### Erreur : "ModuleNotFoundError: No module named 'streamlit'"
Installez Streamlit :
```bash
pip install streamlit
# ou
pip install -r requirements.txt
```

#### L'application démarre mais ne charge pas
- Vérifiez que l'API backend est accessible
- Vérifiez les erreurs dans la console du navigateur (F12)
- Testez l'API manuellement :
  ```bash
  curl https://penguin-949276358023.europe-west9.run.app/
  ```

### 6. Test de connexion manuel

Testez si vous pouvez accéder au serveur depuis la VM :
```bash
curl http://localhost:8501
```

Si cela fonctionne, le problème est probablement dans le port forwarding.

### 7. Alternative : Utiliser le serveur HTTP simple

Si Streamlit pose problème, vous pouvez utiliser le serveur HTTP simple :
```bash
python3 frontend/server.py
```
Puis configurez le port forwarding pour le port 3000.

## Commandes utiles

```bash
# Vérifier les dépendances
python3 frontend/check_dependencies.py

# Démarrer Streamlit
./frontend/start_streamlit.sh

# Vérifier les processus sur le port 8501
lsof -i :8501

# Tester l'API
curl https://penguin-949276358023.europe-west9.run.app/
```

