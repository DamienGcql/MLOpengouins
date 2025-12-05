#!/usr/bin/env python3
"""
Serveur HTTP simple pour servir le front-end
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 3000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Ajouter les en-têtes CORS pour permettre les requêtes vers l'API
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        # Si on accède à la racine, servir index.html
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

if __name__ == "__main__":
    # Changer vers le répertoire du script
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 Serveur front-end démarré sur le port {PORT}")
        print(f"📍 Accédez à : http://localhost:{PORT}")
        print(f"📍 Ou depuis l'extérieur : http://0.0.0.0:{PORT}")
        print("\n⚠️  Configurez le port forwarding dans VS Code/Cursor :")
        print(f"   - Port local : {PORT}")
        print(f"   - Port distant : {PORT}")
        print(f"   - Host : 0.0.0.0")
        print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt du serveur...")
            httpd.shutdown()

