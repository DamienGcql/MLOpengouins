#!/usr/bin/env python3
"""
Script de vérification des dépendances pour l'application Streamlit
"""

import sys

def check_module(module_name, package_name=None):
    """Vérifie si un module est installé"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name} : Installé")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} : NON installé")
        return False

print("=" * 60)
print("VÉRIFICATION DES DÉPENDANCES")
print("=" * 60)

modules = [
    ("streamlit", "Streamlit"),
    ("requests", "Requests"),
    ("pandas", "Pandas"),
    ("plotly", "Plotly"),
    ("plotly.express", "Plotly Express"),
]

all_ok = True
for module, name in modules:
    if not check_module(module, name):
        all_ok = False

print("=" * 60)
if all_ok:
    print("✅ Toutes les dépendances sont installées")
    sys.exit(0)
else:
    print("❌ Certaines dépendances manquent")
    print("\nPour installer les dépendances, exécutez :")
    print("  pip install -r requirements.txt")
    sys.exit(1)

