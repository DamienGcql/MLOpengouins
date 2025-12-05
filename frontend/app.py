"""
Application Streamlit pour la classification de pingouins
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from typing import Dict, Optional

# Configuration de la page
st.set_page_config(
    page_title="Classification de Pingouins",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API par défaut
DEFAULT_API_URL = "https://penguin-949276358023.europe-west9.run.app"

# Titre principal
st.title("🐧 Classification de Pingouins")
st.markdown("---")
st.markdown("Prédisez l'espèce d'un pingouin à partir de ses caractéristiques physiques")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "URL de l'API",
        value=DEFAULT_API_URL,
        help="URL du backend API pour les prédictions"
    )
    
    st.markdown("---")
    st.header("ℹ️ Informations")
    st.info(
        "Remplissez le formulaire avec les caractéristiques du pingouin "
        "pour obtenir une prédiction de son espèce."
    )
    
    # Bouton pour tester la connexion API
    if st.button("🔍 Tester la connexion API"):
        try:
            # Tester d'abord l'endpoint racine
            response = requests.get(f"{api_url}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connectée avec succès!")
                root_data = response.json()
                st.json(root_data)
                
                # Essayer aussi /health si disponible
                try:
                    health_response = requests.get(f"{api_url}/health", timeout=5)
                    if health_response.status_code == 200:
                        st.success("✅ Endpoint /health disponible")
                        health_data = health_response.json()
                        st.json(health_data)
                    else:
                        st.info("ℹ️ Endpoint /health non disponible (normal pour cette API)")
                except:
                    pass
            else:
                st.warning(f"⚠️ API répond avec le code {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur de connexion: {str(e)}")

# Formulaire principal
st.header("📝 Caractéristiques du pingouin")

# Utiliser des colonnes pour un meilleur layout
col1, col2 = st.columns(2)

# Initialiser les valeurs par défaut depuis session_state si disponibles
default_bill_length = st.session_state.get("bill_length", 39.1)
default_bill_depth = st.session_state.get("bill_depth", 18.7)
default_flipper_length = st.session_state.get("flipper_length", 181.0)
default_body_mass = st.session_state.get("body_mass", 3750.0)
default_sex_index = 0 if st.session_state.get("sex", "Male") == "Male" else 1

with col1:
    bill_length = st.number_input(
        "Longueur du bec (mm)",
        min_value=0.0,
        max_value=100.0,
        value=default_bill_length,
        step=0.1,
        help="Valeur typique : 35-50 mm",
        key="bill_length_input"
    )
    
    bill_depth = st.number_input(
        "Profondeur du bec (mm)",
        min_value=0.0,
        max_value=50.0,
        value=default_bill_depth,
        step=0.1,
        help="Valeur typique : 15-20 mm",
        key="bill_depth_input"
    )
    
    flipper_length = st.number_input(
        "Longueur de l'aileron (mm)",
        min_value=0.0,
        max_value=300.0,
        value=default_flipper_length,
        step=0.1,
        help="Valeur typique : 170-230 mm",
        key="flipper_length_input"
    )

with col2:
    body_mass = st.number_input(
        "Masse corporelle (g)",
        min_value=0.0,
        max_value=10000.0,
        value=default_body_mass,
        step=1.0,
        help="Valeur typique : 3000-6000 g",
        key="body_mass_input"
    )
    
    sex = st.selectbox(
        "Sexe",
        options=["Male", "Female"],
        index=default_sex_index,
        key="sex_input"
    )

st.markdown("---")

# Boutons d'action
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    predict_button = st.button("🔮 Prédire l'espèce", type="primary", use_container_width=True)

with col_btn2:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.rerun()

# Exemples rapides
with col_btn3:
    st.markdown("**Exemples rapides :**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    examples = {
        "Adelie": {
            "bill_length": 39.1,
            "bill_depth": 18.7,
            "flipper_length": 181.0,
            "body_mass": 3750.0,
            "sex": "Male"
        },
        "Chinstrap": {
            "bill_length": 46.5,
            "bill_depth": 17.9,
            "flipper_length": 192.0,
            "body_mass": 3500.0,
            "sex": "Female"
        },
        "Gentoo": {
            "bill_length": 46.1,
            "bill_depth": 13.2,
            "flipper_length": 211.0,
            "body_mass": 4500.0,
            "sex": "Female"
        }
    }
    
    with example_col1:
        if st.button("Adelie", use_container_width=True, key="ex_adelie"):
            st.session_state.bill_length = examples["Adelie"]["bill_length"]
            st.session_state.bill_depth = examples["Adelie"]["bill_depth"]
            st.session_state.flipper_length = examples["Adelie"]["flipper_length"]
            st.session_state.body_mass = examples["Adelie"]["body_mass"]
            st.session_state.sex = examples["Adelie"]["sex"]
            st.rerun()
    
    with example_col2:
        if st.button("Chinstrap", use_container_width=True, key="ex_chinstrap"):
            st.session_state.bill_length = examples["Chinstrap"]["bill_length"]
            st.session_state.bill_depth = examples["Chinstrap"]["bill_depth"]
            st.session_state.flipper_length = examples["Chinstrap"]["flipper_length"]
            st.session_state.body_mass = examples["Chinstrap"]["body_mass"]
            st.session_state.sex = examples["Chinstrap"]["sex"]
            st.rerun()
    
    with example_col3:
        if st.button("Gentoo", use_container_width=True, key="ex_gentoo"):
            st.session_state.bill_length = examples["Gentoo"]["bill_length"]
            st.session_state.bill_depth = examples["Gentoo"]["bill_depth"]
            st.session_state.flipper_length = examples["Gentoo"]["flipper_length"]
            st.session_state.body_mass = examples["Gentoo"]["body_mass"]
            st.session_state.sex = examples["Gentoo"]["sex"]
            st.rerun()

# Traitement de la prédiction
if predict_button:
    # Préparer les données
    data = {
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass,
        "sex": sex
    }
    
    # Afficher un spinner pendant la requête
    with st.spinner("🔮 Prédiction en cours..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Afficher les résultats
                st.markdown("---")
                st.header("📊 Résultats de la prédiction")
                
                # Espèce prédite avec style
                predicted_species = result.get("species", "Inconnu")
                
                # Vérifier si l'API retourne les probabilités et la confiance
                has_probabilities = "probabilities" in result
                has_confidence = "confidence" in result
                
                # Afficher dans une métrique
                if has_confidence:
                    confidence = result["confidence"]
                    col_pred1, col_pred2 = st.columns(2)
                    
                    with col_pred1:
                        st.metric(
                            label="🎯 Espèce prédite",
                            value=predicted_species,
                            delta=f"Confiance: {confidence*100:.1f}%"
                        )
                    
                    with col_pred2:
                        st.metric(
                            label="📈 Confiance",
                            value=f"{confidence*100:.1f}%"
                        )
                else:
                    # Version simplifiée si pas de confiance
                    st.metric(
                        label="🎯 Espèce prédite",
                        value=predicted_species
                    )
                    st.info("ℹ️ L'API déployée retourne uniquement l'espèce prédite sans probabilités détaillées.")
                
                # Graphique des probabilités (seulement si disponibles)
                if has_probabilities:
                    st.subheader("📊 Probabilités par espèce")
                    
                    # Préparer les données pour le graphique
                    prob_df = pd.DataFrame([
                        {"Espèce": species, "Probabilité": prob * 100}
                        for species, prob in result["probabilities"].items()
                    ])
                    
                    # Trier par probabilité décroissante
                    prob_df = prob_df.sort_values("Probabilité", ascending=False)
                    
                    # Créer un graphique en barres
                    fig = px.bar(
                        prob_df,
                        x="Espèce",
                        y="Probabilité",
                        color="Probabilité",
                        color_continuous_scale="Blues",
                        text="Probabilité",
                        labels={"Probabilité": "Probabilité (%)", "Espèce": "Espèce"},
                        title="Probabilités de prédiction par espèce"
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher les probabilités sous forme de tableau
                    st.subheader("📋 Détails des probabilités")
                    prob_df_display = prob_df.copy()
                    prob_df_display["Probabilité"] = prob_df_display["Probabilité"].apply(lambda x: f"{x:.2f}%")
                    prob_df_display.columns = ["Espèce", "Probabilité (%)"]
                    st.dataframe(prob_df_display, use_container_width=True, hide_index=True)
                
                # Informations sur les caractéristiques saisies
                with st.expander("📝 Caractéristiques utilisées"):
                    st.json(data)
                
            else:
                error_data = response.json() if response.content else {}
                st.error(f"❌ Erreur HTTP {response.status_code}")
                if "detail" in error_data:
                    st.error(f"Détails: {error_data['detail']}")
                    
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erreur de connexion à l'API: {str(e)}")
            st.info("💡 Vérifiez que l'URL de l'API est correcte et que le serveur est accessible.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Application de classification de pingouins - MLOps"
    "</div>",
    unsafe_allow_html=True
)

