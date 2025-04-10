import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(page_title="ScoreCredit - Prédiction de Solvabilité", page_icon="💳", layout="centered")

# Style CSS personnalisé
st.markdown("""
<style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive {
        background-color: #e6f7e6;
        border-left: 5px solid #2ecc71;
        color: #27ae60;
    }
    .negative {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
        color: #c0392b;
    }
    .medium-risk {
        background-color: #fff8e1;
        border-left: 5px solid #f39c12;
        color: #d35400;
    }
    .risk-value {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource
def load_assets():
    model = joblib.load('modele_logistique.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_assets()

# Fonction de conversion en notation scientifique
def scientific_notation(prob):
    if prob == 0:
        return "0"
    exponent = int(np.floor(np.log10(prob)))
    coefficient = prob / (10 ** exponent)
    return f"{coefficient:.1f}×10^{exponent}"

# Fonction de prédiction
def predict_statut(input_df):
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]
    return prediction, probability

# En-tête
st.markdown("## 💰 Évaluation de la solvabilité client")

# Navigation par onglets
tab_manual, tab_batch = st.tabs(["🔍 Évaluation individuelle", "📊 Analyse de fichier"])

with tab_manual:
    st.markdown("### Informations client")
    
    with st.form("client_form"):
        cols = st.columns(2)
        
        with cols[0]:
            age = st.slider("Âge du client", 18, 100, 35)
            marital = st.selectbox("Situation familiale", 
                                options=[(1, "Célibataire"), 
                                        (2, "Marié(e)"), 
                                        (3, "Divorcé(e)"), 
                                        (4, "Veuf/veve"), 
                                        (5, "Union libre")],
                                format_func=lambda x: x[1])
            expenses = st.number_input("Charges mensuelles (€)", 0, 5000, 1200)
        
        with cols[1]:
            income = st.number_input("Revenu mensuel (€)", 0, 20000, 2500)
            amount = st.number_input("Montant emprunté (€)", 0, 100000, 15000)
            price = st.number_input("Valeur du bien (€)", 0, 200000, 18000)
        
        submitted = st.form_submit_button("Calculer le score", type="primary")
    
    if submitted:
        input_data = pd.DataFrame([[age, marital[0], expenses, income, amount, price]],
                                columns=['Age', 'Marital', 'Expenses', 'Income', 'Amount', 'Price'])
        
        with st.spinner("Analyse en cours..."):
            _, proba = predict_statut(input_data)
            score = proba[0]  # <- extrait le scalaire

            sci_notation = scientific_notation(score)  # si tu veux l'utiliser ailleurs

            st.markdown("### Résultat d'analyse")
            
            if score > 0.7:
                st.markdown(f"""
                <div class="prediction-card negative">
                    <h3>Risque élevé ⚠️</h3>
                    <p class="risk-value">Probabilité de défaut: <strong>{score:.1%}</strong></p>
                    <p>Recommandation: <em>Refus de crédit recommandé</em></p>
                </div>
                """, unsafe_allow_html=True)
            elif score > 0.4:
                st.markdown(f"""
                <div class="prediction-card medium-risk">
                    <h3>Risque modéré 🔍</h3>
                    <p class="risk-value">Probabilité de défaut: <strong>{score:.1%}</strong></p>
                    <p>Recommandation: <em>Analyse complémentaire nécessaire</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card positive">
                    <h3>Faible risque ✅</h3>
                    <p class="risk-value">Probabilité de défaut: <strong>{score:.1%}</strong></p>
                    <p>Recommandation: <em>Acceptation du crédit</em></p>
                </div>
                """, unsafe_allow_html=True)

with tab_batch:
    st.markdown("### Analyse groupée")
    
    uploaded_file = st.file_uploader("Déposer votre fichier clients (CSV)", type="csv")
    
    if uploaded_file:
        try:
            clients = pd.read_csv(uploaded_file, sep=';')
            required_cols = ['Age', 'Marital', 'Expenses', 'Income', 'Amount', 'Price']
            
            if all(col in clients.columns for col in required_cols):
                st.success("Fichier validé avec succès !")
                
                if st.button("Lancer l'analyse complète", key="batch_analyze"):
                    with st.spinner("Traitement des dossiers..."):
                        predictions, probabilities = predict_statut(clients[required_cols])
                        clients['Statut'] = ['Risque' if p == 1 else 'Solvable' for p in predictions]
                        clients['Probabilité'] = probabilities
                        clients['Probabilité de ne pas être solvable'] = [scientific_notation(p) for p in probabilities]

                        st.markdown("### Synthèse des résultats")
                        st.metric("Clients à risque", f"{predictions.sum()} / {len(predictions)}")

                        # Graphique
                        risk_data = pd.DataFrame({
                            'Statut': ['Solvables', 'À risque'],
                            'Count': [len(predictions)-predictions.sum(), predictions.sum()]
                        })
                        st.bar_chart(risk_data.set_index('Statut'), use_container_width=True)

                        # Affichage des résultats
                        cols_to_show = ['Age', 'Marital', 'Expenses', 'Income', 'Amount', 'Price', 'Statut', 'Probabilité de ne pas être solvable']
                        st.dataframe(clients[cols_to_show], use_container_width=True)
                        
                        # Téléchargement
                        csv = clients.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Télécharger les résultats",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
            else:
                st.error(f"Colonnes manquantes. Requises: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Erreur de lecture du fichier: {str(e)}")

# Pied de page
st.markdown("---")
st.caption("ScoreCredit Analytics v1.0 - Outil de scoring financier")
