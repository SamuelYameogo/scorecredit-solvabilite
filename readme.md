# 💳 ScoreCredit - Prédiction de Solvabilité Bancaire

Bienvenue sur **ScoreCredit**, un projet d'analyse prédictive qui statut sur la solvabilité des clients d'une banque!

## 📌 Objectif

Développer un outil intelligent permettant :
- d’évaluer la probabilité qu’un client soit non solvable,
- de classifier les clients en fonction de leur risque financier,
- et de fournir une interface simple et interactive via **Streamlit**.

## 📂 Contenu du projet

| Fichier                          | Description |
|----------------------------------|-------------|
| `app.py`                         | Application Streamlit principale |
| `Etude_solvabilitée.ipynb`       | Notebook d'exploration, modélisation et évaluation des modèles |
| `modele_logistique.joblib`       | Modèle de régression logistique entraîné |
| `modele_knn.joblib`              | Modèle KNN entraîné |
| `scaler.joblib`                  | Standardiseur des données (StandardScaler) |
| `scoring.sav`                    | Base de données initiale au format SPSS |
| `test.csv`                       | Fichier CSV d'exemple pour l'analyse groupée |

## 🧠 Modèles utilisés

- **Régression logistique**
- **K-Nearest Neighbors (KNN)**

Les deux modèles ont été comparés selon plusieurs métriques :
- Accuracy
- Matrice de confusion
- Précision, Rappel
- AUC / ROC
- Cross-validation (K-fold et validation croisée imbriquée)

## 🖥️ Application interactive

Grâce à **Streamlit**, vous pouvez :
- Analyser individuellement un client (âge, revenu, montant emprunté…)
- Télécharger un fichier `.csv` contenant plusieurs clients à évaluer
- Visualiser le risque de défaut sous forme de **probabilité en notation scientifique**
- Obtenir une recommandation de crédit : ✅ Accepté / ❌ Refusé

### ▶️ Pour lancer l'application :
```bash
streamlit run app.py
