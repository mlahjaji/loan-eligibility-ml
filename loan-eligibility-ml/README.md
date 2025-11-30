# 🏦 Loan Eligibility Prediction – Machine Learning Project

Projet réalisé dans le cadre du cours de **Machine Learning** (Master M2MO - Ex Dea Laure Elie).  
Ce projet a été développé par **Mohammed Lahjaji** .

L’objectif est de prédire **si une demande de prêt bancaire sera approuvée** à partir des informations socio-économiques des emprunteurs.  
Le travail suit méthodiquement les étapes  : **EDA, prétraitement, modèles supervisés, interprétabilité, et une partie non supervisée (PCA)**.

---

## 📂 Contenu du dépôt

loan-eligibility-ml/
├── projet_ML.ipynb # Notebook complet (code + explications)
├── figures/ # Graphiques générés (EDA, modèles, PCA)
├── requirements.txt # Environnement Python
└── README.md # Documentation du projet


---

## 📊 Dataset

Dataset utilisé :  
**Loan Prediction Dataset (Kaggle)**  
https://www.kaggle.com/datasets/renatovi/loan-prediction

La cible `Loan_Status` est convertie en variable binaire :
- `1` = Prêt approuvé  
- `0` = Prêt refusé  

---

## 🧹 1. Prétraitement (Pipeline sklearn)

### 🔧 Variables numériques
- Imputation : médiane  
- Standardisation : `StandardScaler`

### 🔧 Variables catégorielles
- Imputation : valeur la plus fréquente  
- Encodage : `OneHotEncoder`

Le tout est assemblé dans un **ColumnTransformer**, garantissant un pipeline propre et reproductible.

---

## 📈 2. Analyse exploratoire des données (EDA)

L'EDA comprend :
- Distributions et boxplots  
- Analyse de la cible  
- Heatmap des valeurs manquantes  
- Corrélations  
- Analyse clé : relation entre `Credit_History` et `Loan_Status`

🎯 **Conclusion EDA importante :**  
> *L’historique de crédit est la variable la plus déterminante : les emprunteurs avec un historique positif ont ~80 % de chance d'obtenir un prêt.*

---

## 🤖 3. Modèles supervisés testés

| Modèle | Type |
|--------|------|
| Logistic Regression | Linéaire |
| SVM (Linear) | Linéaire |
| SVM (RBF) | Non linéaire |
| Random Forest | Ensemble d’arbres |
| MLP (Neural Network) | Non linéaire |

✔️ Évaluation avec **Stratified 5-fold Cross Validation**  
✔️ Métriques : Accuracy, F1, Recall, Precision, ROC-AUC

---

## 🏆 Résultat : Modèle final

Le **Random Forest** est le meilleur modèle.

### ➤ Validation croisée :
- ROC-AUC ≈ **0.78 ± 0.04**

### ➤ Sur le test :
- Accuracy : **0.83**
- F1-score : **0.88**
- ROC-AUC : **0.82**
- Recall : **0.93**

---

## 🔍 4. Interprétation du modèle (Permutation Importance)

Analyse des variables les plus importantes :

1. **Credit_History**  
2. ApplicantIncome  
3. Dependents  
4. Property_Area  

→ Les résultats confirment l’EDA : `Credit_History` domine largement la décision.

---

## 🌀 5. Analyse non supervisée : PCA

Deux premières composantes :
- PC1 : 15.2 %
- PC2 : 11 %

La projection montre un fort mélange entre classes →  
➡️ *il faut des modèles non linéaires (Random Forest / MLP).*

---

## 🚀 Installation

```bash
pip install -r requirements.txt
