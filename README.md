📌 README.md — Predicteur de Prix Immobilier (Flask + Machine Learning)
# 🏡 Prédiction du Prix Immobilier — Rabat  
Application Web Flask + Random Forest

Ce projet est une application web développée avec **Flask** permettant de prédire le **prix des biens immobiliers à Rabat** à partir de différentes caractéristiques (surface, quartier, nombre de chambres, type de bien, étage, parking…).

Le modèle de Machine Learning utilisé est un **Random Forest Regressor**, entraîné localement sur un dataset immobilier marocain.

---

## 🚀 Fonctionnalités

- Interface moderne et intuitive en HTML/CSS  
- Formulaire permettant de saisir les caractéristiques du bien  
- Prédiction instantanée du prix via un modèle Random Forest  
- Traitement backend avec Flask  
- Chargement du modèle grâce à `joblib`  
- Code structuré et simple à comprendre  

---

## 🧠 Modèle de Machine Learning

Le modèle utilisé :

- **Algorithme :** RandomForestRegressor  
- **Bibliothèque :** scikit-learn  
- **Prétraitement :** pandas + numpy  
- **Sauvegarde du modèle :** joblib  

Le script d’entraînement se trouve dans `train.py`, et l’inférence est réalisée dans `app.py`.

---

## 📂 Structure du projet



flask_datascience/
│── app.py
│── train.py
│── model.joblib
│── requirements.txt
│── static/
│ └── styles.css
│── templates/
│ └── index.html
└── README.md


---

## 🛠️ Installation & Exécution

### 1. Cloner le repo

```bash
git clone https://github.com/Abdelhay-Rahmouni/flask_datascience.git
cd flask_datascience

2. Installer les dépendances
pip install -r requirements.txt

3. Lancer l’application
python app.py


L’app tourne ensuite sur :

👉 http://127.0.0.1:5000

📦 Fichier requirements.txt
Flask==3.0.0
scikit-learn==1.3.0
pandas==2.1.0
numpy==1.24.0
joblib==1.3.0



🧑‍💻 Auteur

Rahmouni Abdelhay
Projet pédagogique — Flask + Machine Learning
Prédiction immobilière marocaine
