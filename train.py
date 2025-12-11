import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import numpy as np

# Chemins
BASE_DIR = os.path.dirname(__file__)
possible_files = [
    os.path.join(BASE_DIR, 'prix_rabat.csv'),
    os.path.join(BASE_DIR, 'Housing.csv'),
    os.path.join(BASE_DIR, 'Housing.CSV')
]
CSV_FILE = next((p for p in possible_files if os.path.exists(p)), None)

if CSV_FILE is None:
    raise FileNotFoundError("Aucun fichier CSV connu trouvé (recherchés: prix_rabat.csv, Housing.csv).")

print(f"Utilisation du dataset: {os.path.basename(CSV_FILE)}")
df = pd.read_csv(CSV_FILE)

# Détection de la colonne cible
if 'prix' in df.columns:
    y_col = 'prix'
elif 'price' in df.columns:
    y_col = 'price'
else:
    raise ValueError('Colonne cible introuvable : attendue "prix" ou "price"')

# Colonnes numériques et catégorielles recommandées pour ce dataset
numeric_candidates = ['area', 'surface', 'bedrooms', 'bathrooms', 'stories', 'parking']
cat_candidates = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

cat_candidates = ['quartier', 'property_type', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numeric_features = [c for c in numeric_candidates if c in df.columns]
categorical_features = [c for c in cat_candidates if c in df.columns]

if not numeric_features:
    raise ValueError('Aucune colonne numérique détectée automatiquement. Adaptez le script au CSV.')

original_input_features = numeric_features + categorical_features
X = df[original_input_features].copy()
y = df[y_col].copy()

# Nettoyage simple: replace yes/no by 1/0 for boolean-like categorical columns (operate on X)
for col in categorical_features:
    if col in X.columns and X[col].dropna().isin(['yes', 'no', 'Yes', 'No']).all():
        X[col] = X[col].map(lambda v: 1 if str(v).lower() == 'yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, [c for c in categorical_features if X[c].dtype == object])
    ],
    remainder='passthrough'
)

# Note: categorical variables that are strings will be handled by the ColumnTransformer
# Full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])

# Hyperparameter search space
param_distributions = {
    'regressor__n_estimators': [100, 200, 300, 500],
    'regressor__max_depth': [None, 8, 12, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(pipeline, param_distributions, n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1)

print('Lancement de la recherche d\'hyperparamètres (cela peut prendre du temps)...')
search.fit(X_train, y_train)

best = search.best_estimator_
print('Meilleurs paramètres:', search.best_params_)

# Évaluation
preds = best.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f'Evaluation test — MAE: {mae:.2f}, R2: {r2:.4f}')

# Création du dossier model/ s’il n'existe pas
model_dir = os.path.join(BASE_DIR, 'model')
os.makedirs(model_dir, exist_ok=True)

# Sauvegarde du pipeline (préprocesseur + modèle)
joblib.dump(best, os.path.join(model_dir, 'model_rabat.pkl'))

# Sauvegarde des noms de colonnes d'entrée (utiles pour construire la ligne depuis l'app)
joblib.dump(list(original_input_features), os.path.join(model_dir, 'features.pkl'))

# Obtenir les noms de features après préprocessing (pour debug/inspection)
try:
    feature_names_out = best.named_steps['preprocessor'].get_feature_names_out()
    joblib.dump(list(feature_names_out), os.path.join(model_dir, 'feature_names_out.pkl'))
except Exception:
    pass

# Sauvegarde des colonnes catégorielles de type string utilisées lors de l'entraînement
categorical_string_features = [c for c in categorical_features if c in original_input_features and X[c].dtype == object]
joblib.dump(categorical_string_features, os.path.join(model_dir, 'categorical_string_features.pkl'))

# Sauvegarde des métriques et métadonnées du modèle (pour affichage pédagogique)
metrics = {
    'mae': float(mae),
    'r2': float(r2),
    'n_samples': len(df),
    'n_features': len(original_input_features),
    'algorithm': 'RandomForest',
    'best_params': search.best_params_
}
joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))
    
# Save training metadata for pedagogical fiche
metrics = {
    'mae': float(mae),
    'r2': float(r2),
    'n_samples': int(len(df)),
    'n_features': int(len(original_input_features))
}
joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))

# Save regressor params
try:
    reg = best.named_steps.get('regressor')
    if reg is None:
        # find any regressor
        for name, step in best.named_steps.items():
            if hasattr(step, 'get_params') and 'n_estimators' in step.get_params():
                reg = step
                break
    params = reg.get_params() if reg is not None else {}
    joblib.dump(params, os.path.join(model_dir, 'rf_params.pkl'))
except Exception:
    pass

print("🎉 Modèle entraîné et sauvegardé dans model/")
