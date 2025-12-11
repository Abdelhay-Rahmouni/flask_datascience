from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Charger modèle + features
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model_rabat.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.pkl')

# optional: categorical string features list saved by train.py
CATEGORICAL_STR_PATH = os.path.join(MODEL_DIR, 'categorical_string_features.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError('model/model_rabat.pkl ou model/features.pkl introuvable. Exécutez train.py d\'abord.')

model = joblib.load(MODEL_PATH)
# Note: load features dynamically inside predict() to keep in sync with retraining
# Try to detect which columns the preprocessor expects as categorical (if model is a pipeline)
trained_cat_cols = []
try:
    pre = model.named_steps.get('preprocessor')
    if pre is not None and hasattr(pre, 'transformers_'):
        for name, trans, cols in pre.transformers_:
            if name == 'cat':
                # cols can be list or slice; try to convert to list
                try:
                    trained_cat_cols = list(cols)
                except Exception:
                    trained_cat_cols = cols
                break
except Exception:
    trained_cat_cols = []

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer champs demandés (certains sont optionnels selon le dataset)
    area = request.form.get('area')
    surface = request.form.get('surface')
    if area and not surface:
        surface_val = float(area)
    elif surface:
        surface_val = float(surface)
    else:
        surface_val = 0.0

    chambres = int(request.form.get('chambres') or 0)
    etage = request.form.get('etage')
    if etage is None or etage == '':
        etage = 0
    else:
        etage = int(etage)

    parking = request.form.get('parking')
    if parking is None or parking == '':
        parking = 0
    else:
        try:
            parking = int(parking)
        except ValueError:
            parking = 1 if parking.lower() in ['yes','y','1'] else 0

    mainroad = request.form.get('mainroad')
    if mainroad is None:
        mainroad = 'no'
    # quartier from dropdown
    quartier = request.form.get('quartier')
    # property type (Appartement, Villa, Studio, ...)
    property_type = request.form.get('property_type')

    # (re)load expected input feature names saved by train.py
    features = joblib.load(FEATURES_PATH)
    # load categorical string features list (if available)
    try:
        categorical_string_features = joblib.load(CATEGORICAL_STR_PATH)
    except Exception:
        categorical_string_features = []

    # Build row using columns that the training saved in `features`
    row = {}
    # Try to match common numeric names
    for f in features:
        if f in ['area','surface']:
            row[f] = surface_val
        elif f in ['bedrooms','chambres']:
            row[f] = chambres
        elif f in ['stories','etage']:
            row[f] = etage
        elif f == 'parking':
            row[f] = parking
        elif f == 'mainroad':
            # consistent mapping as in training: yes/no or 1/0
            row[f] = 1 if str(mainroad).lower() in ['yes','y','1'] else 0
        elif f == 'property_type':
            # set property type string if the training used this column
            row[f] = property_type or ''
        else:
            # placeholder, will be filled by zeros later for numeric,
            # or empty string for categorical string features
            if f in categorical_string_features:
                row[f] = ''
            else:
                row[f] = 0

    # If 'quartier' was used in training, ensure one-hot columns are created
    if any(col.startswith('quartier_') for col in features):
        # create dummy for this quartier
        q_col = f"quartier_{quartier}" if quartier else None
        if q_col and q_col in features:
            row[q_col] = 1

    df = pd.DataFrame([row], columns=features)
    df = df.reindex(columns=features, fill_value=0)

    # Ensure categorical columns expected by the trained preprocessor are strings
    for c in trained_cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    prix = model.predict(df)[0]

    # Load saved metrics and rf params if available
    metrics = {}
    rf_params = {}
    try:
        metrics = joblib.load(os.path.join(MODEL_DIR, 'metrics.pkl'))
    except Exception:
        metrics = {}
    try:
        rf_params = joblib.load(os.path.join(MODEL_DIR, 'rf_params.pkl'))
    except Exception:
        # fallback: try to read regressor in pipeline
        try:
            reg = model.named_steps.get('regressor')
            if reg is None:
                for name, step in model.named_steps.items():
                    if hasattr(step, 'get_params') and 'n_estimators' in step.get_params():
                        reg = step
                        break
            if reg is not None:
                params = reg.get_params()
                for k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','random_state']:
                    if k in params:
                        rf_params[k] = params[k]
        except Exception:
            rf_params = {}

    # Input summary: only include fields provided by the user on the form
    input_items = {}
    if surface is not None and surface != '':
        input_items['surface'] = surface_val
    input_items['chambres'] = chambres
    if etage and int(etage) != 0:
        input_items['etage'] = etage
    if parking and int(parking) != 0:
        input_items['parking'] = parking
    if quartier:
        input_items['quartier'] = quartier
    if property_type:
        input_items['property_type'] = property_type

    # Load metrics if available
    metrics = {}
    try:
        metrics = joblib.load(METRICS_PATH)
    except Exception:
        metrics = {}

    # Price formatting helpers
    def fmt_mad(x):
        return f"{int(x):,} MAD".replace(',', ' ')

    def in_millions(x):
        return f"~{round(x/1_000_000,2)}M MAD"

    # Prepare display values
    display = {
        'prix_full': fmt_mad(round(prix)),
        'prix_millions': in_millions(prix)
    }

    return render_template('predict.html', prix=round(prix,2), display=display, rf_params=rf_params, metrics=metrics, input_items=input_items)

    # Prepare display values
    display = {
        'prix_full': fmt_mad(round(prix)),
        'prix_millions': in_millions(prix)
    }

    return render_template('predict.html', prix=round(prix,2), display=display, rf_params=rf_params, metrics=metrics, input_items=input_items)

if __name__ == '__main__':
    app.run(debug=True)
