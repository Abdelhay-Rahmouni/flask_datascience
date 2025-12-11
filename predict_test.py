import joblib
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(__file__))
model = joblib.load(os.path.join(BASE,'model','model_rabat.pkl'))
features = joblib.load(os.path.join(BASE,'model','features.pkl'))
# detect which columns the preprocessor expects as categorical object columns
cat_str = []
try:
    pre = model.named_steps.get('preprocessor')
    if pre is not None and hasattr(pre, 'transformers_'):
        for name, trans, cols in pre.transformers_:
            if name == 'cat':
                try:
                    cat_str = list(cols)
                except Exception:
                    cat_str = cols
                break
except Exception:
    cat_str = []

row = {}
for f in features:
    if f in ['area','surface']:
        row[f] = 75.0
    elif f in ['bedrooms','chambres']:
        row[f] = 3
    elif f in ['stories','etage']:
        row[f] = 2
    elif f == 'parking':
        row[f] = 1
    elif f == 'mainroad':
        row[f] = 1
    elif f in cat_str:
           row[f] = ''
    else:
        row[f] = 0

X = pd.DataFrame([row], columns=features)
print('Input df dtypes:')
print(X.dtypes)
print('Predicting...')
print(model.predict(X))
