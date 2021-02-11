import sys
from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostClassifier, Pool


project_dir = Path(__file__).resolve().parents[2]

params = json.load(open(f"{project_dir}/src/models/params.json"))

model = CatBoostClassifier()

model.load_model(f"{project_dir}/models/heart.cbm")

test_df = pd.read_csv(f"{project_dir}/data/processed/test.csv")

target = test_df.pop(params['data_params']['target'])
X = test_df

test_pool = Pool(X, target, params['data_params']['cat_features'])

print(model.predict(test_pool))

sys.exit(0)