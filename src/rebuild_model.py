import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, EFstrType
from sklearn.model_selection import cross_validate

from utils import convert_cv_score_to_json, write_eval_summary_file, generate_metrics_dict, calculate_metrics

project_dir = Path(__file__).resolve().parents[2]

with open(f"{project_dir}/model/data_params.json") as f:
    data_params = json.load(f)
with open(f"{project_dir}/model/model_params.json") as f:
    model_params = json.load(f)
with open(f"{project_dir}/model/build_spec.json") as f:
    build_spec = json.load(f)

# rebuild from local data..already split

local_dat_dir = f"{project_dir}/data"
data_dir = f"{local_dat_dir}/processed"
label = data_params['label']
features = build_spec['features']
cat_features = model_params['cat_features']
metrics = build_spec['standard_metrics']

# if data_params['hold_out_test_set']:

train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")

X_train = train_df[features]
y_train = train_df[label]
train_data = Pool(data=X_train, cat_features=cat_features, label=y_train)
X_test = test_df[features]
y_test = test_df[label]
test_data = Pool(data=X_test, cat_features=cat_features, label=y_test)

model = CatBoostClassifier(**model_params, verbose=False)

metrics = generate_metrics_dict(build_spec['standard_metrics'], build_spec['custom_metrics'])

# or maybe I can fully specify the metrics in the model itself, have cv return the 5 trained models
# and pull out metrics from each?
scores = cross_validate(model, X_train, y_train, cv=5,
                        scoring=metrics,
                        return_train_score=False)

cv_scores = convert_cv_score_to_json(scores)

# now train model on all data:

'''
I can pass standard metrics to model as shown (note that naming conventions are different than sklearn, so 
I have just hardcoded for now).  

custom metrics can be defined https://catboost.ai/docs/concepts/python-usages-examples.html#user-defined-loss-function....but I can't tell how to use them except using eval_metric=XXX.  

so option is to figure this out OR not supply metrics to catboost model and just 
calculate them using y_true and the predicted values.

model = CatBoostClassifier(**model_params, verbose=False, custom_metric=['Logloss', 'AUC', 'F1', 'PRAUC'])


'''

model = CatBoostClassifier(**model_params,)

# model = CatBoostClassifier(**model_params, verbose=False)

model.fit(train_data, eval_set=test_data, verbose=False, plot=False)

model.save_model(f"{project_dir}/model/model.cbm")

builtin_metrics = model.eval_metrics(train_data, metrics=['Logloss', 'AUC', 'F1', 'PRAUC'])

# write results

hold_out_score = model.get_best_score()

# write_eval_summary_file(cv_scores, hold_out_score)


predict_probas = model.predict_proba(test_data)

test_data_metrics = calculate_metrics(build_spec['standard_metrics'], build_spec['custom_metrics'], y_test, predict_probas[:,1])

write_eval_summary_file(cv_scores, test_data_metrics)

predict_probas = pd.DataFrame(predict_probas, columns=['False', 'True'])
predict_probas.to_csv(f"{project_dir}/model/test_data_prediction.csv")

shap_values = model.get_feature_importance(data=test_data, type=EFstrType.ShapValues, shap_calc_type='Exact')

shap_values_df = pd.DataFrame(shap_values, columns=features + ['expected_value'])

shap_values_df.to_csv(f"{project_dir}/model/test_data_shap_values.csv")

top_features = model.get_feature_importance(None, type=EFstrType.PredictionValuesChange, prettified=True)

top_features.to_csv(f"{project_dir}/model/top_features.csv")
