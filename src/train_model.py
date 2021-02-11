#!/usr/bin/env python

import json
import os
import sys

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from utils import calculate_metrics

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')


def train():
    channel_name = 'training'
    training_path = os.path.join(input_path, channel_name)

    train_df = pd.read_csv(f"{training_path}/train.csv")

    y = train_df.pop('narrowing-diagnosis')
    X = train_df

    params_file = "params.json"

    with open(params_file) as f:
        params = json.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_data = Pool(data=X_train, cat_features=params['data_params']['cat_features'], label=y_train)
    test_data = Pool(data=X_test, cat_features=params['data_params']['cat_features'], label=y_test)

    model = CatBoostClassifier(**params['model_params'])

    model.fit(train_data, eval_set=test_data, verbose=False, plot=False)

    model.save_model(f"{model_path}/heart.cbm")

    print('model has been trained successfully')

    predict_probas = model.predict_proba(test_data)

    model_metrics = calculate_metrics(params['data_params']['metrics'], y_test, predict_probas)

    print('Metrics for trained model:')
    for k, v in model_metrics.items():
        print(f'{k}={v}')

    with open(f"{model_path}/metrics.json", 'w') as fp:
        fp.write(json.dumps(model_metrics))


if __name__ == '__main__':
    train()
    sys.exit(0)
