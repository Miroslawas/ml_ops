import json
import os
import shutil
import uuid
import warnings
from importlib import import_module
from logging import getLogger, FileHandler, DEBUG
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing._label import label_binarize

from pyprojroot import here

import custom_metrics

class_prediction_metrics = ["accuracy"]


def _check_directory(directory: str, if_exists: str) -> str:
    if os.path.exists(directory):
        if if_exists == 'error':
            raise ValueError('directory {} already exists.'.format(directory))
        elif if_exists == 'replace':
            warnings.warn(
                'directory {} already exists. It will be replaced by the new result'.format(directory))
            shutil.rmtree(directory, ignore_errors=True)
        elif if_exists == 'rename':
            postfix_index = 1

            while os.path.exists(directory + '_' + str(postfix_index)):
                postfix_index += 1

            directory += '_' + str(postfix_index)
            warnings.warn('directory is renamed to {} because the original directory already exists.'.format(directory))
        elif if_exists == "rebuild":
            pass

    return directory


class ModelMgr(object):
    def __init__(self,
                 exp_dir: str,
                 write_mode=True,
                 if_exists: str = 'error'
                 ):

        self.project_dir = here()  # will this work if installed as library somewhere else?
        self.logging_directory = f"{self.project_dir}/models/{exp_dir}"
        self.results_dir = f"{self.logging_directory}/results"
        self.specification_dir = f"{self.logging_directory}/specification"

        if write_mode:
            _check_directory(self.logging_directory, if_exists)
            os.makedirs(self.logging_directory, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.specification_dir, exist_ok=True)
            # self.logging_directory = logging_directory
            self.logger = getLogger(str(uuid.uuid4()))
            self.log_path = os.path.join(self.logging_directory, 'log.txt')
            self.logger.addHandler(FileHandler(self.log_path))
            self.logger.setLevel(DEBUG)

    def get_model_spec(self):
        with open(f"{self.specification_dir}/model_spec.json") as f:
            model_spec = json.load(f)
            return model_spec

    def get_training_data(self):
        data_dir = f"{self.project_dir}/data/processed"
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        return train_df, test_df

    # fail if not in write mode
    def save_result_dict(self, obj: Dict, filename: str):
        try:
            path = os.path.join(self.results_dir, filename)
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)
        except IOError:
            self.logger.warning('failed to save file: {}'.format(filename))

    def save_result_df(self, df, filename):
        try:
            df.to_csv(f"{self.results_dir}/{filename}")
        except IOError:
            self.logger.warning('failed to save file: {}'.format(filename))

    def save_model_spec(self, obj: Dict, filename: str):
        try:
            path = os.path.join(self.specification_dir, filename)
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)
        except IOError:
            self.logger.warning('failed to save file: {}'.format(filename))

    def save_model(self, model):
        model.save_model(f"{self.logging_directory}/model.cbm")

    def log(self, text: str):
        """
        Logs a message on the logger for the experiment.

        Args:
            text:
                The message to be written.
        """
        self.logger.info(text)

    def verify_model(self, holdout_results):
        # compare with what is in directory
        pass

    def get_results(self):
        with open(f"{self.results_dir}/holdout_results.json") as f:
            results = json.load(f)
            return results

    def __enter__(self):

        return self

    def __exit__(self, ex_type, ex_value, trace):
        pass


def convert_cv_score_to_json(scores):
    j = {}
    for key, value in scores.items():
        j[key] = value.tolist()

    return j


def write_results_dict(d, filename):
    try:

        with open(filename, mode='w+') as f:
            json.dump(d, f, indent=2)
    except IOError:
        print(IOError)


def get_standard_metric(metric_name):
    scoring_module = import_module(f'sklearn.metrics')
    score_function = getattr(scoring_module, f'{metric_name}_score', None)
    return score_function


def get_custom_metric(metric_name):
    f = getattr(custom_metrics, metric_name, None)
    return f


def get_custom_metric_scorer(metric_name):
    metric_function = custom_metrics.CUSTOM_SCORERS.get(metric_name)
    return metric_function


def get_metrics_dict(metrics):
    d = {}
    for m in metrics:
        if get_standard_metric(m) is not None:
            d[m] = m
        else:
            f = get_custom_metric_scorer(m)
            if f is None:
                continue  # log
            d[m] = f

    return d


# I think there is some dependency on lexicographical ordering...i.e, it is
# important that 'N' < 'Y'...which is ok..but need to be careful with different labels

def calculate_metrics(metrics, y_true, y_pred, labels=None):
    output = {}

    for metric_name in metrics:
        score_function = get_standard_metric(metric_name)

        if score_function is None:
            score_function = get_custom_metric(metric_name)
        if score_function is None:
            continue  # log

        if metric_name in class_prediction_metrics:
            if labels is None:
                labels = np.unique(y_true)

            _y_true = label_binarize(y_true, classes=labels)[:, 0]
            score = score_function(_y_true, np.argmax(y_pred, axis=1))
        else:
            score = score_function(y_true, y_pred[:, 1])

        output[metric_name] = score

    return output


def get_model_mgr(model_dir):
    model_mgr = ModelMgr(model_dir, write_mode=False)

    return model_mgr
