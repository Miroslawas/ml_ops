import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.metrics import precision_recall_curve, make_scorer
from sklearn.preprocessing._label import label_binarize
from scipy.stats.distributions import chi2


def lift(y_true, predict_probas, pct=0.05, labels=None):

    if labels is None:
        labels = np.unique(y_true)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    num_records = len(predict_probas)
    prediction = pd.DataFrame(data=predict_probas, columns=['prediction_proba'])
    prediction['label'] = y_true
    top_pct = math.floor(num_records * pct)
    top = prediction.nlargest(top_pct, ['prediction_proba'])
    failures = len(y_true[(y_true)])  # or 5% if failure rate > 5%
    num_failures_detected_in_top = len(top[top['label']==1])
    lift = num_failures_detected_in_top / failures
    return lift


def pr_auc(y_true, predict_probas, labels=None):
    if labels is None:
        labels = np.unique(y_true)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    precision, recall, _ = precision_recall_curve(y_true, predict_probas)
    pr_auc = sklearn.metrics.auc(recall, precision)
    return pr_auc


def _hosmer_lemeshow(y_true, predict_probas, num_groups=10, labels=None):

    df = pd.DataFrame(data=predict_probas, columns=['prediction_proba'])

    if labels is None:
        labels = np.unique(y_true)

    y_true = label_binarize(y_true, classes=labels)[:, 0]

    df['label'] = y_true
    df['quantile_rank'] = pd.qcut(df['prediction_proba'], num_groups, labels=False, duplicates='drop')
    h = 0
    results = pd.DataFrame(
        columns=['decile', 'lower_bound', 'upper_bound', 'num_observations', 'num_failures', 'predicted_failures'])
    for i in range(num_groups):
        pcat_predictions = df[df['quantile_rank'] == i]
        num_observations = len(pcat_predictions)
        if num_observations == 0:
            continue
        obs1 = len(pcat_predictions[pcat_predictions['label']==1])  # how many were in category 1
        exp1 = pcat_predictions['prediction_proba'].mean() * num_observations
        lower_bound = pcat_predictions['prediction_proba'].min()
        upper_bound = pcat_predictions['prediction_proba'].max()
        obs0 = num_observations - obs1
        exp0 = num_observations - exp1
        h += ((obs1 - exp1) ** 2) / exp1 + ((obs0 - exp0) ** 2) / exp0
        results = results.append({'decile': i + 1, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                                  'num_observations': num_observations,
                                  'num_failures': obs1, 'predicted_failures': exp1}, ignore_index=True)

    p = chi2.sf(h, num_groups - 2)
    return h, p, results

def hosmer_lemeshow(y_true, y_pred, num_groups=10, labels=None):
    h, p, results = _hosmer_lemeshow(y_true, y_pred, num_groups, labels)
    return p

def blended_score(y, y_pred, pos_label='Y'):
    h, p, results = hosmer_lemeshow(y, y_pred, pos_label, 10)
    lift = lift(y, y_pred, 0.05)
    pr_auc = pr_auc(y, y_pred)
    return (p + pr_auc + lift) / 3


hosmer_lemeshow_scorer = make_scorer(hosmer_lemeshow, needs_proba=True)
lift_scorer = make_scorer(lift, needs_proba=True)
pr_auc_scorer = make_scorer(pr_auc, needs_proba=True)


CUSTOM_SCORERS = dict(hosmer_lemeshow=hosmer_lemeshow_scorer,
                      lift=lift_scorer,
                      pr_auc=pr_auc_scorer)