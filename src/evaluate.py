"""Metrics preserve undefined subgroup rates as NaN, never artificial zeros."""
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score,
                             average_precision_score)


def ratio(a, b):
    return float(a / b) if b else np.nan


def metrics(y, probability, threshold=0.5):
    pred = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), accuracy=accuracy_score(y, pred), precision=ratio(tp, tp+fp),
        recall=ratio(tp, tp+fn), f1=ratio(2*tp, 2*tp+fp+fn),
        roc_auc=roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan,
        pr_auc=average_precision_score(y, probability) if len(np.unique(y)) == 2 else np.nan,
        fpr=ratio(fp, fp+tn), fnr=ratio(fn, fn+tp), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def threshold_table(y, p):
    return pd.DataFrame([dict(threshold=float(t), **metrics(y, p, t))
                         for t in np.linspace(0.01, 0.99, 99)])


def choose_threshold(table):
    # Predeclared research constraint, not an asserted production requirement.
    candidates = table[(table.recall >= 0.95) & (table.fpr <= 0.05)]
    feasible = not candidates.empty
    if not feasible:
        candidates = table
    best = candidates.sort_values(['f1', 'fpr', 'threshold'], ascending=[False, True, False]).iloc[0]
    return float(best.threshold), feasible


def audit(frame, probability, threshold):
    records = []
    for column in ['proto', 'service', 'state', 'attack_cat']:
        for group, subset in frame.groupby(column, dropna=False):
            positions = frame.index.get_indexer(subset.index)
            result = metrics(subset.label, probability[positions], threshold)
            # Wilson intervals quantify recall uncertainty for small attack groups.
            n = result['tp'] + result['fn']
            r = result['recall']
            z = 1.96
            if n:
                center = (r + z*z/(2*n))/(1+z*z/n)
                delta = z*np.sqrt(r*(1-r)/n+z*z/(4*n*n))/(1+z*z/n)
            else:
                center = delta = np.nan
            records.append(dict(group_type=column, group=str(group), **result,
                                recall_ci_low=center-delta, recall_ci_high=center+delta))
    return pd.DataFrame(records)
