from pathlib import Path

from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, recall_score, roc_auc_score)

from .data import ROOT


def score(y, pred, proba=None):
    out = {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "minority_recall": recall_score(y, pred, pos_label=0),
    }
    if proba is not None:
        out["auroc"] = roc_auc_score(y, proba)
    return out


def save(df, name):
    path = Path(ROOT) / "outputs" / name
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return path
