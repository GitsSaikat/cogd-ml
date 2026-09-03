"""Binarisation threshold sensitivity and Youden analysis (Tables 15-16)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from cogdml.data import SEED, constructs, features, load_survey
from cogdml.metrics import save, score


def main():
    df = load_survey()
    X = features(df).values
    pb = df[constructs(df)["PB"]].mean(axis=1)

    rows = []
    for t in np.arange(2.25, 4.25, 0.25):
        y = (pb >= t).astype(int).values
        if not 0.1 < y.mean() < 0.95:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        row = {"threshold": round(t, 2), "balance": round(y.mean(), 3),
               "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
        for name, est in (("svm", SVC(kernel="rbf", random_state=SEED)),
                          ("adaboost", AdaBoostClassifier(n_estimators=50, random_state=SEED))):
            pred = est.fit(Xtr, ytr).predict(Xte)
            row |= {f"{name}_acc": round(accuracy_score(yte, pred), 4),
                    f"{name}_f1": round(f1_score(yte, pred), 4)}
        rows.append(row)
    save(pd.DataFrame(rows), "table15_threshold_sensitivity.csv")

    y = (pb >= 2.5).astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    proba = SVC(kernel="rbf", probability=True, random_state=SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fpr, tpr, cuts = roc_curve(yte, proba)
    j = np.argmax(tpr - fpr)

    out = [{"operating_point": "default (0.50)"} | score(yte, (proba >= 0.5).astype(int), proba),
           {"operating_point": f"youden ({cuts[j]:.3f})"} | score(yte, (proba >= cuts[j]).astype(int), proba)]
    save(pd.DataFrame(out), "table16_roc_youden.csv")
    print(f"AUROC {auc(fpr, tpr):.4f}  youden cutoff {cuts[j]:.4f}")


if __name__ == "__main__":
    main()
