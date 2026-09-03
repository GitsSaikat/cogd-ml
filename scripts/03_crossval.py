"""Stratified five-fold cross-validation and paired tests (Tables 12-14)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate

from cogdml.data import SEED, features, label, load_survey
from cogdml.metrics import save
from cogdml.models import zoo


def ci(v):
    mu, sd = v.mean(), v.std(ddof=1)
    h = 1.96 * sd / np.sqrt(len(v))
    return mu, sd, mu - h, mu + h


def main():
    df = load_survey()
    X, y = features(df).values, label(df).values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    folds, rows = {}, []
    for name, est in zoo().items():
        r = cross_validate(est, X, y, cv=cv,
                           scoring=["accuracy", "f1", "precision", "recall", "roc_auc"])
        folds[name] = r["test_accuracy"]
        row = {"model": name}
        for m in ("accuracy", "f1", "precision", "recall", "roc_auc"):
            mu, sd, lo, hi = ci(r["test_" + m])
            row |= {m: round(mu, 3), m + "_sd": round(sd, 3),
                    m + "_ci": f"[{lo:.3f}, {hi:.3f}]"}
        rows.append(row)
    save(pd.DataFrame(rows), "table12_13_crossval.csv")

    tests = [{"comparison": f"SVM vs {k}",
              "t": round(stats.ttest_rel(folds["SVM"], v).statistic, 3),
              "p": round(stats.ttest_rel(folds["SVM"], v).pvalue, 4)}
             for k, v in folds.items() if k != "SVM"]
    save(pd.DataFrame(tests), "table14_paired_tests.csv")


if __name__ == "__main__":
    main()
