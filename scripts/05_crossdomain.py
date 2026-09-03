"""Transferability check against the North India sample (Table 17)."""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cogdml.data import SEED, features, label, load_external, load_survey
from cogdml.metrics import save, score


def main():
    df = load_survey()
    X, y = features(df).values, label(df).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(Xtr)

    Xext, yext = load_external()
    Xext = Xext.fillna(pd.Series(Xtr.mean(axis=0), index=Xext.columns)).values

    rows = []
    for name, est in (("SVM", SVC(kernel="rbf", probability=True, random_state=SEED)),
                      ("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=SEED))):
        est.fit(scaler.transform(Xtr), ytr)
        for domain, Xs, ys in (("in-domain (Bangladesh)", scaler.transform(Xte), yte),
                               ("cross-domain (North India)", scaler.transform(Xext), yext)):
            rows.append({"model": name, "domain": domain}
                        | score(ys, est.predict(Xs), est.predict_proba(Xs)[:, 1]))
    save(pd.DataFrame(rows).round(4), "table17_crossdomain.csv")


if __name__ == "__main__":
    main()
