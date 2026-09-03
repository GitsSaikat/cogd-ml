"""Construct-subset ablation across eight classifiers (Table 9)."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from cogdml.data import SEED, constructs, features, label, load_survey
from cogdml.metrics import save
from cogdml.models import grids, zoo


def main():
    df = load_survey()
    c = constructs(df)
    X, y = features(df), label(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    subsets = {
        "ATTD, SN & PBC": c["ATTD"] + c["SN"] + c["PBC"],
        "ATTD & SN": c["ATTD"] + c["SN"],
        "ATTD & PBC": c["ATTD"] + c["PBC"],
        "SN & PBC": c["SN"] + c["PBC"],
        "ATTD": c["ATTD"], "SN": c["SN"], "PBC": c["PBC"],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, cols in subsets.items():
        for model, est in zoo().items():
            gs = GridSearchCV(Pipeline([("m", est)]), grids()[model], cv=cv, n_jobs=-1)
            gs.fit(Xtr[cols], ytr)
            pred = gs.best_estimator_.predict(Xte[cols])
            rows.append({
                "factors": name, "model": model,
                "accuracy": round(accuracy_score(yte, pred) * 100),
                "precision": round(precision_score(yte, pred, average="weighted", zero_division=0) * 100),
                "recall": round(recall_score(yte, pred, average="weighted") * 100),
                "f1": round(f1_score(yte, pred, average="weighted") * 100),
                "params": gs.best_params_,
            })
    save(pd.DataFrame(rows), "table09_ablation.csv")


if __name__ == "__main__":
    main()
