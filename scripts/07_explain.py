"""SHAP and LIME explanations for the fitted SVM."""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from cogdml.data import SEED, features, label, load_survey
from cogdml.metrics import save


def main():
    df = load_survey()
    X, y = features(df), label(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = SVC(kernel="rbf", probability=True, random_state=SEED).fit(Xtr, ytr)

    background = shap.sample(Xtr, 50, random_state=SEED)
    values = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], background).shap_values(Xte)
    ranking = pd.DataFrame({"feature": X.columns,
                            "mean_abs_shap": np.abs(values).mean(axis=0)})
    save(ranking.sort_values("mean_abs_shap", ascending=False), "shap_feature_ranking.csv")

    explainer = LimeTabularExplainer(Xtr.values, feature_names=list(X.columns),
                                     class_names=["unlikely", "likely"], random_state=SEED)
    local = explainer.explain_instance(Xte.values[0], model.predict_proba, num_features=len(X.columns))
    save(pd.DataFrame(local.as_list(), columns=["rule", "weight"]), "lime_instance_0.csv")


if __name__ == "__main__":
    main()
