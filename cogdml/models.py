from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import SEED


def zoo():
    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=1000, random_state=SEED),
    }


def grids():
    return {
        "Logistic Regression": {"m__C": [0.1, 1, 10], "m__penalty": ["l1", "l2"], "m__solver": ["liblinear"]},
        "MLP": {"m__hidden_layer_sizes": [(50,), (100,), (50, 50)], "m__activation": ["tanh", "relu"],
                "m__solver": ["sgd", "adam"], "m__alpha": [0.0001, 0.01],
                "m__learning_rate": ["constant", "adaptive"]},
        "SVM": {"m__C": [0.1, 1, 10], "m__kernel": ["linear", "rbf", "poly"], "m__gamma": ["scale", "auto"]},
        "KNN": {"m__n_neighbors": [3, 5, 7], "m__weights": ["uniform", "distance"],
                "m__metric": ["euclidean", "manhattan"]},
        "Decision Tree": {"m__criterion": ["gini", "entropy"], "m__max_depth": [None, 10, 20],
                          "m__min_samples_split": [2, 5, 10]},
        "AdaBoost": {"m__n_estimators": [50, 100, 200], "m__learning_rate": [0.01, 0.1, 1]},
        "Random Forest": {"m__n_estimators": [100, 200, 500], "m__max_features": ["sqrt", "log2"],
                          "m__max_depth": [None, 10, 20]},
        "Gradient Boosting": {"m__n_estimators": [10, 50, 100, 200], "m__learning_rate": [0.01, 0.1, 1],
                              "m__max_depth": [1, 2, 3, 5]},
    }
