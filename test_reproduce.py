"""Guards the two things most likely to break silently."""
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from cogdml.data import SEED, constructs, features, label, load_survey
from cogdml.metrics import score


def test_pb_excludes_pbc():
    df = load_survey()
    c = constructs(df)
    assert len(c["PB"]) == 4
    assert not set(c["PB"]) & set(c["PBC"])


def test_headline_accuracy():
    df = load_survey()
    X, y = features(df), label(df)
    assert y.mean() == 289 / 352
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = SVC(kernel="linear", C=10, probability=True, random_state=SEED).fit(Xtr, ytr)
    assert score(yte, model.predict(Xte))["accuracy"] > 0.90


if __name__ == "__main__":
    test_pb_excludes_pbc()
    test_headline_accuracy()
    print("ok")
