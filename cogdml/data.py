from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

THRESHOLD = 2.5
SEED = 42


def _cols(df, prefix):
    # PBC1 also starts with "PB", so PB has to exclude it explicitly.
    if prefix == "PB":
        return [c for c in df.columns if c.startswith("PB") and not c.startswith("PBC")]
    return [c for c in df.columns if c.startswith(prefix)]


def load_survey(path=None):
    return pd.read_csv(path or DATA / "survey_bangladesh.csv")


def constructs(df):
    return {p: _cols(df, p) for p in ("PB", "ATTD", "SN", "PBC")}


def scores(df):
    c = constructs(df)
    out = pd.DataFrame({k: df[v].mean(axis=1) for k, v in c.items()})
    out["activity"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    out["frequency"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return out


def features(df):
    c = constructs(df)
    return df[c["ATTD"] + c["SN"] + c["PBC"]]


def label(df, threshold=THRESHOLD):
    return (df[constructs(df)["PB"]].mean(axis=1) >= threshold).astype(int)


def load_external(path=None):
    """North India green-purchase sample, mapped onto the survey feature space.

    The 7-point scale is rescaled to 5, and the two items with no counterpart
    (SN4, PBC4) are filled with training-set means by the caller.
    """
    df = pd.read_excel(path or DATA / "external_north_india.xlsx")
    scale = 5.0 / 7.0
    X = pd.DataFrame({
        "ATTD1": df["AT1"], "ATTD2": df["AT2"], "ATTD3": df["AT3"], "ATTD4": df["AT4"],
        "SN1": df["SN1"], "SN2": df["SN2"], "SN3": df["SN3"], "SN4": pd.NA,
        "PBC1": df["PBC1"], "PBC2": df["PBC2"], "PBC3": df["PBC3"], "PBC4": pd.NA,
    })
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce") * scale
    y = (df[[f"GPB{i}" for i in range(1, 7)]].mean(axis=1) >= 4.0).astype(int)
    return X, y
