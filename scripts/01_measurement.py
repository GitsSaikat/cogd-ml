"""Reliability, discriminant validity and construct correlations (Tables 5-8, 18)."""
import numpy as np
import pandas as pd
from scipy import stats

from cogdml.data import constructs, load_survey, scores
from cogdml.metrics import save


def cronbach(x):
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def htmt(df, cols_a, cols_b):
    hetero = df[cols_a].corrwith(df[cols_b].mean(axis=1)).mean()
    mono_a = df[cols_a].corr().values[np.triu_indices(len(cols_a), 1)].mean()
    mono_b = df[cols_b].corr().values[np.triu_indices(len(cols_b), 1)].mean()
    return hetero / np.sqrt(mono_a * mono_b)


def main():
    df = load_survey()
    c = constructs(df)
    s = scores(df)

    rel = pd.DataFrame([{"construct": k, "items": len(v), "alpha": cronbach(df[v])}
                        for k, v in c.items()])
    save(rel, "table05_reliability.csv")

    names = list(c)
    ht = pd.DataFrame([{"pair": f"{a}-{b}", "htmt": htmt(df, c[a], c[b])}
                       for i, a in enumerate(names) for b in names[i + 1:]])
    save(ht, "table07_htmt.csv")

    cols = list(s.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r, p = stats.pearsonr(s[a], s[b])
            rows.append({"pair": f"{a} <-> {b}", "r": round(r, 3), "p": p})
    corr = pd.DataFrame(rows).sort_values("r", key=abs, ascending=False)
    save(corr, "table18_correlations.csv")

    assert abs(corr.set_index("pair").loc["PB <-> PBC", "r"] - 0.530) < 0.002
    print(rel.to_string(index=False))


if __name__ == "__main__":
    main()
