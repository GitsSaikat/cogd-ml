"""Pathway tests with Benjamini-Hochberg correction and the multivariate model (Tables 19-20)."""
import numpy as np
import pandas as pd
from scipy import stats

from cogdml.data import load_survey, scores
from cogdml.metrics import save

PATHS = [("H1", "PBC", "PB"), ("H2", "ATTD", "PB"), ("H3", "SN", "PB"),
         ("H4", "frequency", "PB"), ("H5", "activity", "PBC"), ("H6", "activity", "ATTD"),
         ("H7", "frequency", "ATTD"), ("H8", "SN", "PBC"), ("H9", "activity", "SN"),
         ("H10", "frequency", "ATTD")]


def ols(y, *xs):
    X = np.column_stack([np.ones(len(y))] + [x.values for x in xs])
    beta = np.linalg.lstsq(X.T @ X, X.T @ y.values, rcond=None)[0]
    resid = y.values - X @ beta
    n, k = len(y), X.shape[1] - 1
    mse = (resid ** 2).sum() / (n - k - 1)
    se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(abs(t), n - k - 1))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, se, t, p, r2


def bh(p):
    p = np.asarray(p)
    order, adj, run = np.argsort(p), np.empty(len(p)), 1.0
    for rank in range(len(p) - 1, -1, -1):
        i = order[rank]
        run = min(run, p[i] * len(p) / (rank + 1))
        adj[i] = run
    return adj


def main():
    s = scores(load_survey())

    fits = [(h, ols(s[y], s[x])) for h, x, y in PATHS]
    adj = bh([f[3][1] for _, f in fits])
    rows = [{"hypothesis": h, "beta": round(b[1], 3), "se": round(se[1], 3),
             "t": round(t[1], 3), "p": p[1], "p_adj": pa,
             "decision": "supported" if pa < 0.05 else "rejected"}
            for (h, (b, se, t, p, _)), pa in zip(fits, adj)]
    save(pd.DataFrame(rows), "table19_hypotheses.csv")

    beta, se, t, p, r2 = ols(s["PB"], s["ATTD"], s["SN"], s["PBC"])
    n, k = len(s), 3
    multi = pd.DataFrame({
        "predictor": ["intercept", "ATTD", "SN", "PBC"],
        "beta": beta.round(3), "se": se.round(3), "t": t.round(3), "p": p,
    })
    multi.attrs["r2"] = r2
    save(multi, "table20_multivariate.csv")
    print(f"R2 {r2:.4f}  F({k},{n - k - 1}) {(r2 / k) / ((1 - r2) / (n - k - 1)):.2f}")


if __name__ == "__main__":
    main()
