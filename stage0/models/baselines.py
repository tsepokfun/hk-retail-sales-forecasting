"""
Stage 0 — Step 4: the baseline models.

Seasonal Naive, Ridge, LightGBM. All Direct Multi-Horizon (one model per h,
no recursion, no need for future exogenous values).

Every runner returns rows in the same shape:
    (model, origin, h, series, y_true, y_pred)   with y in HK$ million.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import harness as H  # noqa: E402


def _targets(df, mode):
    """
    mode='level'    -> predict y_z(t+h) directly
    mode='residual' -> predict y_z(t+h) - y_z(t), then add y_z(t) back

    Running BOTH modes on the same learner isolates "did the architecture help?"
    from "did the target formulation help?" -- required for a fair comparison
    against the xLSTM, which uses the residual form.
    """
    y = df["target_yz"].to_numpy()
    if mode == "residual":
        y = y - df["y_z"].to_numpy()
    return y


def _reconstruct(pred, df, mode):
    if mode == "residual":
        return pred + df["y_z"].to_numpy()
    return pred


def run_seasonal_naive(wide: pd.DataFrame, origins=H.ORIGINS) -> pd.DataFrame:
    rows = []
    for o in origins:
        for h in H.HORIZONS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            pred = H.seasonal_naive_forecast(wide, o, h)
            actual = wide.loc[tgt]
            for s in wide.columns:
                rows.append(("SeasonalNaive", o, h, s, float(actual[s]), float(pred[s])))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])


def run_drift(wide: pd.DataFrame, origins=H.ORIGINS) -> pd.DataFrame:
    """random walk with drift: yhat(t+h) = y(t) + h * mean monthly change (train)"""
    train = wide.loc[[p for p in wide.index if p <= H.TRAIN_END]]
    drift = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    rows = []
    for o in origins:
        for h in H.HORIZONS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            base = wide.loc[o]
            pred = base + h * drift
            actual = wide.loc[tgt]
            for s in wide.columns:
                rows.append(("Drift", o, h, s, float(actual[s]), float(pred[s])))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])


def run_ridge(feat, wide, scaling, origins=H.ORIGINS, alpha=10.0,
              tag="Ridge", target_mode="level"):
    rows = []
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            Xtr = H.matrix(tr); Xte = H.matrix(te)
            ytr = _targets(tr, target_mode)
            sc = StandardScaler().fit(Xtr)
            mdl = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
            pz = _reconstruct(mdl.predict(sc.transform(Xte)), te, target_mode)
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s, float(wide.loc[H.month_add(o, h), s]), float(yp)))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])


def run_lightgbm(feat, wide, scaling, origins=H.ORIGINS, tag="LightGBM",
                 target_mode="level"):
    import lightgbm as lgb
    params = dict(
        objective="l2", n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4,
    )
    rows = []
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            Xtr = H.matrix(tr); Xte = H.matrix(te)
            ytr = _targets(tr, target_mode)
            mdl = lgb.LGBMRegressor(**params).fit(Xtr, ytr)
            pz = _reconstruct(mdl.predict(Xte), te, target_mode)
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s, float(wide.loc[H.month_add(o, h), s]), float(yp)))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])


# ===========================================================================
# STAGE 0 ADDENDUM — five extra baselines
# ===========================================================================
# Everything below obeys the SAME contract as run_ridge / run_lightgbm:
#   * one model per horizon h (Direct Multi-Horizon, no recursion)
#   * re-fit at every origin; training rows are exactly
#         {target_period <= origin}                       (no leakage)
#   * target_mode in {"level", "residual"} routed through _targets/_reconstruct
#   * predictions returned on the ORIGINAL HK$ million scale
#   * deterministic learners are documented as such (single run, no seed loop)
# ===========================================================================

# the lag channels that live in y_z units (used by NLinear's de-trending)
LAG_COLS = ["lag1", "lag2", "lag3", "lag12", "lag13"]
# the six columns DLinear decomposes, ordered newest -> oldest in time
DLINEAR_COLS = ["y_z", "lag1", "lag2", "lag3", "lag12", "lag13"]
_LAG_IDX = [H.FEATURE_COLS.index(c) for c in LAG_COLS]


# ---------------------------------------------------------------------------
# 1) K-Means (series profiling) + per-cluster Ridge
# ---------------------------------------------------------------------------
def series_profiles(feat) -> pd.DataFrame:
    """
    3-D profile per series, computed on TRAIN ONLY (period <= TRAIN_END):
        y_log mean, y_log std, lag-12 autocorrelation of y_z
    These are properties of the series' history, never of its 2024 outcome.
    """
    tr = feat[feat["period"] <= H.TRAIN_END]
    rows = []
    for s, g in tr.groupby("series", sort=True):
        g = g.sort_values("period")
        ylog = g["y_log"].to_numpy(float)
        a = g["y_z"].to_numpy(float)
        b = g["lag12"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        acf = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 12 else np.nan
        rows.append({"series": s,
                     "y_log_mean": float(np.nanmean(ylog)),
                     "y_log_std": float(np.nanstd(ylog, ddof=1)),
                     "acf12": acf})
    return pd.DataFrame(rows).set_index("series")


def fit_series_clusters(feat, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    KMeans(n_clusters=3, random_state=42, n_init=10) on the standardised
    3-D series profile.  Returns a frame with the profile + 'cluster'.
    Deterministic (random_state pinned), so it is fit once and reused.
    """
    from sklearn.cluster import KMeans

    prof = series_profiles(feat)
    prof = prof[np.isfinite(prof.to_numpy()).all(axis=1)]
    Z = StandardScaler().fit_transform(prof[["y_log_mean", "y_log_std", "acf12"]].to_numpy())
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(Z)
    prof = prof.copy()
    prof["cluster"] = km.labels_.astype(int)
    return prof


def run_kmeans_ridge(feat, wide, scaling, origins=H.ORIGINS, alpha=10.0,
                     tag="KMeansRidge", target_mode="level", n_clusters=3,
                     random_state=42):
    """
    Cluster the 23 series by their TRAIN-period shape, then fit an independent
    Ridge(alpha=10) per cluster / per horizon / per origin.

    Why: a single Ridge has to share one coefficient vector across categories
    whose dynamics differ by an order of magnitude in level and by a lot in
    seasonality.  Clustering is the cheapest way to test whether that pooling
    is the binding constraint.
    """
    prof = fit_series_clusters(feat, n_clusters=n_clusters, random_state=random_state)
    membership = {c: set(g.index) for c, g in prof.groupby("cluster")}

    rows = []
    n_fallback = 0
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            Xtr_all = H.matrix(tr)
            ytr_all = _targets(tr, target_mode)
            # global fallback, fitted once per (h, origin)
            gsc = StandardScaler().fit(Xtr_all)
            gmdl = Ridge(alpha=alpha).fit(gsc.transform(Xtr_all), ytr_all)
            pz = np.empty(len(te), dtype=float)
            te_series = te["series"].to_numpy()
            for c, members in membership.items():
                m_tr = tr["series"].isin(members).to_numpy()
                m_te = np.isin(te_series, list(members))
                if m_tr.sum() < 60:          # too thin -> use the global model
                    n_fallback += int(m_te.sum())
                    if m_te.any():
                        Xs = gsc.transform(H.matrix(te)[m_te])
                        pz[m_te] = gmdl.predict(Xs)
                    continue
                sc = StandardScaler().fit(Xtr_all[m_tr])
                mdl = Ridge(alpha=alpha).fit(sc.transform(Xtr_all[m_tr]), ytr_all[m_tr])
                if m_te.any():
                    pz[m_te] = mdl.predict(sc.transform(H.matrix(te)[m_te]))
            pz = _reconstruct(pz, te, target_mode)
            pred = H.inverse_transform(pz, te_series, scaling)
            for s, yp in zip(te_series, pred):
                rows.append((tag, o, h, s,
                             float(wide.loc[H.month_add(o, h), s]), float(yp)))
    df = pd.DataFrame(rows,
                      columns=["model", "origin", "h", "series", "y_true", "y_pred"])
    df.attrs["cluster_fallback_points"] = n_fallback
    return df


# ---------------------------------------------------------------------------
# 2) kNN
# ---------------------------------------------------------------------------
def run_knn(feat, wide, scaling, origins=H.ORIGINS, tag="kNN",
            target_mode="level", n_neighbors=15):
    """
    KNeighborsRegressor(n_neighbors=15, weights='distance') on the standardised
    H.matrix design.  Deterministic (no seed dependence): a query's neighbours
    are a deterministic function of the training rows.
    """
    from sklearn.neighbors import KNeighborsRegressor

    rows = []
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            Xtr = H.matrix(tr)
            Xte = H.matrix(te)
            ytr = _targets(tr, target_mode)
            sc = StandardScaler().fit(Xtr)
            mdl = KNeighborsRegressor(n_neighbors=n_neighbors,
                                      weights="distance").fit(sc.transform(Xtr), ytr)
            pz = _reconstruct(mdl.predict(sc.transform(Xte)), te, target_mode)
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s,
                             float(wide.loc[H.month_add(o, h), s]), float(yp)))
    return pd.DataFrame(rows,
                        columns=["model", "origin", "h", "series", "y_true", "y_pred"])


# ---------------------------------------------------------------------------
# 3) NLinear  (Zeng et al., AAAI 2023)
# ---------------------------------------------------------------------------
def _nlinear_design(df: pd.DataFrame) -> np.ndarray:
    """
    NLinear's normalisation: subtract the LAST observed value from every lag
    channel, so the linear map only has to learn the deviation, not the level.
    The five lag channels are the ones that live in y_z units; month dummies,
    LNY windows and regime dummies are not level-bearing and are left alone.
    """
    X = H.matrix(df)
    base = df["y_z"].to_numpy(float)
    X[:, _LAG_IDX] = X[:, _LAG_IDX] - base[:, None]
    return X


def run_nlinear(feat, wide, scaling, origins=H.ORIGINS, tag="NLinear",
                target_mode="level"):
    """
    NLinear: X' = lags - y_z(t);  fit OLS on the truth defined by target_mode.
      level    -> target = y_z(t+h)          (no anchor handed to the model)
      residual -> target = y_z(t+h) - y_z(t) (the paper's own form; +y_z(t) back)
    """
    from sklearn.linear_model import LinearRegression

    rows = []
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            Xtr = _nlinear_design(tr)
            Xte = _nlinear_design(te)
            ytr = _targets(tr, target_mode)
            mdl = LinearRegression().fit(Xtr, ytr)
            pz = _reconstruct(mdl.predict(Xte), te, target_mode)
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s,
                             float(wide.loc[H.month_add(o, h), s]), float(yp)))
    return pd.DataFrame(rows,
                        columns=["model", "origin", "h", "series", "y_true", "y_pred"])


# ---------------------------------------------------------------------------
# 4) DLinear
# ---------------------------------------------------------------------------
def _moving_average(x: np.ndarray, k: int = 3) -> np.ndarray:
    """centred moving average along axis 1 with edge replication ('same' size)"""
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), xp], axis=1), axis=1)
    return (c[:, k:] - c[:, :-k]) / float(k)


def _dlinear_design(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    """
    trend/seasonal split of ["y_z","lag1","lag2","lag3","lag12","lag13"].
    The six columns are already ordered newest -> oldest, so a centred moving
    average over them is a crude but honest trend smoother.
    Returns 12 columns: [trend(6) | seasonal(6)].
    """
    x6 = df[DLINEAR_COLS].to_numpy(float)
    trend = _moving_average(x6, k=k)
    return np.hstack([trend, x6 - trend])


def run_dlinear(feat, wide, scaling, origins=H.ORIGINS, tag="DLinear",
                target_mode="level", kernel_size=3):
    """
    DLinear: one linear map for the trend half, one for the seasonal half,
    summed:  yhat = W_trend . trend + W_seasonal . seasonal
    Solved jointly by OLS on the 12-column concatenation (equivalent to the
    two-map form above, with the split coefficients fitted coherently).
    """
    from sklearn.linear_model import LinearRegression

    def design(df):
        return _dlinear_design(df, k=kernel_size)

    rows = []
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            ytr = _targets(tr, target_mode)
            mdl = LinearRegression().fit(design(tr), ytr)
            pz = _reconstruct(mdl.predict(design(te)), te, target_mode)
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s,
                             float(wide.loc[H.month_add(o, h), s]), float(yp)))
    return pd.DataFrame(rows,
                        columns=["model", "origin", "h", "series", "y_true", "y_pred"])


# ---------------------------------------------------------------------------
# 5) LightGBM with a small time-series grid search
# ---------------------------------------------------------------------------
LGB_GRID = {
    "num_leaves": [15, 31],
    "learning_rate": [0.03, 0.1],
    "min_child_samples": [5, 20],
    "n_estimators": [200, 600],
}
LGB_FIXED = dict(objective="l2", subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4)


def tune_lightgbm(feat, n_val_origins: int = 24, seed: int = 42,
                  horizons=H.HORIZONS, grid=None, verbose=True):
    """
    Time-series validation INSIDE the training window only.

    For each horizon the validation origins are the last `n_val_origins`
    months whose TARGET month still satisfies target_period <= TRAIN_END
    (2023-12).  Every candidate is scored by rolling-origin refit at each of
    those origins: train on {target_period <= o}, predict {period == o}.
    2024 never enters any training slice or any model selection decision.

    Selection metric: pooled mean absolute error in y_z space (the same space
    the MASE denominator is a positive per-series constant in), computed on the
    RESIDUAL target form — the form RESULTS.md establishes as the defensible one.
    The selected hyper-parameters are then reused unchanged for BOTH target
    modes, so the level/residual contrast is not confounded by re-tuning.
    """
    import itertools

    import lightgbm as lgb

    grid = grid or LGB_GRID
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]

    chosen, audit = {}, []
    for h in horizons:
        sup = H.make_supervised(feat, h)
        sup = sup[sup["target_period"] <= H.TRAIN_END]     # hard train-only gate
        cand = sorted(sup["period"].unique())
        val_origins = cand[-n_val_origins:]
        # pre-slice once per origin so the 16 candidates reuse identical rows
        slices = []
        for o in val_origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            slices.append((H.matrix(tr), _targets(tr, "residual"),
                           H.matrix(te), _targets(te, "residual")))
        best, best_mae = None, np.inf
        for combo in combos:
            p = dict(LGB_FIXED, seed=seed, **combo)
            ae, n = 0.0, 0
            for Xtr, ytr, Xte, yte in slices:
                m = lgb.LGBMRegressor(**p).fit(Xtr, ytr)
                ae += np.abs(yte - m.predict(Xte)).sum()
                n += len(yte)
            sc = float(ae / max(n, 1))
            audit.append({"h": h, "n_val_origins": len(slices), "n_val_points": n,
                          "val_MAE_yz": round(sc, 6), **combo})
            if sc < best_mae:
                best_mae, best = sc, combo
            if verbose:
                print(f"    h={h:>2}  {combo}  val_MAE_yz={sc:.5f}")
        chosen[h] = dict(best, val_MAE_yz=round(best_mae, 6))
        if verbose:
            print(f"  -> h={h} selected {best}  val_MAE_yz={best_mae:.5f}")
    return chosen, pd.DataFrame(audit)


def run_lightgbm_tuned(feat, wide, scaling, origins=H.ORIGINS, tag="LightGBMTuned",
                       target_mode="level", params_by_h=None, seed=42):
    """run_lightgbm with per-horizon hyper-parameters from tune_lightgbm()."""
    import lightgbm as lgb

    if params_by_h is None:                     # tune on demand (train-only)
        params_by_h, _ = tune_lightgbm(feat, seed=seed, verbose=False)

    rows = []
    for h in H.HORIZONS:
        sel = {k: params_by_h[h][k] for k in LGB_GRID}
        p = dict(LGB_FIXED, seed=seed, **sel)
        sup = H.make_supervised(feat, h)
        for o in origins:
            tr = sup[sup["target_period"] <= o]
            te = sup[sup["period"] == o]
            if len(tr) < 50 or len(te) == 0:
                continue
            Xtr = H.matrix(tr)
            Xte = H.matrix(te)
            ytr = _targets(tr, target_mode)
            mdl = lgb.LGBMRegressor(**p).fit(Xtr, ytr)
            pz = _reconstruct(mdl.predict(Xte), te, target_mode)
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s,
                             float(wide.loc[H.month_add(o, h), s]), float(yp)))
    return pd.DataFrame(rows,
                        columns=["model", "origin", "h", "series", "y_true", "y_pred"])
