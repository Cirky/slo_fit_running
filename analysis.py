"""Revised-study loading, eligibility and CV functions (original numerical code)."""
import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Iterable, Dict
import itertools
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection   import KFold, train_test_split
from sklearn.metrics           import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from helper_functions import get_best_xgb_params, xgb_cv_early_stopping, fit_xgb_with_es, fit_histgb_with_es, fit_gb_with_es, fit_rf_with_oob_es
from sklearn.base import clone
from tqdm.auto import tqdm

FEATURES = []
FEATURES_FUTURE = []
TARGETS = ["dash_60m", "run_600m"]
EXCLUDE_INJECTED_LATEST = True

def load_flatten(
    json_path: str,
    sexes: Iterable[int],
    measurements: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Load once: requested measurements across ages 8..18 into columns '<feat>_<age>', plus 'sex'.
    Target measurements are loaded even when they are not used as model features.
    """
    with open(json_path) as f:
        data = json.load(f)

    measurements_to_load = list(dict.fromkeys(measurements or [*FEATURES, *FEATURES_FUTURE, *TARGETS]))

    recs = []
    for cid, info in data.items():
        if sexes and info["sex"] not in sexes:
            continue
        row = {"child_id": int(cid), "sex": info["sex"]}
        for feat in measurements_to_load:
            vals = info["data"][feat]  # length 11 for ages 8..18
            injected = info.get("injected", {}).get(feat)
            if EXCLUDE_INJECTED_LATEST and injected is None:
                raise ValueError(
                    f"Missing injected labels for {feat!r} in child {cid}. "
                    "Use the labeled refactored dataset."
                )
            if injected is None:
                injected = [False] * len(vals)
            if len(injected) != len(vals):
                raise ValueError(f"Mismatched data and injected lengths for {feat!r} in child {cid}.")
            for age, v, was_injected in zip(range(8, 19), vals, injected):
                row[f"{feat}_{age}"] = np.nan if v is None else v
                row[f"{feat}_{age}__injected"] = bool(was_injected)
        recs.append(row)

    df = pd.DataFrame.from_records(recs).set_index("child_id")
    return df

def build_Xy(
    df: pd.DataFrame,
    features: List[str],
    ages: List[int],
    target: str,
    target_age: int,
    drop_null_features: bool,
    future_features: List[str] | None = None,
    exclude_injected_latest: bool = EXCLUDE_INJECTED_LATEST,
    injected_latest_features: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]:
    """
    Picks columns for X based on (features, ages).
    Also adds '<future_feature>_<target_age>' columns for any future_features.
    y is the '<target>_<target_age>' column (already present in df).
    If drop_null_features=True, drop rows with any NaN in X.
    Always drop rows with NaN in y.
    When requested, excludes rows where any selected feature at the latest
    predictor age was produced by missing-value interpolation.
    Returns (X, y, df_used, n_excluded_injected).
    """
    future_features = future_features or []
    X_cols = [f"{f}_{a}" for f in features for a in ages]
    X_cols += [f"{f}_{target_age}" for f in future_features]
    X_cols = list(dict.fromkeys(X_cols))
    y_col  = f"{target}_{target_age}"

    # Ensure y column exists (it should, because we loaded all <feat>_<age>)
    if y_col not in df.columns:
        raise ValueError(f"Column {y_col} not found. Ensure target_age in [8..18] and target in FEATURE_POOL.")

    # Start with rows that have a target
    used = df.dropna(subset=[y_col]).copy()

    n_excluded_injected = 0
    if exclude_injected_latest:
        latest_age = max(ages)
        required_latest_features = list(dict.fromkeys(injected_latest_features or features))
        injected_cols = [
            f"{feature}_{latest_age}__injected" for feature in required_latest_features
        ]
        missing_label_cols = [column for column in injected_cols if column not in used.columns]
        if missing_label_cols:
            raise ValueError(f"Missing injected-label columns: {missing_label_cols}")
        injected_mask = used[injected_cols].any(axis=1)
        n_excluded_injected = int(injected_mask.sum())
        used = used.loc[~injected_mask].copy()

    # Select X
    X = used[X_cols].copy()
    y = used[y_col].copy()

    # Optional: drop rows with any missing feature
    if drop_null_features:
        mask = ~X.isna().any(axis=1)
        X, y = X[mask], y[mask]
        used = used.loc[X.index]

    return X, y, used, n_excluded_injected

def cv_evaluate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    rank_by_column: str | None = None,  # e.g., f"{target}_{base_age}"
    rank_values: pd.Series | None = None,
    test_split_handling: str = "all"  # "all" or "top" (top 25% only) or "bottom" (bottom 25% only)
) -> Dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2s, maes, rmses, relative_error_pcts, sds = [], [], [], [], []
    if rank_values is not None:
        rank_values = rank_values.reindex(X.index)

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        rank_tr = None
        rank_te = None
        if rank_values is not None:
            rank_tr = rank_values.iloc[tr_idx]
            rank_te = rank_values.iloc[te_idx]
        elif rank_by_column is not None:
            if rank_by_column not in X.columns:
                if test_split_handling != "all":
                    raise ValueError(f"Rank column {rank_by_column} is not available.")
            else:
                rank_tr = X_tr[rank_by_column]
                rank_te = X_te[rank_by_column]

        # --- drop rows with NaN in last known target value, because its unreasonable to rank them ---
        if rank_te is not None:
            train_mask = rank_tr.notna()
            X_tr = X_tr.loc[rank_tr.index[train_mask]]
            y_tr = y_tr.loc[X_tr.index]

            test_mask = rank_te.notna()
            X_te = X_te.loc[rank_te.index[test_mask]]
            y_te = y_te.loc[X_te.index]
            rank_te = rank_te.loc[X_te.index]

        # print(rank_series.head())
        # print(len(rank_series))
        #
        # print(y_te.head())
        # print(len(y_te))

        # decide which part of the TEST fold to evaluate on
        if test_split_handling == "all":
            X_te_eval, y_te_eval = X_te, y_te
        else:
            if rank_te is None:
                raise ValueError("rank_values or rank_by_column is required for top/bottom test-split handling.")
            n = len(y_te)
            if n == 0:
                continue
            k = max(1, int(np.floor(0.25 * n)))  # pick exactly 25% (at least 1)

            # order positions inside the TEST fold
            order = np.argsort(rank_te.to_numpy())  # ascending

            if test_split_handling == "top":
                pos_eval = order[:k]  # best k within TEST fold
            elif test_split_handling == "bottom":
                pos_eval = order[-k:]  # worst k within TEST fold
            else:
                raise ValueError("test_split_handling must be 'all', 'top', or 'bottom'.")

            # subset using POSITIONS inside the TEST fold
            X_te_eval = X_te.iloc[pos_eval]
            y_te_eval = y_te.iloc[pos_eval]

        # choose the right â€śESâ€ť routine
        if isinstance(model, XGBRegressor):
            est = fit_xgb_with_es(model, X_tr, y_tr, val_size=0.2, rounds=40, verbose=False)
            y_pred = est.predict(X_te_eval, iteration_range=(0, est.best_iteration + 1))
        elif isinstance(model, HistGradientBoostingRegressor):
            est = fit_histgb_with_es(model, X_tr, y_tr, val_fraction=0.2, n_no_change=20)
            y_pred = est.predict(X_te_eval)
        elif isinstance(model, GradientBoostingRegressor):
            est = fit_gb_with_es(model, X_tr, y_tr, val_fraction=0.2, n_no_change=15)
            y_pred = est.predict(X_te_eval)
        elif isinstance(model, RandomForestRegressor):
            est = fit_rf_with_oob_es(model, X_tr, y_tr, start=100, step=50, max_trees=1200, patience=2)
            y_pred = est.predict(X_te_eval)
        else:
            est = clone(model)
            est.fit(X_tr, y_tr)  # no ES available
            y_pred = est.predict(X_te_eval)

        errors = y_te_eval - y_pred
        r2 = r2_score(y_te_eval, y_pred)
        mae = mean_absolute_error(y_te_eval, y_pred)
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        abs_actual = np.abs(np.asarray(y_te_eval, dtype=float))
        abs_errors = np.abs(np.asarray(errors, dtype=float))
        relative_error = np.divide(
            abs_errors,
            abs_actual,
            out=np.full_like(abs_errors, np.nan, dtype=float),
            where=abs_actual != 0,
        )
        relative_error_pct = float(np.nanmean(relative_error) * 100)
        sd  = np.std(errors, ddof=0)
        r2s.append(r2)
        maes.append(mae)
        rmses.append(rmse)
        relative_error_pcts.append(relative_error_pct)
        sds.append(sd)

    return {
        "r2_mean":  float(np.mean(r2s)),  "r2_std":  float(np.std(r2s, ddof=0)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes, ddof=0)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses, ddof=0)),
        "relative_error_pct_mean": float(np.mean(relative_error_pcts)),
        "relative_error_pct_std": float(np.std(relative_error_pcts, ddof=0)),
        "sd_mean":  float(np.mean(sds)),  "sd_std":  float(np.std(sds, ddof=0)),
        "pi_95": float(np.mean(sds) * 1.96),  # 95% prediction interval approx
    }
