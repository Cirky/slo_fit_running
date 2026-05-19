from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone


# ======== HELPERS: model-specific “early stopping” ========
def fit_xgb_with_es(est: XGBRegressor, X, y, *, val_size=0.2, rounds=30, verbose=False):
    X_sub, X_val, y_sub, y_val = train_test_split(X, y, test_size=val_size, random_state=1)
    est = clone(est)
    # ensure a high cap; ES will pick the best round
    est.set_params(n_estimators=100, early_stopping_rounds=rounds)
    est.fit(X_sub, y_sub, eval_set=[(X_val, y_val)], verbose=verbose)
    return est  # use est.best_iteration on predict

def fit_histgb_with_es(est: HistGradientBoostingRegressor, X, y, *, val_fraction=0.2, n_no_change=20):
    est = clone(est)
    est.set_params(early_stopping=True, validation_fraction=val_fraction, n_iter_no_change=n_no_change)
    est.fit(X, y)  # it internally splits training using validation_fraction
    return est

def fit_gb_with_es(est: GradientBoostingRegressor, X, y, *, val_fraction=0.2, n_no_change=20, tol=1e-4):
    est = clone(est)
    est.set_params(n_iter_no_change=n_no_change, validation_fraction=val_fraction, tol=tol)
    est.fit(X, y)  # internal validation split
    return est

def fit_rf_with_oob_es(est: RandomForestRegressor, X, y, *, start=50, step=50, max_trees=1000, patience=2):
    """
    Heuristic “early stopping” for RF using OOB score.
    Grows trees in chunks until no OOB improvement for `patience` steps.
    Requires: bootstrap=True (default) and oob_score=True.
    """
    rf = clone(est)
    rf.set_params(warm_start=True, oob_score=True, n_estimators=start)
    best_oob, rounds_no_improve = -np.inf, 0
    while rf.n_estimators <= max_trees:
        rf.fit(X, y)
        oob = rf.oob_score_
        if oob > best_oob + 1e-6:
            best_oob, rounds_no_improve = oob, 0
            rf.set_params(n_estimators=rf.n_estimators + step)
        else:
            rounds_no_improve += 1
            if rounds_no_improve > patience:
                break
            rf.set_params(n_estimators=rf.n_estimators + step)
    # remove warm_start for stable prediction behavior
    rf.set_params(warm_start=False)
    return rf

def xgb_cv_early_stopping(
        X: pd.DataFrame,
        y: pd.Series,
        model,
        n_splits: int = 5,
        early_stopping_rounds: int = 30,
        params: dict = None,
):
    """
    k-fold CV with per-fold early stopping for XGBRegressor.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
    y : Series of shape (n_samples,)
    n_splits : int (default=5)
        Number of outer CV folds.
    early_stopping_rounds : int
        Patience for early stopping on the per-fold validation split.
    params : dict
        Base params for XGBRegressor (without n_estimators).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict of overall CV metrics: {
      'r2_mean', 'r2_std',
      'mae_mean','mae_std',
      'sd_mean', 'sd_std'
    }
    :param model: xgbregressor
    """

    # storage for each fold’s metrics
    r2_scores, mae_scores, sd_scores = [], [], []

    outer_cv = KFold(n_splits=n_splits, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # split train→subtrain/val for early stopping
        subtr_idx, val_idx = next(
            KFold(n_splits=4, shuffle=True).split(X_tr)
        )
        X_subtr = X_tr.iloc[subtr_idx]
        y_subtr = y_tr.iloc[subtr_idx]
        X_val = X_tr.iloc[val_idx]
        y_val = y_tr.iloc[val_idx]

        # fit with early stopping
        model.fit(
            X_subtr, y_subtr,
            eval_set=[(X_val, y_val)],
            # early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

        # predict on the outer test fold using the best_iteration
        y_pred = model.predict(
            X_te,
            iteration_range=(0, model.best_iteration)
        )

        # compute metrics
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        sd = np.std(y_te - y_pred, ddof=0)

        print(f"Fold {fold}: R²={r2:.3f}, MAE={mae:.2f}, SD(res)={sd:.2f}")

        r2_scores.append(r2)
        mae_scores.append(mae)
        sd_scores.append(sd)

    # aggregate
    return {
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores, ddof=0),
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores, ddof=0),
        "sd_mean": np.mean(sd_scores),
        "sd_std": np.std(sd_scores, ddof=0),
    }



def get_best_xgb_params(
    X,
    y,
    param_grid: Dict[str, Any] = None,
    cv: int = 5,
    scoring: str = "r2",
    n_jobs: int = -1,
    gpu: bool = True,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Perform grid‐search hyperparameter tuning for XGBRegressor.

    Parameters
    ----------
    X : array‐like or DataFrame of shape (n_samples, n_features)
        Training features.

    y : array‐like of shape (n_samples,)
        Training target.

    param_grid : dict, optional
        Dictionary with parameters names (str) as keys and lists of
        parameter settings to try as values.  If None, a sensible default grid is used.

    cv : int, default=5
        Number of cross‐validation folds.

    scoring : str, default="r2"
        Scoring metric to optimize.

    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    gpu : bool, default=True
        If True, uses the GPU-accelerated tree method.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    best_params : dict
        Parameter settings that gave the best results.

    best_score : float
        Best cross‐validation score achieved.
    """
    # 1) Default grid if none provided
    if param_grid is None:
        param_grid = {
            "max_depth":       [3, 5, 7],
            "min_child_weight":[1, 5, 10],
            "subsample":       [0.6, 0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
            "gamma":           [0, 0.1, 0.3],
            "reg_alpha":       [0, 0.1, 1],
            "reg_lambda":      [1, 5, 10],
            "learning_rate":   [0.01, 0.1, 0.2],
            "n_estimators":    [100, 300, 500]
        }

    # 2) Initialize the XGBRegressor
    tree_method = "hist"
    #predictor   = "gpu_predictor" if gpu else "cpu_predictor"
    xgb = XGBRegressor(
        tree_method=tree_method,
       # predictor=predictor,
        random_state=random_state,
        verbosity=0
    )

    # 3) Set up GridSearchCV
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False
    )

    # 4) Run the grid search
    grid.fit(X, y)

    # 5) Return best params and score
    return grid.best_params_, grid.best_score_

from sklearn.base import BaseEstimator, RegressorMixin

class PercentileRegressor(BaseEstimator, RegressorMixin):  # scikit-learn-compatible estimator interface
    """
    Same-percentile transport using the FULL dataset (no folds).

    For each row:
      p = percentile of X[target_name_base_age] within ALL rows at base_age
      y_hat = quantile p of ALL rows' target-age distribution

    Assumes the base-age column exists and is non-NaN for all rows.
    """

    def __init__(self, target_name: str, target_age: int, base_age: int):  # constructor with config
        self.target_name = target_name                                      # store target column prefix (e.g., "dash_60m")
        self.target_age  = target_age                                       # store target age T (e.g., 18)
        self.base_age    = base_age                                         # store fixed base age b (e.g., 17), must be < T

    def fit(self, X, y=None):                                               # fit builds the full-data reference distributions
        X = pd.DataFrame(X)                                                 # ensure X is a DataFrame for column access
        base_col = f"{self.target_name}_{self.base_age}"                    # name of base-age column, e.g., "dash_60m_17"
        if base_col not in X.columns:                                       # safety: verify base-age column exists
            raise ValueError(f"Missing base-age column '{base_col}' in X.")

        # Build FULL-DATA distributions (sorted) at base age and target age
        self.base_ref_ = np.sort(                                           # sorted values at base age (for percentile calc)
            pd.to_numeric(X[base_col], errors="coerce").dropna().to_numpy()
        )
        self.tgt_ref_  = np.sort(                                           # sorted values at target age (for quantile lookup)
            pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()
        )

        if self.base_ref_.size == 0 or self.tgt_ref_.size == 0:             # ensure both reference arrays are non-empty
            raise ValueError("Empty distribution for base or target age after cleaning.")

        return self                                                         # scikit-learn convention: return fitted estimator

    @staticmethod
    def _percentile(x, sorted_ref: np.ndarray) -> float:                    # compute empirical percentile of x in sorted_ref
        rank = np.searchsorted(sorted_ref, x, side="right")                 # index where x would be inserted to keep order
        p = rank / sorted_ref.size                                          # convert rank to proportion in [0,1]
        return float(np.clip(p, 1e-6, 1-1e-6))                              # clip away from exact 0/1 to avoid edge issues

    @staticmethod
    def _quantile(p, sorted_ref: np.ndarray) -> float:                      # linear quantile interpolation at percentile p
        n = sorted_ref.size                                                 # number of points in reference
        x = p * (n - 1)                                                     # continuous index into 0..n-1
        lo = int(np.floor(x)); hi = int(np.ceil(x))                         # neighboring integer indices
        if lo == hi: return float(sorted_ref[lo])                           # if exactly on an index, return that value
        w = x - lo                                                          # interpolation weight between lo and hi
        return float((1 - w) * sorted_ref[lo] + w * sorted_ref[hi])         # linear interpolation result

    def predict(self, X):                                                   # map each row to same-percentile at target age
        X = pd.DataFrame(X)                                                 # ensure DataFrame for column access
        base_col = f"{self.target_name}_{self.base_age}"                    # base-age column name
        vals = pd.to_numeric(X[base_col], errors="coerce").to_numpy()       # numeric base-age values for all rows
        ps = np.array([self._percentile(v, self.base_ref_) for v in vals], dtype=float)    # percentile at base age for each row
        preds = np.array([self._quantile(p, self.tgt_ref_) for p in ps], dtype=float)   # map percentile to target-age quantile

        return preds                                                        # return predictions as 1D float array


# ---- utilities (same as in GCC file) ----
def _topk_indices_desc(arr: np.ndarray, k: int) -> np.ndarray:
    if k >= arr.size:
        return np.argsort(-arr)
    part = np.argpartition(-arr, k-1)[:k]
    return part[np.argsort(-arr[part])]

def _cosine_sims(ref_prefix: np.ndarray, q_prefix: np.ndarray, ref_row_norms: np.ndarray, q_norm: float) -> np.ndarray:
    num = ref_prefix @ q_prefix
    denom = ref_row_norms * q_norm
    denom = np.where(denom == 0.0, 1e-12, denom)
    return num / denom

def gcc_predict_all(full_mat: np.ndarray, obs_end_age: int, target_age: int, k_neighbors: int) -> np.ndarray:
    assert 8 <= obs_end_age < target_age <= 18
    L = obs_end_age - 8 + 1
    last_idx   = L - 1
    target_idx = target_age - 8

    ref_prefix    = full_mat[:, :L]                       # (n x L)
    ref_row_norms = np.linalg.norm(ref_prefix, axis=1)    # (n,)
    deltas        = np.diff(full_mat, axis=1)             # (n x 10)
    path_cols     = slice(last_idx, target_idx)           # last observed -> target

    n = full_mat.shape[0]
    y_pred = np.empty(n, dtype=float)

    for i in range(n):
        q_prefix = ref_prefix[i]
        q_norm   = np.linalg.norm(q_prefix)
        sims     = _cosine_sims(ref_prefix, q_prefix, ref_row_norms, q_norm)
        sims[i]  = -np.inf  # exclude self

        k = min(k_neighbors, n-1)
        nbr_idx = _topk_indices_desc(sims, k)

        if target_idx == last_idx:
            y_pred[i] = full_mat[i, target_idx]  # already at target
            continue

        nbr_path = deltas[nbr_idx, path_cols]   # (k x path_len)
        mean_growth = np.mean(nbr_path, axis=0) # (path_len,)
        y_pred[i] = full_mat[i, last_idx] + float(np.sum(mean_growth))

    return y_pred

# ---- scikit-style wrapper ----
class GCCRegressor(BaseEstimator, RegressorMixin):
    """
    GCC as a scikit-style estimator so it can be used with your evaluate_in_sample(...).

    Parameters
    ----------
    target_name : str   e.g. "dash_60m" or "run_600m"
    obs_end_age : int   last observed age for similarity (acts like 'base_age' for ranking)
    target_age  : int   age to predict (default 18)
    k_neighbors : int   number of neighbors (default 100)
    """
    def __init__(self, target_name: str, obs_end_age: int, target_age: int = 18, k_neighbors: int = 100):
        self.target_name = target_name
        self.obs_end_age = obs_end_age
        self.target_age  = target_age
        self.k_neighbors = k_neighbors

        # for compatibility with your evaluator:
        self.base_age     = obs_end_age  # so evaluate_in_sample can rank by f"{target_name}_{base_age}"
        self.rank_column  = f"{target_name}_{obs_end_age}"

    def _cols(self):
        # we require complete series for ages 8..18 for the target_name
        return [f"{self.target_name}_{a}" for a in range(8, self.target_age + 1)]

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self._cols()
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"GCCRegressor: missing columns {missing}")

        # keep only rows with a complete curve 8..18 for this target
        complete_mask = ~X[cols].isna().any(axis=1)
        self._used_index_ = X.index[complete_mask]
        if len(self._used_index_) < 2:
            raise ValueError("GCCRegressor: not enough complete rows to run GCC.")

        self._full_mat_ = X.loc[self._used_index_, cols].to_numpy(dtype=float)
        # precompute predictions for ALL used rows (excludes self internally)
        self._y_pred_all_ = gcc_predict_all(
            self._full_mat_,
            obs_end_age=self.obs_end_age,
            target_age=self.target_age,
            k_neighbors=self.k_neighbors
        )
        # a fast index → position map for predict
        self._pos_ = pd.Series(range(len(self._used_index_)), index=self._used_index_)
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        # intersect with the set of rows we actually computed GCC for
        idx = X.index.intersection(self._used_index_)
        if len(idx) == 0:
            # return all-NaN aligned to incoming X
            return pd.Series(np.nan, index=X.index).to_numpy()

        pos = self._pos_.loc[idx].to_numpy()
        preds = pd.Series(self._y_pred_all_[pos], index=idx)
        # reindex to X: rows not present in GCC (e.g., incomplete curves) → NaN
        return preds.reindex(X.index).to_numpy()