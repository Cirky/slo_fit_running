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

FEATURES = [
    # "height",
    # "weight",
    # "triceps_skinfold",
    # "arm_plate_tapping",
    # "broad_jump",
    # "polygon_backwards",
    # "situps_60s",
    # "sit_and_reach",
    # "bent_arm_hang",
    "dash_60m",
    #"run_600m"
]

TARGETS = [
    "dash_60m",
   # "run_600m",
    # "height",
]

MODEL_CONFIGS = [
    # ("LinearRegression", LinearRegression(),        True),
    #("RandomForestRegressor", RandomForestRegressor(), True),
    ("HistGBRegressor", HistGradientBoostingRegressor(), False),
    #("GBRegressor", GradientBoostingRegressor(), True),
    # ("XGBRegressor", XGBRegressor(
    #     tree_method='hist',
    #     device='cpu',
    # ), False)
]

DO_SWEEP_TARGET_AGES    = True
DO_SWEEP_FEATURE_COMBOS = False
DO_SWEEP_AGE_WINDOWS    = True
DO_SWEEP_AGE_WINDOWS_ONLY_FROM_8 = False

TARGET_AGES  = [18] #[9,10,11,12,13,14,15,16,17,18]               # ages to predict
FEATURE_KS   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]                     # size-k feature combos (must include target feature)
AGE_MIN      = 8
AGE_MAX      = 18                          # inclusive in JSON
WINDOW_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # contiguous window lengths

N_SPLITS     = 5
RANDOM_STATE = 42

JSON_PATH    = "slofit_refactored_37311.json" # "slofit_refactored_37311.json" slofit_refactored_height_weight_44060.json
RESULTS_CSV  = "experiment_results.csv"

# ===============================================


# ------------------ Data loading ------------------
def load_flatten(json_path: str, sexes: Iterable[int]) -> pd.DataFrame:
    """
    Load once: all FEATURE_POOL across ages 8..18 into columns '<feat>_<age>', plus 'sex'.
    No target columns are added here—y will be taken from the same columns.
    """
    with open(json_path) as f:
        data = json.load(f)

    recs = []
    for cid, info in data.items():
        if sexes and info["sex"] not in sexes:
            continue
        row = {"child_id": int(cid), "sex": info["sex"]}
        for feat in FEATURES:
            vals = info["data"][feat]  # length 11 for ages 8..18
            for age, v in zip(range(8, 19), vals):
                row[f"{feat}_{age}"] = np.nan if v is None else v
        recs.append(row)

    df = pd.DataFrame.from_records(recs).set_index("child_id")
    return df


# -------------- Experiment utilities --------------

def build_Xy(
    df: pd.DataFrame,
    features: List[str],
    ages: List[int],
    target: str,
    target_age: int,
    drop_null_features: bool
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Picks columns for X based on (features, ages).
    y is the '<target>_<target_age>' column (already present in df).
    If drop_null_features=True, drop rows with any NaN in X.
    Always drop rows with NaN in y.
    Returns (X, y, df_used) where df_used is X|y for counts/debugging.
    """
    X_cols = [f"{f}_{a}" for f in features for a in ages]
    y_col  = f"{target}_{target_age}"

    # Ensure y column exists (it should, because we loaded all <feat>_<age>)
    if y_col not in df.columns:
        raise ValueError(f"Column {y_col} not found. Ensure target_age in [8..18] and target in FEATURE_POOL.")

    # Start with rows that have a target
    used = df.dropna(subset=[y_col]).copy()

    # Select X
    X = used[X_cols].copy()
    y = used[y_col].copy()

    # Optional: drop rows with any missing feature
    if drop_null_features:
        mask = ~X.isna().any(axis=1)
        X, y = X[mask], y[mask]
        used = used.loc[X.index]

    return X, y, used


def cv_evaluate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    rank_by_column: str | None = None,  # e.g., f"{target}_{base_age}"
    test_split_handling: str = "all"  # "all" or "top" (top 25% only) or "bottom" (bottom 25% only)
) -> Dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2s, maes, sds = [], [], []

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # --- drop rows with NaN in last known age (rank_by_column), because its unreasonable to rank them ---
        X_tr = X_tr.dropna(subset=[rank_by_column])
        y_tr = y_tr.loc[X_tr.index]
        X_te = X_te.dropna(subset=[rank_by_column])
        y_te = y_te.loc[X_te.index]
        # print(" Does X have nans in rank_column?", X_te[rank_by_column].isna().any())
        # print("X has shape:", X_te.shape)

        # --- choose ranking series (current latest age or target-age) ---
        rank_series = X_te[rank_by_column] if rank_by_column is not None else y_te

        # print(rank_series.head())
        # print(len(rank_series))
        #
        # print(y_te.head())
        # print(len(y_te))

        # decide which part of the TEST fold to evaluate on
        if test_split_handling == "all":
            X_te_eval, y_te_eval = X_te, y_te
        else:
            n = len(y_te)
            if n == 0:
                continue
            k = max(1, int(np.floor(0.25 * n)))  # pick exactly 25% (at least 1)

            # order positions inside the TEST fold
            order = np.argsort(rank_series.to_numpy())  # ascending

            if test_split_handling == "top":
                pos_eval = order[:k]  # best k within TEST fold
            elif test_split_handling == "bottom":
                pos_eval = order[-k:]  # worst k within TEST fold
            else:
                raise ValueError("test_split_handling must be 'all', 'top', or 'bottom'.")

            # subset using POSITIONS inside the TEST fold
            X_te_eval = X_te.iloc[pos_eval]
            y_te_eval = y_te.iloc[pos_eval]

        # choose the right “ES” routine
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

        r2 = r2_score(y_te_eval, y_pred)
        mae = mean_absolute_error(y_te_eval, y_pred)
        sd  = np.std(y_te_eval - y_pred, ddof=0)
        r2s.append(r2); maes.append(mae); sds.append(sd)

    return {
        "r2_mean":  float(np.mean(r2s)),  "r2_std":  float(np.std(r2s, ddof=0)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes, ddof=0)),
        "sd_mean":  float(np.mean(sds)),  "sd_std":  float(np.std(sds, ddof=0)),
        "pi_95": float(np.mean(sds) * 1.96),  # 95% prediction interval approx
    }


# ---------------- Master runner ----------------
def _enumerate_feature_sets(feature_pool, do_feature_combos, ks, must_include):
    if do_feature_combos:
        out = []
        for k in ks:
            out += [combo for combo in itertools.combinations(feature_pool, k) if must_include in combo]
        return out
    else:
        return [tuple(feature_pool)]

def _enumerate_age_windows(age_min, target_age, do_age_windows, window_sizes):
    if do_age_windows:
        wins = []
        for L in window_sizes:
            if DO_SWEEP_AGE_WINDOWS_ONLY_FROM_8:
                start = age_min
                win = list(range(start, start + L))
                if win[-1] < target_age:
                    wins.append(win)
            else:
                for start in range(age_min, target_age - L + 1):
                    win = list(range(start, start + L))
                    if win[-1] < target_age:
                        wins.append(win)
        return wins
    else:
        return [list(range(age_min, target_age))]

def run_experiments(
    df_all: pd.DataFrame,
    sexes: Iterable[int],
    targets: Iterable[str],
    target_ages: Iterable[int],
    feature_pool: List[str],
    do_feature_combos: bool,
    ks: Iterable[int],
    do_age_windows: bool,
    window_sizes: Iterable[int],
    age_min: int,
    age_max: int,
    model_configs,
    n_splits: int,
    random_state: int,
    results_csv: str,
):
    rows = []
    tasks = []
    test_split_handling = "bottom"  # "all", "top", "bottom"

    # Precompute all candidate tasks for the top-level progress bar
    for sex in sexes:
        for target in targets:
            if target not in feature_pool:
                continue
            feat_sets = _enumerate_feature_sets(feature_pool, do_feature_combos, ks, must_include=target)
            for target_age in target_ages:
                age_windows = _enumerate_age_windows(age_min, target_age, do_age_windows, window_sizes)
                for features in feat_sets:
                    if target not in features:
                        continue
                    for ages in age_windows:
                        if not ages or max(ages) >= target_age:
                            continue
                        for (model_name, model, drop_null) in model_configs:
                            tasks.append((sex, target, target_age, features, ages, model_name, model, drop_null))

    if not tasks:
        print("No tasks generated. Check your toggles and ranges.")
        return

    pbar = tqdm(total=len(tasks), desc="Experiments", unit="combo")
    completed = 0

    for sex, target, target_age, features, ages, model_name, model, drop_null in tasks:
        df_sex = df_all[df_all.sex == sex]
        if df_sex.empty:
            pbar.update(1); continue

        # Build X, y for this combo
        try:
            X, y, used = build_Xy(
                df_sex,
                features=list(features),
                ages=ages,
                target=target,
                target_age=target_age,
                drop_null_features=drop_null
            )
        except ValueError:
            pbar.update(1); continue  # missing columns, skip

        n_rows = len(used)
        if n_rows < max(n_splits, 20):
            pbar.update(1); continue  # too small to CV

        # Evaluate
        summary = cv_evaluate_model(
            model_name, model,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            rank_by_column=f"{target}_{ages[-1]}",
            test_split_handling=test_split_handling
        )

        # pretty strings with "mean | prediction interval"
        mae_pi = f"{summary['mae_mean']:.1f} | {summary['pi_95']:.1f}"

        rows.append({
            "sex": sex,
            "model": model_name,
            # "drop_null": drop_null,
            "target": target,
            "target_age": target_age,
            # "features": ",".join(features),
            "ages": ",".join(map(str, ages)),
            # "n_features": len(features),
            # "n_ages": len(ages),
            # "n_rows": n_rows,
            # **summary,
            # "r2_mean_pm":  r2_pm,
            "mae_pi": mae_pi,
            # "sd_mean_pm":  sd_pm,
        })

        completed += 1
        pbar.set_postfix({
            "sex": sex,
            "target": f"{target}_{target_age}",
            "model": model_name,
            "n_rows": n_rows
        })
        pbar.update(1)

    pbar.close()

    # final write
    if rows:
        out = pd.DataFrame(rows)
        out.sort_values(
            ["sex", "target", "target_age", "model", "ages"],
            ascending=[True, True, True, True, True],
            inplace=True
        )
        out.to_csv(results_csv, index=False)
        print(f"\nSaved {len(out)} rows → {results_csv}")
    else:
        print("No results produced (check filters/space size).")


# -------------------- MAIN --------------------
def main():
    # 1) Load once for both sexes and all features/ages
    df_all = load_flatten(JSON_PATH, sexes=(1, 2))

    # 2) Run the sweeps you’ve enabled
    run_experiments(
        df_all=df_all,
        sexes=(1, 2),
        targets=TARGETS,
        target_ages=(TARGET_AGES if DO_SWEEP_TARGET_AGES else [18]),
        feature_pool=FEATURES,
        do_feature_combos=DO_SWEEP_FEATURE_COMBOS,
        ks=FEATURE_KS,
        do_age_windows=DO_SWEEP_AGE_WINDOWS,
        window_sizes=WINDOW_SIZES,
        age_min=AGE_MIN,
        age_max=AGE_MAX,
        model_configs=MODEL_CONFIGS,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        results_csv=RESULTS_CSV
    )

if __name__ == "__main__":
    main()
