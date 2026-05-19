import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Iterable, Dict

from matplotlib.ticker import MultipleLocator
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection   import KFold, train_test_split
from sklearn.metrics           import r2_score, mean_absolute_error

from xgboost import XGBRegressor
from helper_functions import get_best_xgb_params, xgb_cv_early_stopping, fit_xgb_with_es, fit_histgb_with_es, \
    fit_gb_with_es, fit_rf_with_oob_es, PercentileRegressor, GCCRegressor
from sklearn.base import clone
import matplotlib.pyplot as plt

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
    # "run_600m"
]

TARGETS = [
    "dash_60m",
    # "run_600m",
    # "height",
]

# -------------- Model configurations --------------
# Each tuple: (name, model_instance, drop_null_features, use_all_features)
MODEL_CONFIGS = [
    ("LR", LinearRegression(),        True, False),
    # ("LR_features", LinearRegression(),        True, True),
    # # ("RandomForestRegressor", RandomForestRegressor(), True),
    ("HGB", HistGradientBoostingRegressor(), False, False),
    # ("HGB_features", HistGradientBoostingRegressor(), False, True),
    ("GB", GradientBoostingRegressor(), True, False),
    ("XGB", XGBRegressor(
        tree_method='hist',
        device='cpu',
    ), False, False),
    ("Percentile", None, False, False),  # special case, handled in code
    ("GCC", None, True, False),
]


# # ─── DATA LOADING ───────────────────────────────────────────────
# def load_flatten(path: str, drop_null: bool=False, target_age: int=18) -> pd.DataFrame:
#     with open(path) as f:
#         data = json.load(f)
#
#     recs = []
#     for cid, info in data.items():
#         base = {"child_id": int(cid), "sex": info["sex"]}
#         for feat in FEATURES:
#             vals = info["data"][feat]
#             for age, v in zip(range(8, 19), vals):
#                 base[f"{feat}_{age}"] = np.nan if v is None else v
#         for tgt in TARGETS:
#             base[f"{tgt}_{target_age}"] = info["data"][tgt][-1]
#         recs.append(base)
#
#     df = pd.DataFrame.from_records(recs).set_index("child_id")
#     df = df.dropna(subset=[f"{t}_{target_age}" for t in TARGETS])
#     if drop_null:
#         feat_cols = [f"{f}_{a}" for f in FEATURES for a in range(8,18)]
#         before = len(df)
#         df = df.dropna(subset=feat_cols)
#         # print(f"Dropped {before-len(df)} rows → {len(df)} remain")
#     return df
#
# def build_Xy(df: pd.DataFrame, target: str, target_age: int) -> Tuple[pd.DataFrame, pd.Series]:
#     X = df[[f"{f}_{a}" for f in FEATURES for a in range(8,target_age)]]
#     y = df[f"{target}_{target_age}"]
#     return X, y



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
    ages: List[int], # ages to include in X (e.g. [8..17] for target_age=18)
    target: str,
    target_age: int,
    drop_null_features: bool,
    feature_type: str = "limited" # all - all measurements, limited - only target_age and target_age-1, some - all except target_age -> for all non run features
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Picks columns for X based on (features, ages).
    y is the '<target>_<target_age>' column (already present in df).
    If drop_null_features=True, drop rows with any NaN in X.
    Always drop rows with NaN in y.
    Returns (X, y, df_used) where df_used is X|y for counts/debugging.
    """

    X_cols: List[str] = []
    if feature_type == "some":
        X_cols = [f"{f}_{a}" for f in features for a in ages]
    elif feature_type == "all":
        for f in features:
            if f in ["dash_60m", "run_600m"]:
                X_cols += [f"{f}_{a}" for a in ages]
            else:
                X_cols += [f"{f}_{a}" for a in ages + [target_age]]
    elif feature_type == "limited":
        for f in features:
            if f in ["dash_60m", "run_600m"]:
                X_cols += [f"{f}_{a}" for a in ages]
            else:
                X_cols += [f"{f}_{a}" for a in [target_age-1, target_age]]
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


# ======== GENERIC K-FOLD CV ========
def cv_evaluate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    rank_by_column: str | None = None,  # e.g., f"{target}_{base_age}"
    test_split_handling: str = "all"  # "all" or "top" (top 25% only) or "bottom" (bottom 25% only)
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2s, maes, sds = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
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

        # X_te_eval = X_te
        # y_te_eval = y_te

        # choose the right “ES” routine
        if isinstance(model, XGBRegressor):
            est = fit_xgb_with_es(model, X_tr, y_tr, val_size=0.2, rounds=30, verbose=False)
            y_pred = est.predict(X_te_eval) #, iteration_range=(0, est.best_iteration + 1))
        elif isinstance(model, HistGradientBoostingRegressor):
            est = fit_histgb_with_es(model, X_tr, y_tr, val_fraction=0.2, n_no_change=30)
            y_pred = est.predict(X_te_eval)
        elif isinstance(model, GradientBoostingRegressor):
            est = fit_gb_with_es(model, X_tr, y_tr, val_fraction=0.2, n_no_change=30)
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
        # print(f"{name} | Fold {fold}: R²={r2:.3f}, MAE={mae:.2f}, SD(res)={sd:.2f}")

        r2s.append(r2); maes.append(mae); sds.append(sd)

    # print("Features have shape:", X.shape)
    # print(X.iloc[0])

    print(f"\n{name} | CV summary:")
    print(f" r2_mean:  {np.mean(r2s):.3f} ± {np.std(r2s, ddof=0):.3f}")
    print(f" mae_mean: {np.mean(maes):.3f} ± {np.std(maes, ddof=0):.3f}")
    print(f" sd_mean:  {np.mean(sds):.3f} ± {np.std(sds, ddof=0):.3f}")
    return float(np.mean(maes))

def evaluate_in_sample(name: str, model, X, y, test_split_handling):
    """Evaluate for all data because its percentile method"""

    model.fit(X, y)

    X_filtered = X.dropna(subset=[f"{model.target_name}_{model.base_age}"])
    y_filtered = y.loc[X_filtered.index]

    rank_column = f"{model.target_name}_{model.base_age}"
    rank_series = X_filtered[rank_column] if rank_column is not None else y_filtered

    print("Does X have nans in rank_column?", X_filtered[rank_column].isna().any())
    print("X has shape:", X_filtered.shape)

    # --- decide which part of the TEST fold to evaluate on (times: lower is better) ---
    if test_split_handling == "all":
        mask = np.ones(len(X_filtered), dtype=bool)
    elif test_split_handling == "top":
        q25 = np.quantile(rank_series, 0.25)
        mask = (rank_series <= q25)  # fastest quartile
    elif test_split_handling == "bottom":
        q75 = np.quantile(rank_series, 0.75)
        mask = (rank_series >= q75)  # slowest quartile
    else:
        raise ValueError("test_split_handling must be 'all', 'top', or 'bottom'.")

    # subset test fold BEFORE predicting
    X_te = X_filtered.loc[mask]
    y_te = y_filtered.loc[mask]

    ### drop rows with NaN in X
    base_col = f"{model.target_name}_{model.base_age}"
    mask = X_te[base_col].notna()

    y_pred = model.predict(X_te.loc[mask])
    y_eval = y_te.loc[mask]

    r2  = r2_score(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    sd  = np.std(y_eval - y_pred, ddof=0)
    # print(f"{name} | In-sample: R²={r2:.3f}, MAE={mae:.2f}, SD(res)={sd:.2f}")
    return float(mae)


def evaluate_GCC(model, X, y, test_split_handling):
    """Evaluate for all data because its GCC method"""

    model.fit(X, y)

    X_filtered = X.dropna(subset=[f"{model.target_name}_{model.obs_end_age}"])
    y_filtered = y.loc[X_filtered.index]

    rank_column = f"{model.target_name}_{model.obs_end_age}"
    rank_series = X_filtered[rank_column] if rank_column is not None else y_filtered

    print("Does X have nans in rank_column?", X_filtered[rank_column].isna().any())
    print("X has shape:", X_filtered.shape)

    # --- decide which part of the TEST fold to evaluate on (times: lower is better) ---
    if test_split_handling == "all":
        mask = np.ones(len(X_filtered), dtype=bool)
    elif test_split_handling == "top":
        q25 = np.quantile(rank_series, 0.25)
        mask = (rank_series <= q25)  # fastest quartile
    elif test_split_handling == "bottom":
        q75 = np.quantile(rank_series, 0.75)
        mask = (rank_series >= q75)  # slowest quartile
    else:
        raise ValueError("test_split_handling must be 'all', 'top', or 'bottom'.")

    # subset test fold BEFORE predicting
    X_te = X_filtered.loc[mask]
    y_te = y_filtered.loc[mask]

    ### drop rows with NaN in X
    base_col = f"{model.target_name}_{model.base_age}"
    mask = X_te[base_col].notna()

    y_pred = model.predict(X_te.loc[mask])
    y_eval = y_te.loc[mask]

    r2 = r2_score(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    sd = np.std(y_eval - y_pred, ddof=0)
    # print(f"{name} | In-sample: R²={r2:.3f}, MAE={mae:.2f}, SD(res)={sd:.2f}")
    return float(mae)


def plot_age_sweep(curves_boys: dict, curves_girls: dict,
                   base_ages: list[int], ylabel: str = "Average error (MAE)", title_add: str = ""):
    """
    curves_*: dict like {"LinearRegression": [mae@8, mae@9, ...], "PercentileMethod": [...], ...}
    base_ages: list of ages used on x-axis
    """

    # Set font for this plot only
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],   # comes with Matplotlib, no missing font errors
        "mathtext.fontset": "cm",         # use Computer Modern for math symbols
        "font.size": 13
    })

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    titles = ["Boys", "Girls"]
    for ax, curves, title in zip(axes, [curves_boys, curves_girls], titles):
        for name, ys in curves.items():
            ax.plot(base_ages, ys, marker='o', label=name)
        ax.set_title(title + title_add)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=10)
    axes[-1].set_xlabel("Age of the latest measurement")
    plt.tight_layout()
    plt.show()


def main():
    json_path = "slofit_refactored_37311.json"
    target_age = 18
    start_age = 8
    sexes      = [1, 2]
    test_split_handling = "bottom"  # "all", "top", "bottom"

    # load once
    df = load_flatten(json_path, sexes=sexes)

    # Ages to sweep: 8..17 (latest available before T)
    BASE_AGES = list(range(start_age, target_age))

    # storage for plot
    curves_boys  = {name: [] for name, _, _, _ in MODEL_CONFIGS}
    curves_girls = {name: [] for name, _, _, _ in MODEL_CONFIGS}

    for sex, curves in [(1, curves_boys), (2, curves_girls)]:
        df_sex = df[df.sex == sex]
        print(f"\nSex={sex}, records: {len(df_sex)}")
        if df_sex.empty:
            continue

        for base_age in BASE_AGES:
            print(f"  Base age = {base_age}")
            for name, mdl, drop_null, use_features in MODEL_CONFIGS:
                for tgt in TARGETS:
                    # Build X/y:
                    if name == "GCC":
                        X, y, used = build_Xy(
                            df_sex,
                            features=FEATURES if use_features else [tgt],
                            ages=list(range(start_age, target_age + 1)),
                            target=tgt,
                            target_age=target_age,
                            drop_null_features=drop_null,
                            feature_type="some"
                        )
                    else:
                        # Percentile uses only the base-age column for the target series,
                        # but we can still construct X with the history 8..base_age.
                        X, y, used = build_Xy(
                            df_sex,
                            features=FEATURES if use_features else [tgt],
                            ages=list(range(start_age, base_age + 1)),  # history start_age..base_age
                            target=tgt,
                            target_age=target_age,
                            drop_null_features=drop_null,
                            feature_type="some"
                        )

                    if name == "Percentile":
                        model = PercentileRegressor(target_name=tgt, target_age=target_age, base_age=base_age)
                        mae = evaluate_in_sample(name, model, X, y, test_split_handling)
                    elif name == "GCC":
                        model = GCCRegressor(
                            target_name=tgt,
                            obs_end_age=base_age,
                            target_age=target_age,
                            k_neighbors=200
                        )
                        mae = evaluate_GCC(model, X, y, test_split_handling)
                    else:
                        mae = cv_evaluate_model(name, mdl, X, y, n_splits=5, random_state=42, rank_by_column=f"{tgt}_{base_age}", test_split_handling=test_split_handling)

                    curves[name].append(mae)  # store MAE for this base_age

    # plot
    title = ""
    target_run = "600-meter run" if "run_600m" in TARGETS else "60-meter dash"
    metric = "seconds" if "run_600m" in TARGETS else "deciseconds"
    if test_split_handling == "all":
        title = f" {target_run}"
    elif test_split_handling == "top":
        title = f" {target_run} - top 25% performers"
    elif test_split_handling == "bottom":
        title = f" {target_run} - bottom 25% performers"
    plot_age_sweep(curves_boys, curves_girls, BASE_AGES, ylabel=f"Average error ({metric})", title_add=title)

if __name__ == "__main__":
    main()