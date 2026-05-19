import json
import time
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Dict, List, Optional

# -----------------------------
# I/O and data preparation
# -----------------------------
def load_feature_matrix(json_path: str, feature: str, sexes: Iterable[int]=(1,2)) -> pd.DataFrame:
    """
    Load one feature's aligned values for ages 8..18 into columns '<feature>_<age>' plus 'sex'.
    Rows are children (index = child_id). Values may include None.
    """
    with open(json_path) as f:
        data = json.load(f)

    recs = []
    for cid, info in data.items():
        sex = info.get("sex", None)
        if sex not in sexes:
            continue
        row = {"child_id": int(cid), "sex": sex}
        vals = info["data"][feature]
        for age, v in zip(range(8, 19), vals):
            row[f"{feature}_{age}"] = np.nan if v is None else float(v)
        recs.append(row)
    df = pd.DataFrame.from_records(recs).set_index("child_id")
    return df


def build_complete_matrix(df: pd.DataFrame, feature: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (matrix, mask) where matrix is (n_complete x 11) for ages 8..18 and
    mask is a boolean array indexing df rows used.
    """
    cols = [f"{feature}_{a}" for a in range(8, 19)]
    complete_mask = ~df[cols].isna().any(axis=1)
    mat = df.loc[complete_mask, cols].to_numpy(dtype=float)
    return mat, complete_mask.to_numpy()


# -----------------------------
# GCC core
# -----------------------------
def _topk_indices_desc(arr: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k values of 1D array arr in descending order (fast)."""
    if k >= arr.size:
        return np.argsort(-arr)
    part = np.argpartition(-arr, k-1)[:k]
    return part[np.argsort(-arr[part])]


def _cosine_sims(ref_prefix: np.ndarray, q_prefix: np.ndarray, ref_row_norms: np.ndarray, q_norm: float) -> np.ndarray:
    """
    Cosine similarity between each row in ref_prefix (n x L) and q_prefix (L,).
    ref_row_norms: precomputed row norms of ref_prefix.
    """
    # (R @ q) / (||R_i|| * ||q||)
    num = ref_prefix @ q_prefix
    denom = ref_row_norms * q_norm
    denom = np.where(denom == 0.0, 1e-12, denom)
    return num / denom


def gcc_predict_all(
    full_mat: np.ndarray,        # n x 11, complete curves 8..18
    obs_end_age: int,            # e.g., 14 means we observe 8..14
    target_age: int = 18,
    k_neighbors: int = 100
) -> np.ndarray:
    """
    Vectorized GCC predictions for every child in full_mat, excluding self as neighbor.

    Steps for each child i:
      1) compute cosine similarity on prefix 8..obs_end_age vs all other children
      2) take top-K neighbors
      3) average their year-to-year deltas from last observed age to target_age
      4) add cumulative mean growth to child's value at last observed age

    Returns: y_pred (n,) for target_age.
    """
    assert 8 <= obs_end_age < target_age <= 18
    L = obs_end_age - 8 + 1
    last_idx = L - 1
    target_idx = target_age - 8

    ref_prefix = full_mat[:, :L]                           # (n x L)
    ref_row_norms = np.linalg.norm(ref_prefix, axis=1)     # (n,)

    # Precompute neighbor deltas once (n x 10) for 8→9, ..., 17→18
    deltas = np.diff(full_mat, axis=1)                     # (n x 10)
    # Slice for path from last observed → target (same slice for all queries)
    delta_slice_cols = slice(last_idx, target_idx)         # e.g., last_idx=6 (14y), target_idx=10 (18y)
    # number of steps to accumulate can be zero (if already at target)
    # handle that below

    n = full_mat.shape[0]
    y_pred = np.empty(n, dtype=float)

    for i in range(n):
        q_prefix = ref_prefix[i]
        q_norm = np.linalg.norm(q_prefix)
        sims = _cosine_sims(ref_prefix, q_prefix, ref_row_norms, q_norm)

        # exclude self
        sims[i] = -np.inf

        k = min(k_neighbors, n-1)
        nbr_idx = _topk_indices_desc(sims, k)

        if target_idx == last_idx:
            # already at target age (nothing to add)
            y_pred[i] = full_mat[i, target_idx]
            continue

        nbr_path = deltas[nbr_idx, delta_slice_cols]   # (k x path_len)
        mean_growth_path = np.mean(nbr_path, axis=0)   # (path_len,)
        y_pred[i] = full_mat[i, last_idx] + float(np.sum(mean_growth_path))

    return y_pred


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_gcc_no_cv(
    df: pd.DataFrame,
    feature: str = "height",
    obs_end_ages: Iterable[int]=(9, 11, 13, 15, 17),
    target_age: int = 18,
    k_neighbors: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate GCC without CV, per sex, excluding self as neighbor.
    Returns:
      preds_df: per-child predictions for each obs_end_age
      summary : per-sex x obs_end_age metrics (R², MAE, residual SD)
    """
    results = []
    all_preds = []

    for sex in (1, 2):
        df_sex = df[df.sex == sex].copy()

        full_mat, used_mask = build_complete_matrix(df_sex, feature)
        if full_mat.shape[0] == 0:
            continue

        # ground-truth at target age
        y_true = full_mat[:, target_age - 8]
        child_ids = df_sex.index[used_mask].to_numpy(dtype=int)

        for obs_end_age in obs_end_ages:
            start_t = time.time()
            y_pred = gcc_predict_all(full_mat, obs_end_age=obs_end_age, target_age=target_age, k_neighbors=k_neighbors)
            elapsed = time.time() - start_t

            err = y_true - y_pred
            ss_res = float(np.sum(err**2))
            ss_tot = float(np.sum((y_true - y_true.mean())**2))
            r2  = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
            mae = float(np.mean(np.abs(err)))
            sd  = float(np.std(err, ddof=0))

            results.append({
                "sex": sex,
                "feature": feature,
                "k_neighbors": k_neighbors,
                "obs_end_age": obs_end_age,
                "target_age": target_age,
                "n_children": int(len(y_true)),
                "r2_mean": r2,
                "mae_mean": mae,
                "sd_mean": sd,
                "runtime_sec": elapsed
            })

            all_preds.append(pd.DataFrame({
                "child_id": child_ids,
                "sex": sex,
                "feature": feature,
                "obs_end_age": obs_end_age,
                "target_age": target_age,
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred
            }))

    summary = pd.DataFrame(results)
    preds_df = pd.concat(all_preds, axis=0, ignore_index=True) if all_preds else pd.DataFrame()
    return preds_df, summary


# -----------------------------
# Example main
# -----------------------------
def main():
    json_path  = "slofit_refactored_37311.json" # "slofit_refactored_37311.json"
    feature    = "run_600m"                 # GCC in the paper; can be any aligned series you have
    obs_list   = (9, 10, 11, 12, 13, 14, 15, 16, 17)      # observe 8..obs_end, predict at 18
    target_age = 18
    k_neigh    = 50

    df = load_feature_matrix(json_path, feature, sexes=(1,2))

    preds_df, summary = evaluate_gcc_no_cv(
        df, feature=feature,
        obs_end_ages=obs_list,
        target_age=target_age,
        k_neighbors=k_neigh
    )

    # Pretty print
    if not summary.empty:
        for _, row in summary.sort_values(["sex","obs_end_age"]).iterrows():
            print(f"GCC | sex={row.sex} | observe 8..{int(row.obs_end_age)} → {int(row.target_age)} | "
                  f"n={int(row.n_children)} | "
                  f"R²={row.r2_mean:.3f}, MAE={row.mae_mean:.2f}, SD(res)={row.sd_mean:.2f} | "
                  f"{row.runtime_sec:.2f}s")

    # Optional: save results
    # if not preds_df.empty:
    #     preds_df.to_csv("gcc_predictions.csv", index=False)
    # if not summary.empty:
    #     # add pretty columns like "mean ±" strings if you like
    #     summary["r2_mean_pm"]  = summary["r2_mean"].map(lambda v: f"{v:.3f}")
    #     summary["mae_mean_pm"] = summary["mae_mean"].map(lambda v: f"{v:.3f}")
    #     summary["sd_mean_pm"]  = summary["sd_mean"].map(lambda v: f"{v:.3f}")
    #     summary.to_csv("gcc_summary.csv", index=False)
    #     print("Saved: gcc_predictions.csv, gcc_summary.csv")

if __name__ == "__main__":
    main()