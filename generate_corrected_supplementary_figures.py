"""Rebuild the all-participant and fastest-quartile supplementary figures."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from helper_functions import GCCRegressor, PercentileRegressor
from training_automated import build_Xy, cv_evaluate_model, load_flatten


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "slofit_refactored_37311_corrected_labeled.json"
CURVES = ROOT / "supplementary_figures_corrected_no_injected.csv"
MODELS = ("LR", "HGB", "GB", "XGB", "Percentile", "GCC")
TARGETS = ("dash_60m", "run_600m")
GROUPS = ("all", "top")
FIELDS = ("group", "sex", "target", "cutoff_age", "model", "mae", "n_rows", "n_excluded_injected")


def in_sample_mae(name, X, y, target, cutoff, group):
    rank_col = f"{target}_{cutoff}"
    X = X.loc[X[rank_col].notna()]
    y = y.loc[X.index]
    if name == "GCC":
        model = GCCRegressor(target, cutoff, 18, k_neighbors=200)
    else:
        model = PercentileRegressor(target, 18, cutoff)
    model.fit(X, y)
    if group == "top":
        keep = X[rank_col] <= np.quantile(X[rank_col], 0.25)
        X, y = X.loc[keep], y.loc[keep]
    pred = model.predict(X)
    return float(np.mean(np.abs(y.to_numpy() - pred)))


def append_result(row):
    with CURVES.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def draw(curves):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13})
    specs = (
        ("all", "dash_60m", "60m.png", "60-meter dash", "deciseconds", "all individuals"),
        ("all", "run_600m", "600m.png", "600-meter run", "seconds", "all individuals"),
        ("top", "dash_60m", "fastest_60m.png", "60-meter dash", "deciseconds", "top 25% performers"),
        ("top", "run_600m", "fastest_600m.png", "600-meter run", "seconds", "top 25% performers"),
    )
    (ROOT / "figures").mkdir(exist_ok=True)
    for group, target, filename, title, unit, group_label in specs:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for sex, ax, sex_name in ((1, axes[0], "Boys"), (2, axes[1], "Girls")):
            subset = curves.loc[(curves.group == group) & (curves.target == target) & (curves.sex == sex)]
            for name in MODELS:
                series = subset.loc[subset.model == name].sort_values("cutoff_age")
                if series.cutoff_age.tolist() != list(range(8, 18)):
                    raise ValueError(f"Incomplete curve: {group} {target} {sex} {name}")
                ax.plot(series.cutoff_age, series.mae, marker="o", label=name)
            ax.set_title(f"{sex_name} {title} - {group_label}")
            ax.set_ylabel(f"Average error ({unit})")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=9, ncol=3)
        axes[1].set_xlabel("Age of the latest measurement")
        axes[1].set_xticks(range(8, 18))
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / filename, dpi=200)
        plt.close(fig)
        print(f"Saved {filename}", flush=True)


def main():
    df = load_flatten(str(DATA), sexes=(1, 2), measurements=TARGETS)
    done = set()
    if CURVES.exists():
        old = pd.read_csv(CURVES)
        done = set(zip(old.group, old.sex, old.target, old.cutoff_age, old.model))

    for group in GROUPS:
        for target in TARGETS:
            for sex in (1, 2):
                cohort = df.loc[df.sex == sex]
                for cutoff in range(8, 18):
                    for name in MODELS:
                        key = (group, sex, target, cutoff, name)
                        if key in done:
                            continue
                        if name == "GCC":
                            X, y, _, _ = build_Xy(
                                cohort, [target], list(range(8, 19)), target, 18, True,
                                exclude_injected_latest=False,
                            )
                            injected = cohort[f"{target}_{cutoff}__injected"].reindex(X.index)
                            excluded = int(injected.sum())
                            X, y = X.loc[~injected], y.loc[~injected]
                        else:
                            X, y, _, excluded = build_Xy(
                                cohort, [target], list(range(8, cutoff + 1)), target, 18,
                                name in ("LR", "GB"), injected_latest_features=[target],
                            )
                        if name in ("Percentile", "GCC"):
                            mae = in_sample_mae(name, X, y, target, cutoff, group)
                        else:
                            model = {
                                "LR": LinearRegression(),
                                "HGB": HistGradientBoostingRegressor(random_state=42),
                                "GB": GradientBoostingRegressor(random_state=42),
                                "XGB": XGBRegressor(tree_method="hist", device="cpu", random_state=42, n_jobs=4),
                            }[name]
                            metrics = cv_evaluate_model(
                                name, model, X, y, n_splits=5, random_state=42,
                                rank_by_column=f"{target}_{cutoff}", test_split_handling=group,
                            )
                            mae = float(metrics["mae_mean"])
                        append_result(dict(group=group, sex=sex, target=target, cutoff_age=cutoff,
                                           model=name, mae=mae, n_rows=len(X), n_excluded_injected=excluded))
                        print(f"{group} {target} sex={sex} age={cutoff} {name}: {mae:.3f}", flush=True)

    curves = pd.read_csv(CURVES)
    if len(curves) != len(GROUPS) * len(TARGETS) * 2 * 10 * len(MODELS):
        raise ValueError(f"Expected 480 curve points, found {len(curves)}")
    draw(curves)


if __name__ == "__main__":
    main()
