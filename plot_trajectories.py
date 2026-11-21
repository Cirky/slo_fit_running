"""Recreate the seeded illustrative trajectory plots using the supplied data."""
from pathlib import Path
import argparse
import json
import random
from typing import Iterable,List,Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def plot_percentile_individuals_many(
    df: pd.DataFrame,
    events: Iterable[str] = ("dash_60m", "run_600m"),
    sexes: Iterable[int] = (1, 2),
    ages: List[int] = list(range(8, 19)),
    percentiles: Iterable[int] = (10, 50, 90),
    require_complete: bool = True,
    random_state: int = 42,
    top_k_candidates: int = 5,
    n_per_percentile: int = 1,
    figsize=(12, 8),
) -> Dict[str, Dict[int, Dict[int, List[int]]]]:
    """
    For each (sex, event) produce one plot line for each percentile in `percentiles`.

    - If n_per_percentile == 1:
        choose one random individual near that percentile (based on age-8 value)
        and plot their full trajectory.
    - If n_per_percentile > 1:
        choose `n_per_percentile` individuals near that percentile and plot
        the *average* trajectory across them.

    Layout:
      - 2x2 subplots
        (row 0, col 0): Male  60m
        (row 1, col 0): Female 60m
        (row 0, col 1): Male  600m
        (row 1, col 1): Female 600m

    Returns
    -------
    results : dict
        results[event][sex][percentile] = list of child_ids used.
        (List has length 1 if n_per_percentile == 1.)
    """
    # Set font for this plot only
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],   # comes with Matplotlib, no missing font errors
        "mathtext.fontset": "cm",         # use Computer Modern for math symbols
        "font.size": 13
    })

    np.random.seed(random_state)
    random.seed(random_state)

    results: Dict[str, Dict[int, Dict[int, List[int]]]] = {}

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    # Desired mapping: 60m left column, 600m right column
    # indices in axes.flatten():
    # 0: row 0, col 0
    # 1: row 0, col 1
    # 2: row 1, col 0
    # 3: row 1, col 1
    plot_idx = {
        (1, "dash_60m"): 0,  # male 60m  -> top-left
        (2, "dash_60m"): 2,  # female 60m -> bottom-left
        (1, "run_600m"): 1,  # male 600m -> top-right
        (2, "run_600m"): 3,  # female 600m -> bottom-right
    }

    colors = {10: "C1", 50: "C2", 90: "C3"}
    markers = {10: "o", 50: "s", 90: "D"}

    for ev in events:
        # Check that at least the first age column exists
        if not any(f"{ev}_{a}" in df.columns for a in ages):
            print(f"Skipping {ev}: missing columns for ages {min(ages)}..{max(ages)}")
            continue

        results[ev] = {}

        for sex in sexes:
            results[ev][sex] = {}

            df_sex = df[df.sex == sex].copy()
            if df_sex.empty:
                print(f"No records for sex={sex}, event={ev}")
                continue

            ev_cols = [f"{ev}_{a}" for a in ages]

            # Filter candidates
            if require_complete:
                df_candidates = df_sex.dropna(subset=ev_cols)
            else:
                # At least need age-8 value to rank by percentile
                df_candidates = df_sex[df_sex[f"{ev}_8"].notna()].copy()

            if df_candidates.empty:
                print(f"No candidates for sex={sex}, event={ev} after completeness filter")
                continue

            series_age8 = df_candidates[f"{ev}_8"].dropna()
            if series_age8.empty:
                print(f"No age-8 measurements for sex={sex}, event={ev}")
                continue

            # Percentile values based on age-8 distribution
            p_values = np.percentile(series_age8.to_numpy(), list(percentiles))

            for p, pval in zip(percentiles, p_values):
                # Absolute distance to target percentile value at age 8
                diffs = (df_candidates[f"{ev}_8"] - pval).abs()

                # Take top_k_candidates closest rows (or fewer if dataset is small)
                smallest_idx = diffs.nsmallest(min(top_k_candidates, len(diffs))).index.to_list()
                if not smallest_idx:
                    continue

                # Decide which child(ren) we use
                if n_per_percentile <= 1:
                    chosen_ids = [random.choice(smallest_idx)]
                else:
                    # Sample up to n_per_percentile from the closest candidates
                    k = min(n_per_percentile, len(smallest_idx))
                    chosen_ids = random.sample(smallest_idx, k=k)

                results[ev][sex][p] = [int(cid) for cid in chosen_ids]

                # Build trajectory: average if multiple children
                data_block = df_candidates.loc[chosen_ids, ev_cols].to_numpy(dtype=float)
                # shape: (n_persons, n_ages)
                if data_block.ndim == 1:
                    # single row
                    times = data_block
                else:
                    times = np.nanmean(data_block, axis=0)

                ages_arr = np.array(ages)
                ax = axes[plot_idx.get((sex, ev), 0)]

                # Legend label: no IDs, just percentile & time at age 8
                t8_vals = df_candidates.loc[chosen_ids, f"{ev}_8"].to_numpy(dtype=float)
                if t8_vals.size == 1:
                    t8_desc = f"{t8_vals[0]:.1f}"
                else:
                    t8_desc = f"{np.mean(t8_vals):.1f}"

                ax.plot(
                    ages_arr,
                    times,
                    marker=markers[p],
                    color=colors[p],
                    label=f"{p}th percentile at age 8",
                )

                # Titles and axis labels
                sex_label = "Boys" if sex == 1 else "Girls"
                event_label = "60-meter dash" if ev == "dash_60m" else "600-meter run"
                rand_samp = f" ({len(chosen_ids)} random samples)" if n_per_percentile > 1 else "(1 random sample)"
                ax.set_title(f"{sex_label} — {event_label} {rand_samp}")

                ax.set_xlabel("Age (years)")
                if ev == "dash_60m":
                    ax.set_ylabel("Time (deciseconds)")
                else:
                    ax.set_ylabel("Time (seconds)")

                ax.grid(True, linestyle="--", alpha=0.4)

    # Finalize legends and layout
    for ax in axes:
        ax.legend(fontsize=9)
        ax.set_xticks(ages)
        ax.set_xlim(min(ages) - 0.2, max(ages) + 0.2)


    plt.tight_layout()
    # Saved by the command-line wrapper below.

    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts'/'data'/'historical_feature_dataset.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'illustrations')
    args=parser.parse_args()
    records=json.loads(args.data.read_text())
    rows=[]
    for cid,r in records.items():
        row={'child_id':int(cid),'sex':r['sex']}
        for target in ('dash_60m','run_600m'):
            row.update({f'{target}_{a}':np.nan if v is None else v for a,v in zip(range(8,19),r['data'][target])})
        rows.append(row)
    frame=pd.DataFrame(rows).set_index('child_id')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for n in (1,10,100):
        plot_percentile_individuals_many(frame,top_k_candidates=100,n_per_percentile=n,random_state=42)
        plt.gcf().savefig(args.output_dir/f'random{n}.png',dpi=200)
        plt.close()
    print('Saved 1/10/100-sample illustrations from the supplied input dataset.')


if __name__=='__main__':main()
