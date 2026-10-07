"""Recompute supplementary age-18 inclusion distributions on the corrected grid."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from label_interpolated_values import _align_metric, _complete_annual_grid, _numeric_values


ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "SLOFIT_LONG_COMPLETE_anon_16.01.2019.csv"
INCLUDED = ROOT / "slofit_refactored_37311_corrected_labeled.json"
EXCLUDED_IDS = ROOT / "slofit_refactored_excluded_completeness_filter.json"
OUTPUT = ROOT / "supplementary_selection_bias_corrected.csv"
TARGETS = {"dash_60m": "@60m_dash", "run_600m": "@600m_run"}
EDGES = {"dash_60m": (70, 80, 90, 100, 110, 120),
         "run_600m": (100, 120, 140, 160, 180, 200)}


def category(target, value):
    return sum(value >= edge for edge in EDGES[target])


def main():
    with INCLUDED.open(encoding="utf-8") as stream:
        included = json.load(stream)
    with EXCLUDED_IDS.open(encoding="utf-8") as stream:
        excluded = json.load(stream)
    included_ids = set(included)
    excluded_ids = set(excluded)
    if included_ids & excluded_ids:
        raise ValueError("Included and excluded cohorts overlap")

    counts = defaultdict(int)
    for child in included.values():
        sex = int(child["sex"])
        for target in TARGETS:
            value = child["data"][target][-1]
            if value is not None:
                counts[target, sex, "included", category(target, value)] += 1

    pieces = []
    for chunk in pd.read_csv(
        RAW, usecols=["CROWD_ID", "Age", "Birth", "Date_measured", *TARGETS.values()],
        parse_dates=["Birth", "Date_measured"], chunksize=250_000, low_memory=False,
    ):
        part = chunk.loc[chunk.CROWD_ID.astype(str).isin(excluded_ids)]
        if not part.empty:
            pieces.append(part)
    raw = pd.concat(pieces, ignore_index=True)
    raw = raw.rename(columns={"Age": "age", "Birth": "birth_date",
                              "Date_measured": "measurement_date", **{v: k for k, v in TARGETS.items()}})
    raw = raw.loc[raw.age.between(6, 19)].copy()
    raw["AGE_YEARS"] = (raw.measurement_date - raw.birth_date).dt.total_seconds() / (365 * 24 * 3600)
    raw = raw.loc[raw.AGE_YEARS.between(6, 19)].copy()
    seen = set()
    for child_id, group in raw.groupby("CROWD_ID"):
        key = str(int(child_id))
        seen.add(key)
        aligned = _complete_annual_grid(group)
        ages = aligned.AGE_YEARS.to_numpy()
        sex = int(excluded[key]["sex"])
        for target in TARGETS:
            values, _ = _align_metric(ages, _numeric_values(aligned[target]))
            if values[-1] is not None:
                counts[target, sex, "excluded", category(target, values[-1])] += 1
    if seen != excluded_ids:
        raise ValueError(f"Missing raw records for {len(excluded_ids - seen)} excluded participants")

    rows = []
    for target in TARGETS:
        for sex in (1, 2):
            for cohort in ("included", "excluded"):
                total = sum(counts[target, sex, cohort, i] for i in range(7))
                for bin_index in range(7):
                    n = counts[target, sex, cohort, bin_index]
                    rows.append(dict(target=target, sex=sex, cohort=cohort,
                                     bin_index=bin_index, n=n, total=total,
                                     percentage=round(100 * n / total, 2)))
                print(target, sex, cohort, total, [counts[target, sex, cohort, i] for i in range(7)], flush=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
