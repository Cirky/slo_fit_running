"""Rebuild a cohort on a complete annual grid and label interpolated values.

The source JSON supplies the fixed participant cohort. Missing annual rows are
inserted before interpolation, so only an isolated missing year can be filled.
The added ``injected`` arrays mark values that depend on this missing-value
interpolation. Birthday alignment propagates those labels but does not itself
create an injected label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "height",
    "weight",
    "triceps_skinfold",
    "arm_plate_tapping",
    "broad_jump",
    "polygon_backwards",
    "situps_60s",
    "sit_and_reach",
    "bent_arm_hang",
    "dash_60m",
    "run_600m",
]

RAW_COLUMNS = [
    "CROWD_ID",
    "Age",
    "Birth",
    "Date_measured",
    "Height",
    "Weight",
    "Triceps_sf",
    "Arm_plate_tapping",
    "Standing_broad_jump",
    "Polygon_backwards",
    "Sit_ups_60s",
    "Stand_and_reach",
    "Bent_arm_hang",
    "@60m_dash",
    "@600m_run",
]

RENAME_COLUMNS = {
    "Age": "age",
    "Birth": "birth_date",
    "Date_measured": "measurement_date",
    "Height": "height",
    "Weight": "weight",
    "Triceps_sf": "triceps_skinfold",
    "Arm_plate_tapping": "arm_plate_tapping",
    "Standing_broad_jump": "broad_jump",
    "Polygon_backwards": "polygon_backwards",
    "Sit_ups_60s": "situps_60s",
    "Stand_and_reach": "sit_and_reach",
    "Bent_arm_hang": "bent_arm_hang",
    "@60m_dash": "dash_60m",
    "@600m_run": "run_600m",
}


def _numeric_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _complete_annual_grid(child_rows: pd.DataFrame) -> pd.DataFrame:
    """Insert absent nominal ages as missing rows before interpolation."""
    child_rows = child_rows.sort_values("AGE_YEARS")
    observed_age_offset = (child_rows["AGE_YEARS"] - child_rows["age"]).median()
    child_rows = child_rows.set_index("age").reindex(range(6, 20))
    child_rows["age"] = child_rows.index.astype(int)
    child_rows["AGE_YEARS"] = child_rows["AGE_YEARS"].fillna(
        pd.Series(child_rows.index + observed_age_offset, index=child_rows.index)
    )
    return child_rows.sort_values("AGE_YEARS")


def _align_metric(
    ages: np.ndarray,
    raw_values: np.ndarray,
) -> tuple[list[float | None], list[bool]]:
    """Replay preprocessing and return ages 8..18 plus injected labels."""
    filled = raw_values.copy()
    injected_rows = np.zeros(len(filled), dtype=bool)

    for i in range(1, len(filled) - 1):
        if np.isnan(filled[i]) and np.isfinite(filled[i - 1]) and np.isfinite(filled[i + 1]):
            filled[i] = (filled[i - 1] + filled[i + 1]) / 2
            injected_rows[i] = True

    aligned_values: list[float | None] = []
    aligned_injected: list[bool] = []
    for age in range(7, 19):
        close_idx = np.where(np.abs(ages - age) <= (2 / 12))[0]
        if len(close_idx) > 0:
            idx = close_idx[np.argmin(np.abs(ages[close_idx] - age))]
            value = filled[idx]
            if np.isnan(value):
                aligned_values.append(None)
                aligned_injected.append(False)
            else:
                aligned_values.append(round(float(value), 1))
                aligned_injected.append(bool(injected_rows[idx]))
            continue

        before = np.where(ages < age)[0]
        after = np.where(ages > age)[0]
        if len(before) == 0 or len(after) == 0:
            aligned_values.append(None)
            aligned_injected.append(False)
            continue

        i0 = before.max()
        i1 = after.min()
        v0, v1 = filled[i0], filled[i1]
        if np.isnan(v0) or np.isnan(v1):
            aligned_values.append(None)
            aligned_injected.append(False)
            continue

        t0, t1 = ages[i0], ages[i1]
        fraction = (age - t0) / (t1 - t0)
        value = v0 + (v1 - v0) * fraction
        aligned_values.append(round(float(value), 1))
        aligned_injected.append(bool(injected_rows[i0] or injected_rows[i1]))

    # Age 7 was calculated only to reproduce the original preprocessing exactly.
    return aligned_values[1:], aligned_injected[1:]


def _same_values(expected: list, actual: list) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_value, actual_value in zip(expected, actual):
        if expected_value is None or actual_value is None:
            if expected_value is not None or actual_value is not None:
                return False
        elif not np.isclose(float(expected_value), float(actual_value), atol=1e-9, rtol=0):
            return False
    return True


def add_labels(source_json: Path, raw_csv: Path, output_json: Path) -> None:
    with source_json.open(encoding="utf-8") as source_file:
        children = json.load(source_file)

    child_ids = {int(child_id) for child_id in children}
    raw_parts = []
    for chunk in pd.read_csv(
        raw_csv,
        usecols=RAW_COLUMNS,
        parse_dates=["Birth", "Date_measured"],
        chunksize=250_000,
        low_memory=False,
    ):
        selected = chunk[chunk["CROWD_ID"].isin(child_ids)]
        if not selected.empty:
            raw_parts.append(selected)

    raw = pd.concat(raw_parts, ignore_index=True).rename(columns=RENAME_COLUMNS)
    raw = raw[(raw["age"] >= 6) & (raw["age"] <= 19)].copy()
    raw["AGE_YEARS"] = (
        (raw["measurement_date"] - raw["birth_date"]).dt.total_seconds()
        / (365 * 24 * 3600)
    )
    raw = raw[(raw["AGE_YEARS"] >= 6) & (raw["AGE_YEARS"] <= 19)].copy()

    seen_ids: set[int] = set()
    injected_value_count = 0
    changed_value_count = 0
    for child_id, child_rows in raw.groupby("CROWD_ID"):
        child_id = int(child_id)
        if child_id not in child_ids:
            continue
        seen_ids.add(child_id)
        child_rows = _complete_annual_grid(child_rows)
        ages = child_rows["AGE_YEARS"].to_numpy()
        corrected_data = {}
        labels = {}
        for metric in METRICS:
            values, injected = _align_metric(ages, _numeric_values(child_rows[metric]))
            expected = children[str(child_id)]["data"][metric]
            changed_value_count += sum(
                not _same_values([old_value], [new_value])
                for old_value, new_value in zip(expected, values)
            )
            corrected_data[metric] = values
            labels[metric] = injected
            injected_value_count += sum(injected)
        children[str(child_id)]["data"] = corrected_data
        children[str(child_id)]["injected"] = labels

    missing_ids = child_ids - seen_ids
    if missing_ids:
        example = sorted(missing_ids)[:5]
        raise ValueError(f"Raw records were not found for {len(missing_ids)} children: {example}")

    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(children, output_file, indent=2)

    print(f"Reprocessed and labeled {len(children):,} children on a complete annual grid.")
    print(f"Changed {changed_value_count:,} stored feature values relative to the original JSON.")
    print(f"Marked {injected_value_count:,} aligned feature values as injected.")
    print(f"Wrote {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-json",
        type=Path,
        default=Path("slofit_refactored_37311.json"),
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("../SLOFIT_LONG_COMPLETE_anon_16.01.2019.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("slofit_refactored_37311_corrected_labeled.json"),
    )
    args = parser.parse_args()
    add_labels(args.source_json, args.raw_csv, args.output_json)


if __name__ == "__main__":
    main()
