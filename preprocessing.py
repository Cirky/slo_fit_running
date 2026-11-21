"""Original revised-study numeric preprocessing, retained unchanged."""
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
