"""Summarize filled running measurements within age-18-valid cohorts."""

import json
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "slofit_refactored_37311_corrected_labeled.json"
AGE18_RESULTS_PATH = ROOT / "supplementary_AB_all_corrected_no_injected.csv"
OUTPUT_PATH = ROOT / "supplementary_interpolated_running_measurements.csv"
TARGETS = ("dash_60m", "run_600m")


def main():
    with JSON_PATH.open(encoding="utf-8") as stream:
        participants = json.load(stream)

    counts = defaultdict(lambda: {"available": 0, "filled": 0, "filled_without_next_target": 0})
    cohort_sizes = defaultdict(int)
    for participant in participants.values():
        for target in TARGETS:
            values = participant["data"][target]
            labels = participant["injected"][target]
            if len(values) != 11 or len(labels) != 11:
                raise ValueError(f"Expected ages 8 through 18 for {target}")
            if values[-1] is None:
                continue
            cohort_sizes[target] += 1
            for age, value, filled in zip(range(8, 18), values[:-1], labels[:-1]):
                if value is not None:
                    counts[target, age]["available"] += 1
                if filled:
                    if value is None:
                        raise ValueError(f"Filled value missing at age {age}: {target}")
                    counts[target, age]["filled"] += 1
                    if values[age - 8 + 1] is None:
                        counts[target, age]["filled_without_next_target"] += 1

    # The age-17 exclusions must match the independently computed age-18
    # forecasting results for observation windows beginning at age 8.
    age18_exclusions = defaultdict(int)
    with AGE18_RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["start_age"]) == 8 and int(row["cutoff_age"]) == 17:
                age18_exclusions[row["target"]] += int(row["n_excluded_injected"])

    rows = []
    for age in range(8, 18):
        row = {"age": age, "next_year_target_age": age + 1}
        for target in TARGETS:
            available = counts[target, age]["available"]
            filled = counts[target, age]["filled"]
            excluded = filled - counts[target, age]["filled_without_next_target"]
            if not 0 <= excluded <= filled <= available <= cohort_sizes[target]:
                raise ValueError(f"Inconsistent counts for {target} at age {age}")
            row[f"{target}_age18_cohort"] = cohort_sizes[target]
            row[f"{target}_available"] = available
            row[f"{target}_filled"] = filled
            row[f"{target}_filled_pct"] = round(100 * filled / available, 1)
            row[f"{target}_filled_without_next_target"] = counts[target, age]["filled_without_next_target"]
            row[f"{target}_excluded_next_year"] = excluded
            if age == 17 and excluded != age18_exclusions[target]:
                raise ValueError(f"Age-18 exclusion differs from forecasting results: {target}")
        rows.append(row)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
