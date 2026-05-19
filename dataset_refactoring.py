import pandas as pd
import json
import numpy as np
import datetime


def compute_age_years(df):
    # Calculate age in years (with decimals)
    df['AGE_YEARS'] = (df['measurement_date'] - df['birth_date']).dt.days / 365
    return df


def main():
    """
    Refactor the SLOFIT dataset to a more usable format.
    The output is a JSON file with children as keys and their
    """

    # 1. Load
    df = pd.read_csv("../SLOFIT_LONG_COMPLETE_anon_16.01.2019.csv", parse_dates=["Birth", "Date_measured"])

    # 2. Drop unneeded
    df = df.drop(columns=["School_ID", "Grade", "Municipality", "Region", "BMI", "WS_WOF", "Total_FI"])

    # 3. Rename for clarity
    df = df.rename(columns={
        "Age":        "age",
        "Sex":         "sex",
        "Birth":        "birth_date",
        "Height":      "height",
        "Weight":      "weight",
        "Triceps_sf":     "triceps_skinfold",
        "Arm_plate_tapping": "arm_plate_tapping",
        "Standing_broad_jump":"broad_jump",
        "Polygon_backwards":"polygon_backwards",
        "Sit_ups_60s":    "situps_60s",
        "Stand_and_reach":"sit_and_reach",
        "Bent_arm_hang":  "bent_arm_hang",
        "@60m_dash":      "dash_60m",
        "@600m_run":      "run_600m",
        "Year_measured":  "measurement_year",
        "Date_measured":  "measurement_date"
    })

    # 4. Keep only ages 6–18
    df = df[(df.age >= 6) & (df.age <= 19)]

    # # Find all rows where the (CROWD_ID, Age) pair occurs more than once
    # dup = df[df.duplicated(subset=["CROWD_ID", "age"], keep=False)] \
    #     .sort_values(["CROWD_ID", "age", "measurement_date"])
    #
    # print(f"Found {len(dup)} total duplicate‐age rows across {dup.CROWD_ID.nunique()} children\n")
    #
    # # 4. Print them grouped by child
    # for cid, group in dup.groupby("CROWD_ID"):
    #     print(f"--- Child ID {cid} has {len(group)} records at these ages: {group.age.unique().tolist()} ---")
    #     print(group.loc[:, ["age", "measurement_date", "height", "weight"] +
    #                        ["triceps_skinfold", "broad_jump", "dash_60m", "run_600m"]
    #           ].to_string(index=False))
    #     print()

    # Identify & drop *all* children who have any duplicate‐age rows
    dup_pairs = (
        df[df.duplicated(subset=["CROWD_ID", "age"], keep=False)]
        .loc[:, ["CROWD_ID"]]
        .drop_duplicates()
    )
    bad_ids = set(dup_pairs.CROWD_ID)
    print(f"Dropping {len(bad_ids)} children with duplicate ages")
    df = df[~df.CROWD_ID.isin(bad_ids)]

    # 4) Only keep kids who have *at least one* record at age 18
    has18 = set(df[df.age == 18].CROWD_ID)
    print(f"Keeping {len(has18)} children with data at age 18")
    df = df[df.CROWD_ID.isin(has18)]


    # 5. Define the metric columns we want to turn into year‑indexed lists
    metrics = [
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
        "run_600m"
    ]

    # Compute fractional age in years and restrict to 6–19 inclusive
    df["AGE_YEARS"] = ((df["measurement_date"] - df["birth_date"]).dt.total_seconds()
                       / (365 * 24 * 3600))
    df = df[(df["AGE_YEARS"] >= 6) & (df["AGE_YEARS"] <= 19)].copy()

    # 6. Group and build
    children = {}

    # Process each child
    for cid, sub in df.groupby("CROWD_ID"):
        sex = sub.sex.iloc[0]
        birth = sub.birth_date.iloc[0].date().isoformat()
        sub = sub.sort_values("AGE_YEARS")
        # Copy and interpolate each metric linearly, interior only
        sub_interp = sub.copy()
        drop_child = False
        for feat in metrics:
            metric_values = []
            for v in sub_interp[feat].values:
                if isinstance(v, str):
                    v = v.strip()
                    if v == '' or v == ' ':
                        metric_values.append(None)
                    else:
                        try:
                            metric_values.append(float(v))
                        except ValueError:
                            metric_values.append(None)
                elif pd.isna(v):
                    metric_values.append(None)
                elif isinstance(v, np.generic):
                    metric_values.append(v.item())
                else:
                    metric_values.append(v)
            sub_interp[feat] = metric_values
            # build a real float‐typed Series so we can do .isna() correctly
            ser = pd.Series(metric_values, index=sub_interp.index, dtype="float")

            # if cid == 1000005096:
            #     print(f"DEBUG: {feat} for child {cid} has {sub_interp[feat].isna().sum()} NaNs")
            #     print(sub[feat])
            #     print(metric_values)
            # If too many missing values, drop this child
            non_missing = ser.notna().sum()
            if non_missing <= 6:
                # print(f"Dropping child {cid} — feature {feat} has {non_missing} values")
                drop_child = True
                break

            # Fill missing values
            filled = ser.copy()
            for i in range(1, len(filled) - 1):
                if pd.isna(filled.iat[i]) and pd.notna(filled.iat[i - 1]) and pd.notna(filled.iat[i + 1]):
                    # fill by simple average of immediate neighbors
                    filled.iat[i] = (filled.iat[i - 1] + filled.iat[i + 1]) / 2

            sub_interp[feat] = filled.values

        if drop_child:
            continue

        # Now align ages to birthdays 8–18
        data = {}
        ages = sub_interp["AGE_YEARS"].values
        for feat in metrics:
            values = sub_interp[feat].values
            seq = []
            for age in range(7, 19):  # ages 7 to 18 inclusive
                # If measurement within ±2 months of the birthday, use it directly
                close_idx = np.where(np.abs(ages - age) <= (2/12))[0]
                if len(close_idx) > 0:
                    idx = close_idx[np.argmin(np.abs(ages[close_idx] - age))]
                    val = values[idx]
                    if pd.isna(val):
                        seq.append(None)
                    else:
                        seq.append(round(float(val), 1))
                else:
                    # Otherwise, linear interpolation between nearest neighbours
                    before = np.where(ages < age)[0]
                    after = np.where(ages > age)[0]
                    if len(before) > 0 and len(after) > 0:
                        i0 = before.max()
                        i1 = after.min()
                        t0, t1 = ages[i0], ages[i1]
                        v0, v1 = values[i0], values[i1]
                        # Only interpolate if both sides are non-NaN
                        if not pd.isna(v0) and not pd.isna(v1):
                            frac = (age - t0) / (t1 - t0)
                            seq.append(round(float(v0 + (v1 - v0) * frac), 1))
                        else:
                            seq.append(None)
                    else:
                        seq.append(None)
            data[feat] = seq[1:]  # skip age 7, we don't have data for it

        children[str(cid)] = {
            "sex": sex,
            "birth": birth,
            "data": data
        }

    # 7. Dump to a JSON file
    with open("slofit_refactored.json", "w") as f:
        json.dump(children, f, indent=2, default=_json_convert)

    print("Wrote", len(children), "children to slofit_refactored.json")

def _json_convert(obj):
    """
    Turn any NumPy scalar or pandas date into a native Python type.
    json.dump will call this on any un-serializable object.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    # let the error bubble up if it’s truly un-handled
    raise TypeError(f"Type {type(obj)} not serializable")


if __name__ == '__main__':
    main()