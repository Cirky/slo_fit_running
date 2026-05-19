import pandas as pd
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

# ----------------- helpers -----------------

def _parse_ages(x: Union[str, int, float]) -> List[int]:
    """Parse the CSV 'ages' field into a sorted list of ints."""
    if pd.isna(x):
        return []
    if isinstance(x, (int, float)) and not pd.isna(x):
        return [int(x)]
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return sorted(int(p) for p in parts)

def _infer_axes_from_data(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Infer B-axis (last age in window) and C-axis (target_age)."""
    b_axis = sorted(df["B"].unique().tolist())
    c_axis = sorted(df["target_age"].unique().tolist())
    return b_axis, c_axis

def _make_BC_table(
    df: pd.DataFrame,
    b_axis: List[int],
    c_axis: List[int],
    value_col: str = "mae_pi",
) -> pd.DataFrame:
    """
    Build a (B rows) x (C columns) table using (B,C)->value mapping.
    Assumes df has columns ['B','target_age', value_col] with string values.
    """
    lookup = {(int(r.B), int(r.target_age)): str(r[value_col]) for _, r in df.iterrows()}
    mat = []
    for B in b_axis:
        row = []
        for C in c_axis:
            row.append(lookup.get((B, C), ""))
        mat.append(row)
    return pd.DataFrame(mat, index=b_axis, columns=c_axis)

def _latex_table_from_df_BC(
    table_df: pd.DataFrame,
    caption: str,
    label: str,
    col_align: str = "r",
    width_factor: float = 0.9,
) -> str:
    """
    Render B↓ / C→ LaTeX table with booktabs and adjustbox.
    """
    c_axis = list(table_df.columns)
    n = len(c_axis)
    colfmt = "l" + (col_align * n)

    header_cols = " & " + " & ".join([f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in c_axis]) + " \\\\"

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{adjustbox}}{{width={width_factor}\\paperwidth,center}}")
    lines.append(f"\\begin{{tabular}}{{{colfmt}}}")
    lines.append("\\toprule")
    lines.append("B$\\downarrow$ / C$\\rightarrow$")
    lines.append(header_cols)
    lines.append("\\midrule")

    for B in table_df.index:
        cells = [("" if pd.isna(table_df.loc[B, C]) else str(table_df.loc[B, C])) for C in c_axis]
        cells = [("" if (c is None or str(c).lower() == "nan") else c) for c in cells]
        row_str = f"{B}  & " + " & ".join(cells) + " \\\\"
        lines.append(row_str)

    lines.append("\\bottomrule")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# ----------------- main API -----------------

def csv_to_latex_tables_BC(
    csv_source: Union[str, StringIO],
    *,
    require_start_age: Optional[int] = 8,      # ensure windows start at this age; set None to skip check
    b_axis: Optional[List[int]] = None,        # override row ages (B)
    c_axis: Optional[List[int]] = None,        # override column ages (C = target_age)
    caption_template: str = (
        "Average error (in deciseconds; MAE ± 95\\% prediction interval) for forecasting the mature 60 m dash ({sex_word}) "
        "using 60 m dash windows that always start at age 8 and end at B (rows), evaluated for target age C (columns)."
    ),
    label_template: str = "tab:60m_{sex_word}_B_by_C",
) -> Dict[int, str]:
    """
    Reads CSV with columns:
      sex, model, target, target_age, ages, mae_pi
    where 'ages' is a comma-separated list that ALWAYS starts at 8 and ends at B.

    Returns dict {sex_value: latex_string}.
    """
    df = pd.read_csv(csv_source)

    # Basic checks
    required = {"sex", "model", "target", "target_age", "ages", "mae_pi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Parse ages, derive B = max(ages)
    df = df.copy()
    df["ages_list"] = df["ages"].apply(_parse_ages)
    df = df[df["ages_list"].map(len) > 0]

    if require_start_age is not None:
        # Keep only rows that start with the required start age (default 8)
        df = df[df["ages_list"].map(lambda a: min(a) == require_start_age)]

    df["B"] = df["ages_list"].map(max)
    df["target_age"] = df["target_age"].astype(int)

    # Infer axes if not provided
    if b_axis is None or c_axis is None:
        _b, _c = _infer_axes_from_data(df)
        if b_axis is None:
            b_axis = _b
        if c_axis is None:
            c_axis = _c

    # Build per-sex LaTeX tables
    results: Dict[int, str] = {}
    for sex in sorted(df["sex"].unique()):
        sub = df[df["sex"] == sex][["B", "target_age", "mae_pi"]].drop_duplicates()
        table_df = _make_BC_table(sub, b_axis=b_axis, c_axis=c_axis, value_col="mae_pi")
        sex_word = "boys" if sex == 1 else "girls"
        caption = caption_template.format(sex_word=sex_word)
        label = label_template.format(sex_word=sex_word)
        latex = _latex_table_from_df_BC(table_df, caption=caption, label=label)
        results[sex] = latex

    return results

# ----------------- example usage -----------------
if __name__ == "__main__":
    # Example 1: read from a file (uncomment and set your path)
    csv_path = "experiment_results_bottom_60_ages.csv"

    tables = csv_to_latex_tables_BC(
        csv_source=csv_path,
        # Fix axes to your preferred layout (optional):
        b_axis=list(range(8, 18)),         # rows: B in 8..17
        c_axis=list(range(9, 19)),        # cols: C in 15..17
    )

    print("===== BOYS (sex=1) =====")
    print(tables.get(1, "No boys data."))

    print("\n===== GIRLS (sex=2) =====")
    print(tables.get(2, "No girls data."))