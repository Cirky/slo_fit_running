import pandas as pd
from io import StringIO
from typing import Dict, List, Tuple, Optional, Union

def _parse_ages(x: Union[str, int, float]) -> List[int]:
    """Parse the CSV 'ages' field into a sorted list of ints."""
    if pd.isna(x):
        return []
    if isinstance(x, (int, float)) and not pd.isna(x):
        return [int(x)]
    # string case
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return sorted(int(p) for p in parts)

def _span_from_ages(ages: List[int]) -> Optional[Tuple[int, int]]:
    """Return (A, B) from a list of ages. If empty, None."""
    if not ages:
        return None
    return (min(ages), max(ages))

def _collect_age_axis(df: pd.DataFrame) -> List[int]:
    """Collect the complete sorted age axis from all A,B spans."""
    spans = df["ages_list"].apply(_span_from_ages).dropna().tolist()
    if not spans:
        return []
    all_ages = set()
    for a, b in spans:
        for age in range(a, b + 1):
            all_ages.add(age)
    return sorted(all_ages)

def _make_upper_triangle_table(
    df: pd.DataFrame,
    ages_axis: List[int],
    value_col: str = "mae_pi",
) -> pd.DataFrame:
    """
    Build a (A rows) x (B columns) upper-triangular table using (A,B)->value mapping.
    df must include 'A', 'B', and `value_col`.
    """
    # Map (A,B) -> value
    lookup = {(int(r.A), int(r.B)): str(r[value_col]) for _, r in df.iterrows()}
    # Build table
    mat = []
    for A in ages_axis:
        row = []
        for B in ages_axis:
            if B < A:
                row.append("")  # lower triangle stays empty
            else:
                row.append(lookup.get((A, B), ""))
        mat.append(row)
    table = pd.DataFrame(mat, index=ages_axis, columns=ages_axis)
    return table

def _latex_table_from_df(
    table_df: pd.DataFrame,
    caption: str,
    label: str,
    colfmt_align: str = "r",  # LaTeX alignment for numeric columns
    width_factor: float = 0.9,
) -> str:
    """
    Render the upper-triangular table to a LaTeX string that matches your style:
    - booktabs rules
    - Adjustbox width
    - Header "A↓ / B→" and column labels
    """
    ages_axis = list(table_df.columns)
    # Column format: one leading 'l' for the row header, then r repeated N times
    n = len(ages_axis)
    colfmt = "l" + (colfmt_align * n)

    header_cols = " & " + " & ".join([f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in ages_axis]) + " \\\\"
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{adjustbox}}{{width={width_factor}\\paperwidth,center}}")
    lines.append(f"\\begin{{tabular}}{{{colfmt}}}")
    lines.append("\\toprule")
    lines.append("A$\\downarrow$ / B$\\rightarrow$")
    lines.append(header_cols)
    lines.append("\\midrule")

    # Body rows
    for A in table_df.index:
        cells = [str(table_df.loc[A, B]) if table_df.loc[A, B] is not None else "" for B in ages_axis]
        # Ensure empty cells are exactly empty (no 'nan')
        cells = [("" if (c is None or str(c).lower() == "nan") else c) for c in cells]
        row_str = f"{A}  & " + " & ".join(cells) + " \\\\"
        lines.append(row_str)

    lines.append("\\bottomrule")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def csv_to_latex_tables(
    csv_source: Union[str, StringIO],
    ages_axis: Optional[List[int]] = None,
    caption_template: str = (
        "Average error (in deciseconds; MAE ± 95\\% prediction interval) for forecasting the mature 60 m dash ({sex_word}) "
        "using 60 m dash values beginning at age A and ending at age B. -  Bottom 25\\% performers."
    ),
    label_template: str = "tab:60m_{sex_word}",
) -> Dict[int, str]:
    """
    Reads CSV and returns LaTeX table strings for each sex present:
    {1: latex_for_boys, 2: latex_for_girls}
    - csv_source: path to CSV file or a StringIO with CSV text
    - model/target/target_age: optional filters
    - ages_axis: optional explicit list of ages (e.g., [8,9,...,17]); if None it is inferred
    """
    df = pd.read_csv(csv_source)
    # Normalize columns
    required_cols = {"sex", "model", "target", "target_age", "ages", "mae_pi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Parse ages -> spans (A,B)
    df = df.copy()
    df["ages_list"] = df["ages"].apply(_parse_ages)
    df = df[df["ages_list"].map(len) > 0]
    spans = df["ages_list"].apply(_span_from_ages)
    df["A"] = spans.apply(lambda x: x[0] if x else None)
    df["B"] = spans.apply(lambda x: x[1] if x else None)
    df = df.dropna(subset=["A", "B"])
    df["A"] = df["A"].astype(int)
    df["B"] = df["B"].astype(int)

    # Age axis
    if ages_axis is None:
        ages_axis = _collect_age_axis(df)
    # Guarantee strictly increasing axis
    ages_axis = sorted(set(ages_axis))

    # Build one LaTeX table per sex
    results: Dict[int, str] = {}
    for sex in sorted(df["sex"].unique()):
        sub = df[df["sex"] == sex][["A", "B", "mae_pi"]].drop_duplicates()
        table_df = _make_upper_triangle_table(sub, ages_axis, value_col="mae_pi")
        sex_word = "boys" if sex == 1 else "girls"
        caption = caption_template.format(sex_word=sex_word)
        label = label_template.format(sex_word=sex_word)
        latex = _latex_table_from_df(table_df, caption=caption, label=label)
        results[sex] = latex
    return results

# ---------- Example usage ----------
if __name__ == "__main__":
    # Option 1: from a file path
    csv_path = "experiment_results.csv"

    # Infer ages from the data (or pass explicit range like list(range(8, 18)))
    tables = csv_to_latex_tables(
        csv_source=csv_path,
        ages_axis=list(range(8, 18)),  # force columns/rows 8..17 to match your layout
    )

    # Access LaTeX strings:
    # Boys (sex==1):
    print("===== BOYS TABLE =====")
    print(tables.get(1, "No boys data."))

    # Girls (sex==2):
    print("\n===== GIRLS TABLE =====")
    print(tables.get(2, "No girls data."))
