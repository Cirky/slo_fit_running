from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import training_automated as exp


START_MARKER = "% BEGIN RMSE SUPPLEMENTARY TABLES"
END_MARKER = "% END RMSE SUPPLEMENTARY TABLES"


RUNS = [
    {
        "target": "dash_60m",
        "csv": "experiment_results_bottom_60_rmse.csv",
        "name": "60-meter dash",
        "short": "60m",
        "unit": "deciseconds",
    },
    {
        "target": "run_600m",
        "csv": "experiment_results_bottom_600_rmse.csv",
        "name": "600-meter run",
        "short": "600m",
        "unit": "seconds",
    },
]


class SilentProgress:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def parse_ages(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def run_bottom_25_experiment(target: str, csv_path: str) -> None:
    exp.FEATURES = [target]
    exp.TEST_SPLIT_HANDLING = "bottom"
    exp.DO_SWEEP_AGE_WINDOWS_ONLY_FROM_8 = False
    exp.tqdm = SilentProgress

    df_all = exp.load_flatten(exp.JSON_PATH, sexes=(1, 2))
    exp.run_experiments(
        df_all=df_all,
        sexes=(1, 2),
        targets=[target],
        target_ages=[18],
        feature_pool=[target],
        do_feature_combos=False,
        ks=[1],
        do_age_windows=True,
        window_sizes=list(range(1, 11)),
        age_min=8,
        age_max=18,
        model_configs=[
            ("HistGBRegressor", HistGradientBoostingRegressor(), False),
        ],
        n_splits=exp.N_SPLITS,
        random_state=exp.RANDOM_STATE,
        results_csv=csv_path,
    )


def make_rmse_table(
    csv_path: str,
    *,
    target_name: str,
    short_name: str,
    unit: str,
    sex: int,
    ages: Iterable[int] = range(8, 18),
) -> str:
    df = pd.read_csv(csv_path)
    df = df[(df["sex"] == sex) & (df["target_age"] == 18)].copy()
    df["ages_list"] = df["ages"].apply(parse_ages)
    df["A"] = df["ages_list"].apply(min)
    df["B"] = df["ages_list"].apply(max)
    if "rmse_relative_error" not in df.columns:
        if "relative_error_pct" not in df.columns:
            raise KeyError("CSV must contain rmse_relative_error or relative_error_pct.")
        df["rmse_relative_error"] = df.apply(
            lambda row: f"{row.rmse} | {float(row.relative_error_pct):.1f}",
            axis=1,
        )

    sex_word = "boys" if sex == 1 else "girls"
    label_sex = "boys" if sex == 1 else "girls"
    lookup = {(int(row.A), int(row.B)): str(row.rmse_relative_error) for row in df.itertuples()}
    age_axis = list(ages)

    colfmt = "l" + ("r" * len(age_axis))
    header = " & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{age}}}" for age in age_axis) + " \\\\"

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        (
            f"\\caption{{Root mean squared error (RMSE) and relative prediction error "
            f"for the 25\\% of {sex_word} "
            f"with the lowest fitness for forecasting mature {target_name} performance "
            f"(in {unit}). Rows indicate the starting age $A$ and columns the ending age $B$ "
            "of the observation window used for prediction. Each cell is reported in the format "
            "RMSE~|~relative prediction error (\\%).}"
        ),
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{width=\\textwidth,center}",
        f"\\begin{{tabular}}{{{colfmt}}}",
        "\\toprule",
        "A$\\downarrow$ / B$\\rightarrow$",
        header,
        "\\midrule",
    ]

    for a in age_axis:
        cells = [lookup.get((a, b), "") if b >= a else "" for b in age_axis]
        lines.append(f"{a}  & " + " & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            f"\\label{{tab:rmse_{short_name}_{label_sex}}}",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def replace_generated_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    generated = f"{START_MARKER}\n{block}\n{END_MARKER}"

    if START_MARKER in text and END_MARKER in text:
        before, rest = text.split(START_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        text = before.rstrip() + "\n\n" + generated + after
    else:
        text = text.replace("\\end{document}", generated + "\n\n\\end{document}")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    tables: list[str] = []

    for run in RUNS:
        run_bottom_25_experiment(run["target"], run["csv"])
        for sex in (1, 2):
            tables.append(
                make_rmse_table(
                    run["csv"],
                    target_name=run["name"],
                    short_name=run["short"],
                    unit=run["unit"],
                    sex=sex,
                )
            )

    replace_generated_block(Path("supplementary_files.tex"), "\n\n".join(tables))


if __name__ == "__main__":
    main()
