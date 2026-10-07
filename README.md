# SLOFIT Running Performance Analysis

This repository contains code accompanying the manuscript on predicting
60-meter dash and 600-meter run performance from longitudinal SLOFIT
measurements. It includes data preparation, model evaluation, and scripts
used to produce supporting figures and tables.

Individual-level SLOFIT data are restricted and are not included in this
repository. Running the analyses requires authorized access to the data.

## Repository Contents

| File | Purpose |
| --- | --- |
| `dataset_refactoring.py` | Converts the raw longitudinal CSV into age-aligned participant records. Cohort completeness criteria are configured in the script. |
| `label_interpolated_values.py` | Builds a complete annual age grid for the selected cohort and records which measurements were interpolated. |
| `training_automated.py` | Evaluates prediction models across running tests, observation windows, target ages, and participant groups; writes the results to CSV. |
| `helper_functions.py` | Provides shared model-fitting utilities and the percentile and Growth Curve Comparison estimators. |
| `training_pipeline.py` | Runs model comparisons and produces MAE plots for configured prediction settings. |
| `generate_corrected_supplementary_figures.py` | Produces prediction-error curves and supplementary figure files. |
| `generate_rmse_supplementary_tables.py` | Computes RMSE and relative prediction error and generates Supplementary Tables S5–S8. |
| `recompute_selection_bias_tables.py` | Compares age-18 running performance between children included in and excluded from the study cohort for Supplementary Tables S9–S10. |
| `summarize_interpolated_running_measurements.py` | Summarizes interpolated running measurements by age for Supplementary Table S11. |
| `height_gcc_repeat.py` | Evaluates Growth Curve Comparison on complete height trajectories. |
| `plot_individuals.py` | Produces illustrative plots of individual performance trajectories. |
| `requirements.txt` | Lists the Python dependencies. |

## Installation

Python 3.10 or newer is recommended. Create an environment and install the
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`.

## Data Availability

The analysis requires individual-level SLOFIT data. These data are restricted
and are not redistributed here.

The preprocessing script expects the raw CSV at:

```text
../SLOFIT_LONG_COMPLETE_anon_16.01.2019.csv
```

Running `dataset_refactoring.py` creates:

```text
slofit_refactored.json
```

Several experiment scripts currently use this default derived-data filename:

```text
slofit_refactored_37311.json
```

If you have authorized data access, either rename the generated JSON file to
match the script defaults or edit the `json_path` / `JSON_PATH` constants in the
scripts before running them.

The derived JSON has one record per child and stores age-aligned values for ages
8 to 18:

```json
{
  "12345": {
    "sex": 1,
    "birth": "YYYY-MM-DD",
    "data": {
      "height": [null, 1331.3, "..."],
      "dash_60m": [null, 123.4, "..."],
      "run_600m": [null, 210.5, "..."]
    }
  }
}
```

The available measurement keys are:

```text
height, weight, triceps_skinfold, arm_plate_tapping, broad_jump,
polygon_backwards, situps_60s, sit_and_reach, bent_arm_hang,
dash_60m, run_600m
```

## Analysis workflow

After preparing the selected cohort, run
`label_interpolated_values.py` with the cohort JSON and authorized raw CSV.
The resulting age-aligned, labeled JSON is the default input to
`training_automated.py`. Configure the target test, target ages, observation
windows, participant group, models, and result filename near the top of that
script.

The scripts for supplementary figures and tables read the applicable
analysis data or results. `recompute_selection_bias_tables.py` additionally
requires a file identifying children excluded by the completeness criterion.
`generate_rmse_supplementary_tables.py` writes generated tables into
`supplementary_files.tex`, which must be available when that script is run.
Check the input and output paths in each script before execution.

## Reproducibility

The repository provides the computational methods for inspection. Numerical
reproduction requires authorized access to the SLOFIT data, the study cohort
and exclusion records, and the experiment settings reported in the
manuscript.
