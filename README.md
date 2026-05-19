# SLOFIT Running Performance Analysis

This repository contains the analysis code accompanying the manuscript on
longitudinal prediction of youth running performance from SLOFIT measurements.
It is intended to make the computational workflow inspectable for reviewers and
readers.

The SLOFIT database is not included in this repository because it cannot be
redistributed publicly. The scripts therefore cannot be run end-to-end unless
the user has independently obtained access to the restricted data from the data
owner or through the manuscript authors, subject to the applicable data access
conditions.

## Repository Contents

| File | Purpose |
| --- | --- |
| `dataset_refactoring.py` | Converts the restricted raw SLOFIT longitudinal CSV into the aligned JSON format used by the experiments. |
| `training_pipeline.py` | Main model comparison pipeline for age-window prediction and MAE plots. |
| `training_automated.py` | Batch experiment runner that writes tabular results to `experiment_results.csv`. |
| `helper_functions.py` | Shared model fitting, cross-validation, early-stopping, percentile, and GCC helper code. |
| `height_gcc_repeat.py` | Standalone GCC evaluation script for complete age trajectories. |
| `plot_individuals.py` | Generates percentile-based individual trajectory plots. |

## Installation

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

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

## Typical Workflow

1. Obtain authorized access to the restricted SLOFIT data.
2. Place the raw CSV outside the repository at the path expected by
   `dataset_refactoring.py`.
3. Run preprocessing:

   ```bash
   python dataset_refactoring.py
   ```

4. Rename the generated JSON if needed:

   ```bash
   mv slofit_refactored.json slofit_refactored_37311.json
   ```

5. Run the desired analysis script:

   ```bash
   python training_pipeline.py
   python training_automated.py
   python height_gcc_repeat.py
   python plot_individuals.py
   ```

## Reproducibility Notes

The scripts expose the main experimental settings near the top of each file,
including selected features, targets, model configurations, cross-validation
folds, random seeds, and output paths.

Because the database is restricted, this public repository supports code review
and methodological inspection. Full numerical reproduction requires access to
the same SLOFIT data release and the preprocessing choices encoded in
`dataset_refactoring.py`.
