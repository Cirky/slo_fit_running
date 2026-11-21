# SLOFIT Running Performance Analysis

Code for *Machine Learning Enhances the Forecasting of Running Performance as an
Indicator of Physical Fitness in Low-Fit Children*.

The default workflow reproduces the **reported revised analysis**, including its
365-day age convention and interpolation-label exclusions. Calendar-based
preprocessing and boundary-case exclusions are separate sensitivity analyses.
Individual-level data are restricted and are not distributed here.

## Installation

The reconstruction was checked with Python 3.12.2 and the versions pinned in
`requirements.txt`. From this repository directory:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Reproduce the reported running analyses

Obtain authorized access to the same raw release,
`SLOFIT_LONG_COMPLETE_anon_16.01.2019.csv`. Replace `/path/to/raw.csv` below with
its actual path. Quote paths containing spaces. Commands are the same in
PowerShell and a POSIX shell.

```bash
python prepare_data.py --raw-csv /path/to/raw.csv
python describe_data.py --raw-csv /path/to/raw.csv
python run_analysis.py --job all
python validate_results.py --report artifacts/validation.json
python export_outputs.py --results-dir artifacts/results
```

There is no required previously prepared cohort JSON and no manual filename
renaming. `prepare_data.py` generates all three files needed by later steps:

| Output | Purpose |
| --- | --- |
| `artifacts/data/study_dataset.json` | Selected cohort, corrected aligned values, and `injected` labels |
| `artifacts/data/excluded_cohort.json` | Private participant keys and metadata for the completeness-excluded source cohort |
| `artifacts/data/cohort_summary.json` | Aggregate selection counts, source-file hash, and optional private-reference verification status |

All generated files belong under the ignored `artifacts/` directory. Keep
participant-level inputs and outputs out of the public repository.

For a short execution check, use a separate results directory:

```bash
python run_analysis.py --job tables_ab --max-tasks 1 --output-dir artifacts/smoke
```

This intentionally records an incomplete job. It does not count as reproducing
the full study. Complete model jobs can take several minutes or longer,
depending on hardware. Interrupted jobs resume from completed configurations;
input/code/configuration fingerprints prevent mixing incompatible runs.

## Exact preprocessing profile

`configs/study.json` records the configuration. `preprocessing.py` preserves the
numerical operations used to generate the corrected reported dataset.

1. Keep nominal ages 6 through 19.
2. Exclude **all records of a child** if any nominal age is duplicated.
3. Require a raw record at nominal age 18. Outcome-specific availability is
   checked after alignment; a row at age 18 does not itself imply a valid target.
4. Calculate chronological age as elapsed seconds divided by `365*24*3600`, and
   retain chronological ages 6 through 19.
5. Before imputation, require at least **seven observed numeric values in every
   one of the 11 metrics**. The criterion applies to the retained raw annual
   records, not to the number of nonmissing aligned predictor values in a window.
6. Insert absent nominal years on a complete 6–19 grid. In the reported profile,
   missing-row age coordinates use the child's median observed age offset.
7. Fill only an isolated missing annual value with the average of its immediate
   neighbours. Consecutive missing years remain missing.
8. Align to integer ages 8–18. Use the closest observation within `2/12` years
   directly; otherwise interpolate between the nearest rows on either side,
   provided both have values. Round to one decimal place.
9. Mark dependence on missing-value filling and propagate these labels through
   alignment. Birthday alignment alone does not create an `injected` label.

The runner excludes a participant-window when a selected feature at its latest
age is injected. For quartile tasks, this also includes the running measurement
used for ranking. Earlier injected predictors are not automatically discarded.
For LR and GB, remaining missing predictor values cause complete-case exclusion;
HGB and XGB can accept earlier missing predictors. The existing fold-level
handling of a missing latest running measurement is preserved.

The output field `cutoff_age` is the ending **aligned age B**, not necessarily
the date when a forecast could be issued. Operational availability requires the
source assessments used by the predictors, including the later observation when
birthday alignment uses two assessments. The optional audit describes these
dependencies and the rare boundary exceptions in the original convention.

The source cohort has 37,311 children. Of these, 32,288 have an aligned age-18
target in at least one running outcome: 16,258 boys and 16,030 girls. Valid targets
number 31,546 for 60 m and 31,408 for 600 m. The intermediate selection counts and
the historical denominator distinction are explained in `REPRODUCIBILITY.md`.

## Experiment configuration and manuscript outputs

There is no need to edit constants inside a Python file. Select a job or supply
an explicit alternative JSON configuration with `--config`.

| Job / command | Configuration | Manuscript output |
| --- | --- | --- |
| `run_analysis.py --job tables_ab` | HGB, both sexes/outcomes, target 18, all 55 contiguous predictor windows, slowest quartile | Main Tables 1–4; RMSE and relative error for Supplementary Tables S5–S8 |
| `run_analysis.py --job tables_bc` | HGB, target ages 9–18, windows beginning at 8, slowest quartile | Supplementary Tables S1–S4 |
| `run_analysis.py --job curves_bottom` | LR, HGB, GB, XGB, percentile and GCC; target 18; latest ages 8–17 | Main performance figures |
| `run_analysis.py --job curves_all` | Same methods and ages, all participants | Supplementary Figure 1 |
| `run_analysis.py --job curves_top` | Same methods and ages, fastest quartile | Supplementary Figure 2 |
| `describe_data.py --raw-csv ...` | Reconstruct completeness-excluded aligned outcomes | Supplementary Tables S9–S10 |
| `describe_data.py` | Count missing-value interpolation labels by age | Supplementary Table S11 |
| `plot_trajectories.py` | Historical input; seeded illustrative trajectories, 1/10/100 nearest-percentile samples | Illustrations corresponding to Supplementary Figures 3–5 |

The ML evaluations use five shuffled individual-level folds with seed 42. The
slowest/fastest quartiles are selected within the held-out fold, using the
latest running measurement. Training retains all eligible performance levels.
Boosting estimator seeds are 42. Original early-stopping helpers and their
settings are retained: validation fraction 0.2, HGB patience 20, GB patience 15,
and XGB patience 40 with a cap of 100 estimators and validation split seed 1.
GCC uses 200 neighbours. The percentile and GCC baselines preserve the original
in-sample reference-population evaluation; they are not silently changed to a
different validation protocol.

Outputs record numeric MAE, RMSE, relative error, PI half-width, cohort size, and
interpolation exclusions where applicable. Baseline rows contain MAE only.
60 m times remain in **deciseconds**, and 600 m times in **seconds**. The PI
quantity is `1.96 * mean(fold SD of signed errors)`, not a full interval width.
Each job writes a manifest with its settings, software versions and input/code
hashes.

`export_outputs.py` writes fifteen LaTeX table fragments (including S9–S11 when
the descriptive CSVs are available) and six performance
plots. The fragments require `booktabs` in the containing LaTeX document. They
are exported separately and do not modify a manuscript automatically.

## Recreate displays without restricted data

`reference_results/` contains only aggregate outputs from the reported revision,
with source filenames and SHA-256 hashes in its manifest. No participant rows
are included. To recreate their performance figures and table fragments:

```bash
python export_outputs.py --reference --output-dir artifacts/reference_exports
```

This regenerates displays from archived aggregates, **not** model fitting.
`validate_results.py` compares a genuine data-to-results rerun with those
aggregates. `VALIDATION.json` records the completed local comparison of all five
experiment jobs and both descriptive summaries. See `REPRODUCIBILITY.md` for its
scope and numerical tolerance.

## Optional exploratory feature comparisons

The eight named profiles in `configs/feature_search_profiles.json` retain all
8,030 distinct feature/window/model task definitions from the archived searches.
These experiments are separate from the running-only results above.

For historical searches, first reconstruct their original observed-row
preprocessing, before the complete annual-grid correction:

```bash
python prepare_historical_features.py --raw-csv /path/to/raw.csv
python run_feature_search.py --profile experiment_results_features_600 --max-tasks 1
```

Omit `--max-tasks` to run all tasks in the chosen profile. Use a fresh output
filename for each run. The profile
`experiment_results_features_600_bottom_future_no_injected_latest` instead uses
`study_dataset.json` and the revised interpolation exclusions. Historical seeds
were not stored in those CSVs; the optional runner explicitly uses seed 42 for
repeatable reruns and does not claim to recover their original random states.
Some archived tasks use target-age covariates; those are conditional exploratory
comparisons and must not be described as prospective forecasts at an earlier
cutoff. The exact feature lists and future-feature fields remain visible in the
task CSV. The current local validation covers dataset reconstruction and runner
smoke checks, not rerunning every exploratory task.

After preparing the historical input, reproduce the illustrative trajectories
with `python plot_trajectories.py`. Its input path can be overridden with
`--data`; choosing a different preprocessing profile can change the illustrations.

## Optional temporal and calendar sensitivity checks

These checks support the review response and are **not required to reproduce the
reported tables and figures**. Detailed commands and interpretation are in
`sensitivity/README.md`. They use separate output paths and preserve the study
dataset and aggregate reference results.

## Code map

`prepare_data.py` performs cohort selection and label generation;
`preprocessing.py` contains the original numeric transformations;
`analysis.py` contains the original loading, eligibility and CV functions;
`helper_functions.py` contains the original estimators and fitting helpers.
`run_analysis.py` provides named reproducible jobs, `describe_data.py` provides
the descriptive summaries, and `export_outputs.py` and `validate_results.py`
handle displays and verification. Optional feature and sensitivity scripts are
documented separately above. Cite this work using `CITATION.cff`; code is
distributed under the existing MIT license.
