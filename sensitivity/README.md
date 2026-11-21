# Optional review-response sensitivity analyses

Run these from the repository root after `prepare_data.py`. They are separate
from the commands that reproduce the reported results. Replace the raw-data path
with the authorized CSV location.

## Trace source observation dates

```bash
python sensitivity/audit_dates.py --data artifacts/data/study_dataset.json --raw-csv /path/to/raw.csv
```

The audit replays the original two interpolation operations and verifies the
running values and labels against the selected study dataset. For a retained
forecast, its proposed assessment cutoff is the latest source date of the
latest non-injected predictor. Earlier running predictors are checked against
that date. The entire earlier prefix is checked, which also covers its shorter
contiguous windows. The audit reports when the target's source observations are
already available at that cutoff. It writes aggregate counts and a **private**
`artifacts/sensitivity/source_latest.json` used by the next step.

This traces measurement-value dependencies. It does not assert that the original
full-record median estimate of missing-row age coordinates is invariant to
truncating future records. The calendar alternative below removes that estimate.

## Exclude only the target-availability boundary cases

```bash
python sensitivity/boundary_results.py --data artifacts/data/study_dataset.json --provenance artifacts/sensitivity/source_latest.json
```

This retains the original aligned values and compares HGB results before and
after excluding already-available targets from age-17-ending windows. The raw
source audit identifies 17 cases per running outcome in the study data. Outer
folds are formed after exclusions, as in the original pipeline; changed cohort
sizes can therefore change both outer-fold and early-stopping membership. The
output is a sensitivity CSV, not a replacement main result file.

## Broader calendar realignment

```bash
python sensitivity/calendar_data.py --data artifacts/data/study_dataset.json --raw-csv /path/to/raw.csv
python run_analysis.py --data artifacts/sensitivity/calendar_dataset.json --job tables_ab --output-dir artifacts/sensitivity/calendar_results
```

This places missing annual records on April 15, aligns to actual calendar
birthdays, uses a two-calendar-month direct-measurement tolerance and interpolates
on elapsed days. February-29 anniversaries use February 28 in non-leap years.
The source cohort and original raw-row selection are held fixed. Dependence on
missing-value filling is still labelled, and the latest-injected exclusion is
unchanged. Supply other job names, or `--job all`, to run further comparisons.

This is a broader change than excluding the 17 boundary cases: it can slightly
alter many predictors and targets. Calendar results should not be compared with
the archived main outputs as though they were the same preprocessing profile.
Use their manifest and a separate output directory to retain provenance. The
main input and reference results remain intact.
