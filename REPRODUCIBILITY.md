# Reproduction scope and implementation details

The reported-result profile preserves the original revised numerical methods.
It does not adopt calendar-based realignment or remove the rare date-boundary
cases from the main outputs. The optional sensitivity directory evaluates those
alternatives explicitly.

## Local verification

Using the authorized original raw CSV, `prepare_data.py --verify-against` rebuilt
the private corrected/labeled dataset with exact equality of all participant
keys, metadata, 11 measurement arrays, and interpolation-label arrays. The
private reference JSON is not needed by ordinary users: it is an optional
author-side verification input, not a hidden prerequisite for cohort generation.

All named reported-analysis jobs were rerun using the packaged code. The
aggregate checks in `VALIDATION.json` cover:

- 220 HGB A-to-B window results (main tables and supporting metrics);
- 220 HGB B-to-C results (intermediate target ages);
- 720 method/age/sex/outcome curve results across the full sample and both quartiles;
- 56 selection-bias distribution cells;
- 10 age-specific interpolation-summary rows.

Comparison keys, eligibility counts and reported numeric metrics all match the
archived aggregates. The table and descriptive metrics match exactly; the curve
MAE differences are at floating-point rounding scale (maximum below `6e-15` in
native units). The checker uses absolute and relative tolerances of `1e-9`.
This confirms the reconstruction with the tested dependency versions; arbitrary
future library versions and hardware are not promised to be bitwise identical.

## Selection counts: distinct denominators

| Stage | Children |
| --- | ---: |
| No duplicate nominal ages and a nominal-age-18 raw row | 120,406 |
| At least one retained raw row after chronological-age restriction | 120,377 |
| Seven observed values in every metric, before filling | 37,311 |
| Outside that completeness-selected source cohort | 83,066 |
| Selected source cohort with either corrected aligned running target at 18 | 32,288 |

The historical no-completeness dataset contains **97,063** children with either
aligned age-18 running target. Repeating alignment on a complete annual grid
gives **95,028** such children across the current candidate population:
32,288 retained plus 62,740 completeness-excluded. The manuscript's 97,063
denominator comes from the historical preprocessing and should not be described
as the count produced by the corrected annual-grid preprocessing. Likewise,
64,775 is its arithmetic difference from 32,288, not the corrected grid's
completeness-excluded age-18-valid count of 62,740. The packaged selection-bias
table reproduces the archived corrected distribution counts exactly.

The main numerical results and final cohort counts are preserved here. This
distinction is exposed for manuscript review rather than concealed by relabeling
a stage or changing the analysis cohort.

## Protocol conventions retained for faithful reproduction

- Metadata are taken from the first retained raw record for a participant before
  sorting by chronological age, as in the original workflow.
- The completeness filter counts numeric observed values before interpolation.
- Missing-row age coordinates use the full-record median age offset in the
  reported profile. The known April-15 calendar alternative is a separate check.
- Two operations are distinguished: filling missing annual measurements and
  birthday alignment. Only dependence on the first sets an injected label.
- The latest-injected exclusion applies even when the age-18 target was not a
  source of that interpolation. Earlier injected predictors are retained.
- Native-missing models can enter CV with a missing latest running value; the
  original evaluation removes those rows within folds when ranking is needed.
  Consequently, the CSV `n_rows` is the pre-fold input count and is not always
  the number of scored predictions. Preserve this distinction when reporting N.
- Model folds are regenerated after task-level exclusions. Changing eligibility
  can change fold membership even with the same random seed.
- Actual early-stopping patience is HGB 20, GB 15, XGB 40 in the recorded helpers.
  The manuscript's generic statement of 30 rounds does not describe these three
  separate code settings. The package preserves the settings that reproduce the
  archived results.
- Baseline percentile and GCC comparisons use the original in-sample reference
  distributions; GCC requires complete running curves through the target age.

## Historical feature exploration

The archived exploratory CSVs retain the exact feature sets, future-feature
fields, age windows, target ages, sex and model choices in
`configs/feature_search_tasks.csv`. Historical profiles are clearly distinguished
from the revised labeled-data profile. Historical estimator RNG states were not
saved, so exact numerical identity for all old exploratory results is not
claimed. The default seed for new reruns is explicit. The corrected main
running-only jobs have instead been numerically verified in full.

## Generated artifacts and publication

Restricted raw data, derived participant arrays, excluded-cohort keys,
source-date provenance, and illustrative individual trajectories are generated
only on authorized users' machines under ignored artifact paths. The public
reference CSVs are aggregate results only. No GitHub or archive update is
performed by any reproduction command.

`VALIDATION_DATA.json` records exact author-side checks of the corrected study
input, the historical exploratory input, and the prior calendar-sensitivity
input. The packaged boundary-only HGB sensitivity also matches all 40 prior
window comparisons. Both feature-runner input profiles were smoke-tested, all
Python files compiled, and the six preprocessing/exclusion/calendar tests passed.
