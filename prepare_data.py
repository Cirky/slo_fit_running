"""Rebuild the reported cohort and injected labels directly from authorized raw data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import METRICS, RAW_COLUMNS, RENAME_COLUMNS, _complete_annual_grid, _align_metric, _numeric_values

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_cohort(raw):
    """Original selection order: duplicates, age-18 row, date range, completeness."""
    raw = raw.loc[raw.age.between(6,19)].copy()
    duplicated = raw.loc[raw.duplicated(['CROWD_ID','age'],keep=False),'CROWD_ID'].unique()
    raw = raw.loc[~raw.CROWD_ID.isin(duplicated)]
    has18 = raw.loc[raw.age.eq(18),'CROWD_ID'].unique()
    raw = raw.loc[raw.CROWD_ID.isin(has18)].copy()
    counts = {'candidates_after_duplicate_and_age18_row_rules': len(has18)}
    raw['AGE_YEARS'] = (raw.measurement_date-raw.birth_date).dt.total_seconds()/(365*24*3600)
    raw = raw.loc[raw.AGE_YEARS.between(6,19)].copy()
    for metric in METRICS:
        raw[metric] = pd.to_numeric(raw[metric], errors='coerce')
    coverage = raw.groupby('CROWD_ID')[METRICS].count()
    selected = set(coverage.index[coverage.ge(7).all(axis=1)])
    counts.update(candidates_after_chronological_age_restriction=len(coverage),
                  selected_source_cohort=len(selected), excluded_source_cohort=len(coverage)-len(selected))
    return raw, selected, counts


def check_config(config):
    p = config['preprocessing']
    expected = {'nominal_age_min':6,'nominal_age_max':19,'chronological_age_min':6,
        'chronological_age_max':19,'minimum_observed_values_per_metric':7,
        'completeness_metrics':METRICS,'duplicate_policy':'exclude_child_with_any_duplicate_nominal_age',
        'require_nominal_age_18_row':True,'age_year_days':365,
        'alignment_tolerance_years':2/12,'aligned_ages':list(range(8,19)),'round_decimals':1}
    if p != expected:
        raise ValueError('This implementation reproduces the recorded study profile; config preprocessing differs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-csv',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'data')
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'study.json')
    parser.add_argument('--verify-against',type=Path,help='Optional private original corrected/labeled JSON; never distributed.')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    check_config(config)
    raw=pd.read_csv(args.raw_csv,usecols=[*RAW_COLUMNS,'Sex'],parse_dates=['Birth','Date_measured'],low_memory=False)
    raw=raw.rename(columns={**RENAME_COLUMNS,'Sex':'sex'})
    raw,selected,counts=select_cohort(raw)
    print(json.dumps(counts),flush=True)
    excluded={str(int(cid)):{'sex':int(g.sex.iloc[0]),'birth':g.birth_date.iloc[0].date().isoformat()}
              for cid,g in raw.loc[~raw.CROWD_ID.isin(selected)].groupby('CROWD_ID')}
    raw=raw.loc[raw.CROWD_ID.isin(selected)].copy()
    expected=json.loads(args.verify_against.read_text()) if args.verify_against else None
    if expected is not None and set(expected)!={str(int(cid)) for cid in selected}:
        raise ValueError('Reconstructed cohort IDs differ from the private reference.')
    children={}
    for i,(cid,group) in enumerate(raw.groupby('CROWD_ID'),1):
        key=str(int(cid))
        record={'sex':int(group.sex.iloc[0]),'birth':group.birth_date.iloc[0].date().isoformat(),'data':{},'injected':{}}
        annual=_complete_annual_grid(group)
        ages=annual.AGE_YEARS.to_numpy()
        for metric in METRICS:
            values,labels=_align_metric(ages,_numeric_values(annual[metric]))
            record['data'][metric],record['injected'][metric]=values,labels
        if expected is not None and record!=expected[key]:
            raise ValueError('A reconstructed participant differs from the original private data (values, metadata or labels).')
        children[key]=record
        if i%5000==0:print(f'Rebuilt and checked {i:,}/{len(selected):,} records',flush=True)
    del expected
    valid=Counter()
    for record in children.values():
        for target in ('dash_60m','run_600m'):
            if record['data'][target][-1] is not None:valid['valid_age18_'+target]+=1
        if any(record['data'][t][-1] is not None for t in ('dash_60m','run_600m')):
            valid['valid_age18_either_running_target']+=1
            valid['valid_age18_either_'+('boys' if record['sex']==1 else 'girls')]+=1
    counts.update(valid)
    if counts!=config['expected_counts']:
        raise ValueError(f'Study cohort counts do not match the configured data release: {counts}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'study_dataset.json').write_text(json.dumps(children),encoding='utf-8')
    (args.output_dir/'excluded_cohort.json').write_text(json.dumps(excluded),encoding='utf-8')
    summary={'profile':config['profile'],'counts':counts,'raw_csv_sha256':sha256(args.raw_csv),
        'exact_private_reference_match':bool(args.verify_against),
        'selection':'observed pre-interpolation coverage across nominal ages 6..19; 7 or more values in every metric',
        'note':'37311 is the retained source cohort; 32288 have at least one valid aligned age-18 running target.'}
    (args.output_dir/'cohort_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
