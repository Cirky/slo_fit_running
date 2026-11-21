"""Optional observation-source audit of the original revised running predictors."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import argparse
import numpy as np
import pandas as pd
from preprocessing import _numeric_values,_complete_annual_grid

def align_with_sources(annual, metric):
    ages = annual.AGE_YEARS.to_numpy()
    values = _numeric_values(annual[metric]).copy()
    dates = annual.measurement_date.to_numpy(dtype="datetime64[D]").astype("int64")
    sources = [({i} if np.isfinite(v) else set()) for i, v in enumerate(values)]
    injected = np.zeros(len(values), dtype=bool)
    for i in range(1, len(values) - 1):
        if np.isnan(values[i]) and np.isfinite(values[i-1]) and np.isfinite(values[i+1]):
            values[i] = (values[i-1] + values[i+1]) / 2
            sources[i] = sources[i-1] | sources[i+1]
            injected[i] = True
    aligned, labels, latest, masks = [], [], [], []
    for age in range(8, 19):
        near = np.flatnonzero(np.abs(ages - age) <= 2/12)
        if len(near):
            idx = near[np.argmin(np.abs(ages[near] - age))]
            indices = [idx]
            value = values[idx]
        else:
            before, after = np.flatnonzero(ages < age), np.flatnonzero(ages > age)
            indices = [before.max(), after.min()] if len(before) and len(after) else []
            if indices:
                i, j = indices
                value = values[i] + (values[j] - values[i]) * (age - ages[i]) / (ages[j] - ages[i])
            else:
                value = np.nan
        if not np.isfinite(value):
            aligned.append(None)
            labels.append(False)
            latest.append(-1)
            masks.append(0)
            continue
        source = set().union(*(sources[i] for i in indices))
        if not source or any(dates[i] == np.iinfo(np.int64).min for i in source):
            raise ValueError("Nonmissing value has missing source measurement dates")
        aligned.append(round(float(value), 1))
        labels.append(bool(any(injected[i] for i in indices)))
        latest.append(max(dates[i] for i in source))
        masks.append(sum(1 << i for i in source))
    return aligned, labels, latest, masks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw-csv',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts'/'sensitivity')
    args=parser.parse_args();data=json.loads(args.data.read_text());ids={int(k) for k in data}
    columns=['CROWD_ID','Age','Birth','Date_measured','@60m_dash','@600m_run']
    parts=[]
    for chunk in pd.read_csv(args.raw_csv,usecols=columns,parse_dates=['Birth','Date_measured'],chunksize=250000,low_memory=False):
        parts.append(chunk.loc[chunk.CROWD_ID.isin(ids)])
    raw=pd.concat(parts,ignore_index=True).rename(columns={'Age':'age','Birth':'birth_date','Date_measured':'measurement_date','@60m_dash':'dash_60m','@600m_run':'run_600m'})
    raw=raw.loc[raw.age.between(6,19)].copy()
    raw['AGE_YEARS']=(raw.measurement_date-raw.birth_date).dt.total_seconds()/(365*24*3600)
    raw=raw.loc[raw.AGE_YEARS.between(6,19)]
    provenance={};counts={}
    for i,(cid,group) in enumerate(raw.groupby('CROWD_ID'),1):
        key=str(int(cid));annual=_complete_annual_grid(group);provenance[key]={}
        for target in ('dash_60m','run_600m'):
            values,labels,latest,_=align_with_sources(annual,target)
            if values!=data[key]['data'][target] or labels!=data[key]['injected'][target]:raise ValueError('Observation-source replay differs from study input.')
            provenance[key][target]=[int(day) for day in latest]
            for b in range(10):
                if latest[-1]<0 or latest[b]<0 or labels[b]:continue
                c=counts.setdefault((target,b+8),[0,0,0]);c[0]+=1
                c[1]+=int(max(latest[:b+1])>latest[b]);c[2]+=int(latest[-1]<=latest[b])
        if i%5000==0:print(f'Source-date audit: {i:,}/{len(data):,}',flush=True)
    if set(provenance)!=set(data):raise ValueError('Incomplete raw cohort')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # This file contains participant keys and belongs only in private artifacts.
    (args.output_dir/'source_latest.json').write_text(json.dumps(provenance))
    out=pd.DataFrame([dict(target=k[0],latest_predictor_age=k[1],eligible=v[0],earlier_source_after_cutoff=v[1],target_already_available=v[2]) for k,v in sorted(counts.items())])
    out.to_csv(args.output_dir/'source_date_summary.csv',index=False)
    print(out.to_string(index=False),flush=True)
    print('This traces measurement-value sources; it does not establish that full-record grid-timing estimates are cutoff-invariant.')


if __name__=='__main__':main()
