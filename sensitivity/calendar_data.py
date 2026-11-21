"""Optional calendar-based sensitivity preprocessing; does not replace study data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import argparse
import numpy as np
import pandas as pd
from functools import lru_cache
from preprocessing import METRICS,RAW_COLUMNS,RENAME_COLUMNS,_numeric_values

def calendar_grid(group):
    birth = pd.Timestamp(group.birth_date.iloc[0])
    # Nominal age is completed age at the annual April assessment.
    year_offset = birth.year + int((birth.month, birth.day) > (4, 15))
    if not (group.measurement_date.dt.year - group.age == year_offset).all():
        raise ValueError('Nominal age does not match the April-15 calendar rule')
    if not ((group.measurement_date.dt.month == 4) & (group.measurement_date.dt.day == 15)).all():
        raise ValueError('Unexpected assessment date')
    annual = group.set_index('age').reindex(range(6, 20))
    annual['age'] = annual.index
    annual['measurement_date'] = pd.to_datetime([f'{year_offset+a}-04-15' for a in annual.index])
    return annual, birth

@lru_cache(maxsize=65536)
def calendar_geometry(birth, year_offset):
    dates = np.array([np.datetime64(f'{year_offset+a}-04-15', 'D').astype('int64') for a in range(6,20)])
    geometry = []
    for age in range(8, 19):
        birthday = birth + pd.DateOffset(years=age)
        day = birthday.to_datetime64().astype('datetime64[D]').astype('int64')
        lo = (birthday - pd.DateOffset(months=2)).to_datetime64().astype('datetime64[D]').astype('int64')
        hi = (birthday + pd.DateOffset(months=2)).to_datetime64().astype('datetime64[D]').astype('int64')
        near = np.flatnonzero((dates >= lo) & (dates <= hi))
        if len(near):
            indices = [near[np.argmin(np.abs(dates[near] - day))]]
            fraction = 0.0
        else:
            before, after = np.flatnonzero(dates < day), np.flatnonzero(dates > day)
            indices = [before.max(), after.min()] if len(before) and len(after) else []
            if indices:
                i, j = indices
                fraction = (day-dates[i]) / (dates[j]-dates[i])
            else:
                fraction = 0.0
        geometry.append((indices, fraction))
    return dates, geometry

def align_calendar(annual, birth, metric):
    dates, geometry = calendar_geometry(birth, int(annual.measurement_date.iloc[0].year)-6)
    values = _numeric_values(annual[metric]).copy()
    filled = np.zeros(len(values), dtype=bool)
    source_date = np.where(np.isfinite(values), dates, -1)
    for i in range(1, len(values)-1):
        if np.isnan(values[i]) and np.isfinite(values[i-1]) and np.isfinite(values[i+1]):
            values[i] = (values[i-1] + values[i+1]) / 2
            filled[i] = True
            source_date[i] = max(source_date[i-1], source_date[i+1])
    result, labels, latest = [], [], []
    for indices, fraction in geometry:
        if len(indices) == 1:
            value = values[indices[0]]
        elif len(indices) == 2:
            i,j = indices
            value = values[i] + (values[j]-values[i]) * fraction
        else:
            value = np.nan
        if not np.isfinite(value):
            result.append(None); labels.append(False); latest.append(-1)
        else:
            result.append(round(float(value), 1))
            labels.append(bool(any(filled[i] for i in indices)))
            latest.append(int(max(source_date[i] for i in indices)))
    return result, labels, latest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw-csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts'/'sensitivity'/'calendar_dataset.json')
    parser.add_argument('--verify-against',type=Path,help='Optional private prior calendar-sensitivity JSON.')
    args=parser.parse_args();data=json.loads(args.data.read_text());ids={int(k) for k in data}
    parts=[]
    for chunk in pd.read_csv(args.raw_csv,usecols=RAW_COLUMNS,parse_dates=['Birth','Date_measured'],chunksize=250000,low_memory=False):
        parts.append(chunk.loc[chunk.CROWD_ID.isin(ids)])
    raw=pd.concat(parts,ignore_index=True).rename(columns=RENAME_COLUMNS)
    raw=raw.loc[raw.age.between(6,19)].copy()
    legacy_age=(raw.measurement_date-raw.birth_date).dt.total_seconds()/(365*24*3600)
    raw=raw.loc[legacy_age.between(6,19)]
    result={};changes={m:0 for m in METRICS}
    for i,(cid,group) in enumerate(raw.groupby('CROWD_ID'),1):
        key=str(int(cid));annual,birth=calendar_grid(group)
        record=dict(sex=data[key]['sex'],birth=data[key]['birth'],data={},injected={},source_latest={})
        for metric in METRICS:
            values,labels,dates=align_calendar(annual,birth,metric)
            record['data'][metric],record['injected'][metric],record['source_latest'][metric]=values,labels,dates
            changes[metric]+=sum(a!=b for a,b in zip(values,data[key]['data'][metric]))
        result[key]=record
        if i%5000==0:print(f'Calendar sensitivity: {i:,}/{len(data):,}',flush=True)
    if result.keys()!=data.keys():raise ValueError('Calendar cohort changed')
    if args.verify_against and result!=json.loads(args.verify_against.read_text()):
        raise ValueError('Calendar sensitivity differs from the private prior run.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result))
    args.output.with_suffix('.changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    print('Wrote calendar sensitivity dataset; original input was preserved.',flush=True)


if __name__=='__main__':main()
