"""Generate supplementary interpolation and cohort-selection summaries."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pandas as pd
from preprocessing import _align_metric,_complete_annual_grid,_numeric_values

ROOT=Path(__file__).resolve().parent
TARGETS={'dash_60m':'@60m_dash','run_600m':'@600m_run'}
EDGES={'dash_60m':(70,80,90,100,110,120),'run_600m':(100,120,140,160,180,200)}


def interpolation_frequency(data):
    sizes=Counter();counts=defaultdict(Counter)
    for record in data.values():
        for target in TARGETS:
            values=record['data'][target];labels=record['injected'][target]
            if values[-1] is None:continue
            sizes[target]+=1
            for i in range(10):
                counts[target,i]['available']+=int(values[i] is not None)
                counts[target,i]['filled']+=int(labels[i])
                counts[target,i]['without_next']+=int(labels[i] and values[i+1] is None)
    rows=[]
    for i in range(10):
        row={'age':i+8,'next_year_target_age':i+9}
        for target in TARGETS:
            c=counts[target,i]
            row.update({target+'_age18_cohort':sizes[target],target+'_available':c['available'],
                target+'_filled':c['filled'],target+'_filled_pct':round(100*c['filled']/c['available'],1),
                target+'_filled_without_next_target':c['without_next'],target+'_excluded_next_year':c['filled']-c['without_next']})
        rows.append(row)
    return pd.DataFrame(rows)


def selection_bias(data,excluded,raw_csv):
    counts=Counter()
    for record in data.values():
        for target in TARGETS:
            value=record['data'][target][-1]
            if value is not None:counts[target,record['sex'],'included',sum(value>=e for e in EDGES[target])]+=1
    ids={int(k) for k in excluded}
    pieces=[]
    for chunk in pd.read_csv(raw_csv,usecols=['CROWD_ID','Age','Birth','Date_measured',*TARGETS.values()],
            parse_dates=['Birth','Date_measured'],chunksize=250000,low_memory=False):
        pieces.append(chunk.loc[chunk.CROWD_ID.isin(ids)])
    raw=pd.concat(pieces,ignore_index=True).rename(columns={'Age':'age','Birth':'birth_date',
        'Date_measured':'measurement_date',**{v:k for k,v in TARGETS.items()}})
    raw=raw.loc[raw.age.between(6,19)].copy()
    raw['AGE_YEARS']=(raw.measurement_date-raw.birth_date).dt.total_seconds()/(365*24*3600)
    raw=raw.loc[raw.AGE_YEARS.between(6,19)]
    union=0;seen=set()
    for i,(cid,group) in enumerate(raw.groupby('CROWD_ID'),1):
        key=str(int(cid));seen.add(key)
        annual=_complete_annual_grid(group);any_target=False
        for target in TARGETS:
            values,_=_align_metric(annual.AGE_YEARS.to_numpy(),_numeric_values(annual[target]))
            value=values[-1]
            if value is not None:
                any_target=True
                counts[target,excluded[key]['sex'],'excluded',sum(value>=e for e in EDGES[target])]+=1
        union+=int(any_target)
        if i%10000==0:print(f'Excluded-cohort summaries: {i:,}/{len(excluded):,}',flush=True)
    if seen!=set(excluded):raise ValueError('Raw records do not cover all excluded participants')
    rows=[]
    for target in TARGETS:
        for sex in (1,2):
            for group in ('included','excluded'):
                total=sum(counts[target,sex,group,b] for b in range(7))
                for b in range(7):
                    n=counts[target,sex,group,b]
                    rows.append(dict(target=target,sex=sex,cohort=group,bin_index=b,n=n,total=total,percentage=round(100*n/total,2)))
    return pd.DataFrame(rows),union


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'artifacts'/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'results')
    parser.add_argument('--raw-csv',type=Path,help='Required to reconstruct the excluded cohort for S9/S10.')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=json.loads((args.data_dir/'study_dataset.json').read_text())
    interpolation_frequency(data).to_csv(args.output_dir/'interpolation_frequency.csv',index=False)
    if args.raw_csv:
        excluded=json.loads((args.data_dir/'excluded_cohort.json').read_text())
        table,excluded_valid=selection_bias(data,excluded,args.raw_csv)
        table.to_csv(args.output_dir/'selection_bias.csv',index=False)
        retained_valid=sum(any(r['data'][t][-1] is not None for t in TARGETS) for r in data.values())
        flow={'aligned_age18_valid_either_in_current_complete_grid_candidate_population':retained_valid+excluded_valid,
              'aligned_age18_valid_either_in_excluded_source_cohort':excluded_valid,'retained_age18_valid':retained_valid,
              'note':'The revised full annual grid gives 95028, whereas the manuscript cites a historical pre-correction denominator of 97063. These denominators must not be conflated.'}
        (args.output_dir/'selection_flow.json').write_text(json.dumps(flow,indent=2)+'\n')
        print(json.dumps(flow),flush=True)
    else:print('Wrote interpolation_frequency.csv; supply --raw-csv for S9/S10.')


if __name__=='__main__':main()
