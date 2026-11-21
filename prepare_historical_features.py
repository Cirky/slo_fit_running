"""Optional: reconstruct the pre-correction input used by archived feature searches.

The historical algorithm interpolates observed rows without first inserting
absent nominal years. It is intentionally separate from the reported revised
running analyses; it must not replace prepare_data.py.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from preprocessing import METRICS,RAW_COLUMNS,RENAME_COLUMNS,_align_metric,_numeric_values
from prepare_data import select_cohort

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'data'/'historical_feature_dataset.json')
    parser.add_argument('--verify-against',type=Path,help='Optional private pre-correction JSON.')
    args=parser.parse_args()
    raw=pd.read_csv(args.raw_csv,usecols=[*RAW_COLUMNS,'Sex'],parse_dates=['Birth','Date_measured'],low_memory=False)
    raw=raw.rename(columns={**RENAME_COLUMNS,'Sex':'sex'})
    raw,selected,_=select_cohort(raw)
    raw=raw.loc[raw.CROWD_ID.isin(selected)]
    reference=json.loads(args.verify_against.read_text()) if args.verify_against else None
    result={}
    for i,(cid,group) in enumerate(raw.groupby('CROWD_ID'),1):
        key=str(int(cid))
        record={'sex':int(group.sex.iloc[0]),'birth':group.birth_date.iloc[0].date().isoformat(),'data':{}}
        # Preserve original metadata selection before sorting observations.
        group=group.sort_values('AGE_YEARS')
        for metric in METRICS:
            values,_=_align_metric(group.AGE_YEARS.to_numpy(),_numeric_values(group[metric]))
            record['data'][metric]=values
        if reference is not None and record!=reference[key]:raise ValueError('Historical input does not match the private archived dataset.')
        result[key]=record
        if i%5000==0:print(f'Historical input: {i:,}/{len(selected):,}',flush=True)
    if reference is not None and result.keys()!=reference.keys():raise ValueError('Historical cohorts differ')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result))
    print(f'Wrote historical exploratory dataset; exact reference match={bool(args.verify_against)}',flush=True)


if __name__=='__main__':main()
