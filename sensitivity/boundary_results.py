"""Optional before/after HGB analysis excluding already-available targets.

Uses the original data, and only age-17-ending windows forecasting age 18.
Outer folds are formed after filtering, preserving the original pipeline rule.
Changing cohort size can change fold and early-stopping validation membership.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from analysis import load_flatten,build_Xy,cv_evaluate_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True,help='Private source_latest.json from audit_dates.py.')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts'/'sensitivity'/'boundary_results.csv')
    args=parser.parse_args();raw=json.loads(args.data.read_text());dates=json.loads(args.provenance.read_text());flags={}
    for target in ('dash_60m','run_600m'):
        flags[target]=[int(cid) for cid,r in raw.items() if r['data'][target][9] is not None
            and r['data'][target][10] is not None and not r['injected'][target][9]
            and dates[cid][target][9]>=0 and dates[cid][target][10]<=dates[cid][target][9]]
    print('Flagged records per outcome:',{t:len(v) for t,v in flags.items()},flush=True)
    del raw,dates
    frame=load_flatten(str(args.data),(1,2),measurements=list(flags));rows=[]
    for target in flags:
        for sex in (1,2):
            subset=frame.loc[frame.sex==sex]
            for a in range(8,18):
                X,y,_,_=build_Xy(subset,[target],list(range(a,18)),target,18,False,injected_latest_features=[target])
                for variant in ('original','boundary_excluded'):
                    keep=~X.index.isin(flags[target]) if variant=='boundary_excluded' else [True]*len(X)
                    used_x,used_y=X.loc[keep],y.loc[keep]
                    rank=subset[f'{target}_17'].reindex(used_x.index)
                    metrics=cv_evaluate_model('HGB',HistGradientBoostingRegressor(random_state=42),used_x,used_y,
                        random_state=42,n_splits=5,rank_values=rank,test_split_handling='bottom')
                    rows.append(dict(target=target,sex=sex,start_age=a,latest_age=17,target_age=18,variant=variant,n_rows=len(used_x),**metrics))
            print(f'Boundary sensitivity: {target}, sex={sex}',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,index=False)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
