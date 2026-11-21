"""Replay explicitly archived exploratory feature/window configurations.

Select one source profile from configs/feature_search_profiles.json. Historical
profiles use the historical input, whereas the corrected profile uses the
reported labeled study dataset. These optional tasks do not generate the main
running-only tables. Historical estimator random states were not archived;
seed 42 is supplied for repeatable reruns, not claimed to recover lost states.
"""
import argparse
import csv
import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
import analysis

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,help='Name from configs/feature_search_profiles.json.')
    parser.add_argument('--data',type=Path,help='Override the profile-specific dataset.')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-tasks',type=int,help='Optional smoke-check limit.')
    args=parser.parse_args()
    profile=json.loads((ROOT/'configs'/'feature_search_profiles.json').read_text())[args.profile]
    data=args.data or ROOT/'artifacts'/'data'/profile['input_filename']
    plan=pd.read_csv(ROOT/'configs'/'feature_search_tasks.csv').query('profile == @args.profile').fillna('')
    if plan.empty:raise ValueError('Unknown or empty feature profile')
    analysis.EXCLUDE_INJECTED_LATEST=profile['exclude_injected_latest']
    frame=analysis.load_flatten(str(data),(1,2),measurements=json.loads((ROOT/'configs'/'study.json').read_text())['preprocessing']['completeness_metrics'])
    output=args.output or ROOT/'artifacts'/'features'/f'{args.profile}.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Choose a fresh output file for this feature-search run.')
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=None
        with threadpool_limits(limits=2):
            for i,task in enumerate(plan.to_dict('records')):
                if args.max_tasks is not None and i>=args.max_tasks:break
                features=str(task['features']).split(',');future=[x for x in str(task['features_future']).split(',') if x]
                ages=[int(x) for x in str(task['ages']).split(',')]
                target=task['target'];c=int(task['target_age']);sex=int(task['sex'])
                name=task['model']
                model={'HistGBRegressor':lambda:HistGradientBoostingRegressor(random_state=42),
                    'LinearRegression':lambda:LinearRegression(),
                    'XGBRegressor':lambda:XGBRegressor(tree_method='hist',device='cpu',random_state=42,n_jobs=4)}[name]()
                subset=frame.loc[frame.sex==sex]
                X,y,used,excluded=analysis.build_Xy(subset,features,ages,target,c,name=='LinearRegression',future_features=future,
                    exclude_injected_latest=profile['exclude_injected_latest'],
                    injected_latest_features=[*features,*([target] if profile['group']!='all' else [])])
                rank=used[f'{target}_{ages[-1]}'] if profile['group']!='all' else None
                result=analysis.cv_evaluate_model(name,model,X,y,random_state=42,n_splits=5,rank_values=rank,test_split_handling=profile['group'])
                row={**task,'n_rows':len(X),'n_excluded_injected':excluded,**result}
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush()
                if i%10==0:print(f'{args.profile}: {i+1}/{len(plan)}',flush=True)


if __name__=='__main__':main()
