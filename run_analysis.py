"""Run named reported-study experiments without editing source-code constants."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from analysis import load_flatten, build_Xy, cv_evaluate_model
from helper_functions import GCCRegressor, PercentileRegressor
from prepare_data import sha256

ROOT=Path(__file__).resolve().parent
FIELDS=['sex','model','target','target_age','features','features_future','ages','start_age','cutoff_age','group',
        'n_rows','n_excluded_injected','mae','pi_95','mae_pi','rmse','relative_error_pct','rmse_relative_error']


def versions():
    return {p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','xgboost','matplotlib','scipy','threadpoolctl')}


def tasks(spec, config):
    for sex in config['sexes']:
        for target in config['targets']:
            for c in spec['target_ages']:
                for b in range(8,c):
                    for a in (range(8,b+1) if spec['windows']=='all_contiguous' else [8]):
                        for model in spec['models']:
                            yield sex,target,c,a,b,model


def estimator(name, config):
    seed=config['estimator_random_state']
    return {'LR':lambda:LinearRegression(),
        'HGB':lambda:HistGradientBoostingRegressor(random_state=seed),
        'GB':lambda:GradientBoostingRegressor(random_state=seed),
        'XGB':lambda:XGBRegressor(tree_method='hist',device='cpu',random_state=seed,n_jobs=config['xgb_n_jobs'])}[name]()


def make_xy(frame,target,c,a,b,name):
    if name=='GCC':
        X,y,_,_=build_Xy(frame,[target],list(range(8,c+1)),target,c,True,exclude_injected_latest=False)
        injected=frame[f'{target}_{b}__injected'].reindex(X.index)
        excluded=int(injected.sum())
        return X.loc[~injected],y.loc[~injected],excluded
    X,y,_,excluded=build_Xy(frame,[target],list(range(a,b+1)),target,c,name in ('LR','GB'),injected_latest_features=[target])
    return X,y,excluded


def baseline(name,X,y,target,c,b,group,neighbors):
    # Preserve the original reference-population and in-sample protocol.
    rank_col=f'{target}_{b}'
    keep=X[rank_col].notna()
    X,y=X.loc[keep],y.loc[keep]
    model=GCCRegressor(target,b,c,k_neighbors=neighbors) if name=='GCC' else PercentileRegressor(target,c,b)
    model.fit(X,y)
    if group!='all':
        rank=X[rank_col]
        keep=rank >= np.quantile(rank,.75) if group=='bottom' else rank <= np.quantile(rank,.25)
        X,y=X.loc[keep],y.loc[keep]
    prediction=model.predict(X)
    return float(np.mean(np.abs(y.to_numpy()-prediction)))


def run_job(frame,job,spec,config,outdir,fingerprint,max_tasks=None):
    path=outdir/f'{job}.csv'
    meta_path=outdir/f'{job}.manifest.json'
    metadata={'input_and_code':fingerprint,'job':job,'specification':spec,'experiments':config,'versions':versions()}
    done=set()
    if path.exists():
        if not meta_path.exists():raise ValueError(f'{path}: missing manifest; choose a fresh output directory')
        previous=json.loads(meta_path.read_text())
        if previous['configuration']!=metadata:raise ValueError(f'{path}: different input, code or settings; use a fresh output directory')
        import pandas as pd
        existing=pd.read_csv(path)
        done=set(existing[['sex','target','target_age','start_age','cutoff_age','model']].itertuples(index=False,name=None))
    meta_path.write_text(json.dumps({'configuration':metadata,'complete':False},indent=2)+'\n')
    plan=list(tasks(spec,config))
    completed=0
    start=time.monotonic()
    for sex,target,c,a,b,name in plan:
        key=(sex,target,c,a,b,name)
        if key in done:continue
        if max_tasks is not None and completed>=max_tasks:break
        subset=frame.loc[frame.sex==sex]
        X,y,excluded=make_xy(subset,target,c,a,b,name)
        row=dict(sex=sex,model=name,target=target,target_age=c,features=target,features_future='',
            ages=','.join(map(str,range(a,b+1))),start_age=a,cutoff_age=b,group=spec['group'],
            n_rows=len(X),n_excluded_injected=excluded)
        if name in ('GCC','Percentile'):
            mae=baseline(name,X,y,target,c,b,spec['group'],config['gcc_neighbors'])
            row.update(mae=mae,pi_95='',mae_pi='',rmse='',relative_error_pct='',rmse_relative_error='')
        else:
            rank=subset[f'{target}_{b}'].reindex(X.index)
            result=cv_evaluate_model(name,estimator(name,config),X,y,
                n_splits=config['cv_folds'],random_state=config['cv_random_state'],
                rank_values=rank,test_split_handling=spec['group'])
            row.update(mae=result['mae_mean'],pi_95=result['pi_95'],mae_pi=f"{result['mae_mean']:.1f} | {result['pi_95']:.1f}",
                rmse=result['rmse_mean'],relative_error_pct=result['relative_error_pct_mean'],
                rmse_relative_error=f"{result['rmse_mean']:.1f} | {result['relative_error_pct_mean']:.1f}")
        with path.open('a',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS)
            if stream.tell()==0:writer.writeheader()
            writer.writerow(row)
        done.add(key);completed+=1
        if completed%10==0 or completed==1:
            print(f'{job}: {len(done)}/{len(plan)} configurations; {time.monotonic()-start:.0f}s',flush=True)
    complete=len(done)==len(plan)
    meta_path.write_text(json.dumps({'configuration':metadata,'complete':complete,'completed':len(done),'expected':len(plan)},indent=2)+'\n')
    print(f'{job}: complete={complete}; rows={len(done)}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts'/'data'/'study_dataset.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'study.json')
    parser.add_argument('--job',default='all',help='all or a job name in configs/study.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'results')
    parser.add_argument('--max-tasks',type=int,help='Smoke check only; records an incomplete job manifest.')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())['experiments']
    expected_es={'validation_fraction':.2,'hgb_patience':20,'gb_patience':15,'xgb_patience':40,
        'xgb_max_estimators':100,'xgb_validation_split_seed':1}
    if not config['exclude_injected_latest'] or config['early_stopping']!=expected_es:
        raise ValueError('Exclusion/early-stopping settings differ from the preserved numerical implementation.')
    jobs=config['jobs'] if args.job=='all' else {args.job:config['jobs'][args.job]}
    frame=load_flatten(str(args.data),config['sexes'],measurements=config['targets'])
    fingerprint={'data_sha256':sha256(args.data),'code':{p:sha256(ROOT/p) for p in ('analysis.py','helper_functions.py','run_analysis.py')}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        for job,spec in jobs.items():run_job(frame,job,spec,config,args.output_dir,fingerprint,args.max_tasks)


if __name__=='__main__':main()
