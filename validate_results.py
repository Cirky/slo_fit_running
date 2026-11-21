"""Compare regenerated aggregate outputs against the archived reported revision."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def compare(actual,expected,keys,metrics):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():raise ValueError('Duplicate result keys')
    merged=expected.merge(actual,on=keys,suffixes=('_reference','_generated'),how='outer',indicator=True)
    unmatched=int((merged._merge!='both').sum())
    report={'rows_reference':len(expected),'rows_generated':len(actual),'unmatched_rows':unmatched,'metrics':{}}
    for metric in metrics:
        a=pd.to_numeric(merged[metric+'_reference'],errors='coerce')
        b=pd.to_numeric(merged[metric+'_generated'],errors='coerce')
        same=np.isclose(a,b,rtol=1e-9,atol=1e-9,equal_nan=True)
        report['metrics'][metric]={'different_rows':int((~same).sum()),'max_absolute_difference':float((a-b).abs().max()) if (a-b).notna().any() else 0.0}
    report['match']=not unmatched and all(v['different_rows']==0 for v in report['metrics'].values())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'artifacts'/'results')
    parser.add_argument('--report',type=Path,help='Optional aggregate JSON validation report.')
    parser.add_argument('--allow-partial',action='store_true',help='Only check files present; does not claim full reproduction.')
    args=parser.parse_args();reports={};missing=[]
    for name in ('tables_ab','tables_bc','curves_bottom','curves_all','curves_top','selection_bias','interpolation_frequency'):
        path=args.results_dir/f'{name}.csv'
        if not path.exists():missing.append(name);continue
        actual=pd.read_csv(path)
        if name.startswith('curves_'):
            group=name.split('_',1)[1]
            if group=='bottom':expected=pd.read_csv(ROOT/'reference_results'/'main_curves.csv')
            else:
                expected=pd.read_csv(ROOT/'reference_results'/'supplement_curves.csv').query('group == @group')
            keys=['sex','target','cutoff_age','model'];metrics=['mae','n_rows','n_excluded_injected']
        elif name in ('tables_ab','tables_bc'):
            expected=pd.read_csv(ROOT/'reference_results'/f'{name}.csv')
            keys=['sex','target','target_age','ages'];metrics=['mae','pi_95','rmse','relative_error_pct','n_rows','n_excluded_injected']
            actual['ages']=actual.ages.astype(str);expected['ages']=expected.ages.astype(str)
        elif name=='selection_bias':
            expected=pd.read_csv(ROOT/'reference_results'/f'{name}.csv')
            keys=['target','sex','cohort','bin_index'];metrics=['n','total','percentage']
        else:
            expected=pd.read_csv(ROOT/'reference_results'/f'{name}.csv')
            keys=['age'];metrics=[c for c in expected if c!='age']
        reports[name]=compare(actual,expected,keys,metrics)
    result={'complete':not missing,'missing_outputs':missing,'checks':reports,
            'all_present_outputs_match':all(v['match'] for v in reports.values()) and bool(reports)}
    print(json.dumps(result,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_present_outputs_match'] or (missing and not args.allow_partial):raise SystemExit(1)


if __name__=='__main__':main()
