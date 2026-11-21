"""Export tables/figures from regenerated or archived aggregate result CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MODELS=('LR','HGB','GB','XGB','Percentile','GCC')


def table(frame,rows,cols,row_name,col_name,field,caption,label):
    lookup={(int(r[rows]),int(r[cols])):str(r[field]) for _,r in frame.iterrows()}
    ri=sorted(frame[rows].astype(int).unique());ci=sorted(frame[cols].astype(int).unique())
    lines=['\\begin{table}[ht]','\\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
        '\\begin{tabular}{l'+'r'*len(ci)+'}','\\toprule',
        f'{row_name}$\\downarrow$ / {col_name}$\\rightarrow$ & '+' & '.join(map(str,ci))+' \\\\','\\midrule']
    for r in ri:lines.append(str(r)+' & '+' & '.join(lookup.get((r,c),'') for c in ci)+' \\\\')
    return '\n'.join(lines+['\\bottomrule','\\end{tabular}','\\end{table}',''])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--reference',action='store_true',help='Use archived reported aggregate results; no restricted data required.')
    choice.add_argument('--results-dir',type=Path,help='Directory containing run_analysis.py outputs.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'exports')
    args=parser.parse_args()
    source=ROOT/'reference_results' if args.reference else (args.results_dir or ROOT/'artifacts'/'results')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    ab=pd.read_csv(source/'tables_ab.csv');bc=pd.read_csv(source/'tables_bc.csv')
    for frame in (ab,bc):
        if 'start_age' not in frame:frame['start_age']=frame.ages.astype(str).map(lambda s:int(s.split(',')[0]))
        frame['mae_pi']=frame.apply(lambda r:f'{float(r.mae):.1f} | {float(r.pi_95):.1f}',axis=1)
        frame['rmse_relative_error']=frame.apply(lambda r:f'{float(r.rmse):.1f} | {float(r.relative_error_pct):.1f}',axis=1)
    for target in ('dash_60m','run_600m'):
        unit='deciseconds' if target=='dash_60m' else 'seconds'
        for sex in (1,2):
            group=ab.loc[(ab.target==target)&(ab.sex==sex)]
            for field in ('mae_pi','rmse_relative_error'):
                text=table(group,'start_age','cutoff_age','A','B',field,
                    f'{target}, sex {sex}: {field.replace("_"," ")} ({unit}; relative error in percent).',f'tab:{target}-{sex}-{field}')
                (args.output_dir/f'{target}_sex{sex}_{field}.tex').write_text(text)
            group=bc.loc[(bc.target==target)&(bc.sex==sex)]
            text=table(group,'cutoff_age','target_age','B','C','mae_pi',
                f'{target}, sex {sex}: intermediate target ages; MAE and PI half-width ({unit}).',f'tab:{target}-{sex}-bc')
            (args.output_dir/f'{target}_sex{sex}_bc.tex').write_text(text)
    descriptive_tables=0
    if (source/'selection_bias.csv').exists():
        distribution=pd.read_csv(source/'selection_bias.csv')
        for target in ('dash_60m','run_600m'):
            part=distribution.loc[distribution.target==target]
            lines=['\\begin{table}[ht]','\\centering',f'\\caption{{Age-18 {target} distributions: included and completeness-excluded cohorts.}}',
                '\\begin{tabular}{lrrrr}','\\toprule','Bin & Boys included (\\%) & Boys excluded (\\%) & Girls included (\\%) & Girls excluded (\\%) \\\\','\\midrule']
            for b in range(7):
                values=[float(part.loc[(part.sex==sex)&(part.cohort==cohort)&(part.bin_index==b),'percentage'].iloc[0]) for sex,cohort in ((1,'included'),(1,'excluded'),(2,'included'),(2,'excluded'))]
                edges=[70,80,90,100,110,120] if target=='dash_60m' else [100,120,140,160,180,200]
                label=f'$t<{edges[0]}$' if b==0 else (f'$t\\ge {edges[-1]}$' if b==6 else f'${edges[b-1]}\\le t<{edges[b]}$')
                lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+' \\\\')
            lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
            (args.output_dir/f'{target}_selection_bias.tex').write_text('\n'.join(lines)+'\n');descriptive_tables+=1
    if (source/'interpolation_frequency.csv').exists():
        freq=pd.read_csv(source/'interpolation_frequency.csv')
        lines=['\\begin{table}[ht]','\\centering','\\caption{Missing-value interpolation by age within age-18-valid running cohorts.}',
            '\\begin{tabular}{rrrrrrr}','\\toprule','Age & 60 m available & Filled & Filled (\\%) & 600 m available & Filled & Filled (\\%) \\\\','\\midrule']
        for _,r in freq.iterrows():
            lines.append(f"{int(r.age)} & {int(r.dash_60m_available)} & {int(r.dash_60m_filled)} & {r.dash_60m_filled_pct:.1f} & {int(r.run_600m_available)} & {int(r.run_600m_filled)} & {r.run_600m_filled_pct:.1f} \\\\")
        lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
        (args.output_dir/'interpolation_frequency.tex').write_text('\n'.join(lines)+'\n');descriptive_tables+=1
    if args.reference:
        bottom=pd.read_csv(source/'main_curves.csv');bottom['group']='bottom'
        curves=pd.concat([bottom,pd.read_csv(source/'supplement_curves.csv')],ignore_index=True)
    else:
        curves=pd.concat([pd.read_csv(source/f'curves_{g}.csv') for g in ('bottom','all','top')],ignore_index=True)
    for group in ('bottom','all','top'):
        for target in ('dash_60m','run_600m'):
            fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
            for sex,ax in zip((1,2),axes):
                for model in MODELS:
                    values=curves.loc[(curves.group==group)&(curves.target==target)&(curves.sex==sex)&(curves.model==model)].sort_values('cutoff_age')
                    if values.cutoff_age.tolist()!=list(range(8,18)):raise ValueError(f'Incomplete curve: {group} {target} {sex} {model}')
                    ax.plot(values.cutoff_age,values.mae,marker='o',label=model)
                ax.set_title(('Boys' if sex==1 else 'Girls')+f' — {target} — {group}')
                ax.set_ylabel('MAE (deciseconds)' if target=='dash_60m' else 'MAE (seconds)')
                ax.grid(alpha=.3);ax.legend(ncol=3)
            axes[-1].set_xlabel('Latest aligned predictor age');axes[-1].set_xticks(range(8,18))
            fig.tight_layout();fig.savefig(args.output_dir/f'{group}_{target}.png',dpi=200);plt.close(fig)
    print(f'Exported {12+descriptive_tables} table fragments and 6 performance figures to {args.output_dir}')


if __name__=='__main__':main()
