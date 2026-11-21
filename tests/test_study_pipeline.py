import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
from preprocessing import METRICS,_complete_annual_grid,_align_metric,_numeric_values
from prepare_data import select_cohort
from analysis import build_Xy


class StudyPipelineChecks(unittest.TestCase):
    def records(self,cid=1):
        age=np.arange(6,20)
        data=dict(CROWD_ID=cid,age=age,sex=1,birth_date=pd.Timestamp('1990-01-01'),
            measurement_date=pd.to_datetime([f'{1990+a}-04-15' for a in age]))
        data.update({m:np.arange(14,dtype=float)+100 for m in METRICS})
        return pd.DataFrame(data)

    def test_complete_grid_does_not_fill_two_year_gaps(self):
        g=self.records();g['AGE_YEARS']=g.age+.7
        single=_complete_annual_grid(g.loc[g.age!=10])
        values,labels=_align_metric(single.AGE_YEARS.to_numpy(),_numeric_values(single.dash_60m))
        self.assertIsNotNone(values[2]);self.assertTrue(labels[2])
        double=_complete_annual_grid(g.loc[~g.age.isin([10,11])])
        values,_=_align_metric(double.AGE_YEARS.to_numpy(),_numeric_values(double.dash_60m))
        self.assertIsNone(values[2]);self.assertIsNone(values[3])

    def test_cohort_selection_uses_observed_coverage_and_excludes_duplicate_children(self):
        good=self.records(1)
        insufficient=self.records(2);insufficient.loc[insufficient.age>=12,'height']=np.nan
        duplicate=self.records(3);duplicate=pd.concat([duplicate,duplicate.iloc[[2]]])
        no18=self.records(4).query('age != 18')
        _,selected,counts=select_cohort(pd.concat([good,insufficient,duplicate,no18],ignore_index=True))
        self.assertEqual(selected,{1})
        self.assertEqual(counts['candidates_after_duplicate_and_age18_row_rules'],2)

    def test_latest_interpolation_exclusion_is_broader_than_target_dependency(self):
        frame=pd.DataFrame({'dash_60m_8':[100,110],'dash_60m_9':[101,111],'dash_60m_18':[90,95],
            'dash_60m_8__injected':[True,False],'dash_60m_9__injected':[False,True]})
        X,_,_,n=build_Xy(frame,['dash_60m'],[8], 'dash_60m',18,False)
        self.assertEqual(list(X.index),[1]);self.assertEqual(n,1)
        X,_,_,n=build_Xy(frame,['dash_60m'],[8,9], 'dash_60m',18,False)
        self.assertEqual(list(X.index),[0]);self.assertEqual(n,1)


if __name__=='__main__':unittest.main()
