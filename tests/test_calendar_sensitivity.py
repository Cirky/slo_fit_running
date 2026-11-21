import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
from sensitivity.calendar_data import calendar_grid, align_calendar


class CalendarChecks(unittest.TestCase):
    def group(self, birth='1998-06-20'):
        birth = pd.Timestamp(birth)
        offset = birth.year + int((birth.month,birth.day) > (4,15))
        return pd.DataFrame(dict(age=range(6,20), birth_date=birth,
            measurement_date=pd.to_datetime([f'{offset+a}-04-15' for a in range(6,20)]),
            dash_60m=np.arange(14,dtype=float)+90))

    def test_missing_row_has_fixed_date_with_or_without_future_rows(self):
        full = self.group().query('age != 10')
        a,birth = calendar_grid(full)
        partial,_ = calendar_grid(full.query('age <= 12'))
        self.assertEqual(a.loc[10,'measurement_date'], pd.Timestamp('2009-04-15'))
        self.assertEqual(a.loc[10,'measurement_date'],partial.loc[10,'measurement_date'])
        expected = align_calendar(a,birth,'dash_60m')[0][:4]
        self.assertEqual(expected, align_calendar(partial,birth,'dash_60m')[0][:4])

    def test_no_target_already_available_for_daily_birthdays_and_leap_day(self):
        for birth in list(pd.date_range('1998-01-01','1998-12-31')) + [pd.Timestamp('2000-02-29')]:
            annual,b = calendar_grid(self.group(birth))
            _,labels,latest = align_calendar(annual,b,'dash_60m')
            self.assertFalse(any(labels))
            self.assertTrue(all(a < z for a,z in zip(latest,latest[1:])))

    def test_only_isolated_gaps_filled_and_labels_propagate(self):
        g = self.group()
        g.loc[g.age == 10,'dash_60m'] = np.nan
        annual,birth = calendar_grid(g)
        values,labels,_ = align_calendar(annual,birth,'dash_60m')
        self.assertTrue(labels[2])
        self.assertIsNotNone(values[2])
        g.loc[g.age == 11,'dash_60m'] = np.nan
        annual,birth = calendar_grid(g)
        values,labels,_ = align_calendar(annual,birth,'dash_60m')
        self.assertIsNone(values[2])


if __name__ == '__main__':
    unittest.main()
