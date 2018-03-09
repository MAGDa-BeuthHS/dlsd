import unittest

import pandas as pd

from dlsd_2.src import Inefficient_Sensors_Remover


class Inefficient_Sensors_Remover_Tests(unittest.TestCase):
    def setUp(self):
        self.dir = 'dlsd_2/tests/dataset/dataset_helpers/inefficient_sensors_remover_test_csvs/'
        isr = Inefficient_Sensors_Remover()
        df = pd.read_csv(self.dir + 'inefficient_sensors.csv', index_col=0)
        self.new_df = isr.remove_inefficient_sensors(df, 0.85)

    def test_two_columns(self):
        self.assertEqual(self.new_df.shape[1], 2)

    def test_col_values(self):
        for i in range(self.new_df.shape[0]):
            self.assertEqual(self.new_df.iloc[i, 0], 0)
            self.assertEqual(self.new_df.iloc[i, 1], 0)
