from dlsd_2.dataset.dataset_helpers.Time_Gap_Filler import Time_Gap_Filler
import pandas as pd
import numpy as np
import datetime
import logging
import unittest

class Time_Gap_Filler_Tests(unittest.TestCase):
	def setUp(self):
		self.dir = 'dlsd_2/tests/dataset/dataset_helpers/time_gap_filler_test_csvs/'
		self.time_gap_filler = Time_Gap_Filler()
		self.time_gap_filler.set_time_format_with_string('%Y-%m-%d %H:%M:%S')
		self.read_no_gap()
		self.fill_single_time_gap()
		self.fill_double_time_gap()

	def read_no_gap(self):
		self.df_no_gap = pd.read_csv(self.dir + 'no_time_gap.csv',index_col=0)

	def fill_single_time_gap(self):
		self.df_single_gap = pd.read_csv(self.dir + 'single_gap_at_idx_10_to_15.csv',index_col=0)
		self.df_filled_single_gap = self.time_gap_filler.fill_time_gaps_in_dataframe(self.df_single_gap)

	def fill_double_time_gap(self):
		self.df_double_gap = pd.read_csv(self.dir + 'double_gap_at_idx_1_to_5_and_15_to_19.csv',index_col=0)
		self.df_filled_double_gap = self.time_gap_filler.fill_time_gaps_in_dataframe(self.df_double_gap)

	def test_if_filled_correct_size(self):
		self.assertEqual(self.df_filled_single_gap.shape[0],self.df_no_gap.shape[0])
		self.assertEqual(self.df_filled_double_gap.shape[0],self.df_no_gap.shape[0])

	def test_if_single_filled_top_fill_junction_correct(self):
		idx_non_gap = 9
		idx_gap = 10
		self.perform_fill_junction_test_on_df_with_idxs(self.df_filled_single_gap, idx_non_gap, idx_gap)

	def test_if_single_filled_bottom_fill_junction_correct(self):
		idx_non_gap = 15
		idx_gap = 14
		self.perform_fill_junction_test_on_df_with_idxs(self.df_filled_single_gap, idx_non_gap, idx_gap)

	def test_if_double_filled_first_top_fill_junction_correct(self):
		idx_non_gap = 0
		idx_gap = 1
		self.perform_fill_junction_test_on_df_with_idxs(self.df_filled_double_gap, idx_non_gap, idx_gap)

	def test_if_double_filled_second_top_fill_junction_correct(self):
		idx_non_gap = 14
		idx_gap = 15
		self.perform_fill_junction_test_on_df_with_idxs(self.df_filled_double_gap, idx_non_gap, idx_gap)

	def test_if_double_filled_first_bottom_fill_junction_correct(self):
		idx_non_gap = 5
		idx_gap = 4
		self.perform_fill_junction_test_on_df_with_idxs(self.df_filled_double_gap, idx_non_gap, idx_gap)

	def test_if_double_filled_second_bottom_fill_junction_correct(self):
		idx_non_gap = 19
		idx_gap = 18
		self.perform_fill_junction_test_on_df_with_idxs(self.df_filled_double_gap, idx_non_gap, idx_gap)
	
	def perform_fill_junction_test_on_df_with_idxs(self, df, idx_non_gap, idx_gap):
		self.assertEqual(df.iloc[idx_non_gap,0],self.df_no_gap.iloc[idx_non_gap,0])
		self.assertEqual(df.iloc[idx_non_gap,1],self.df_no_gap.iloc[idx_non_gap,1])
		self.assertTrue(np.isnan(df.iloc[idx_gap,0]))
		self.assertTrue(np.isnan(df.iloc[idx_gap,1]))

	def test_if_timestamps_correct(self):
		for i in range(0,self.df_no_gap.shape[0]):
			#filled_stamp = datetime.datetime.stfptime(time_stamps[i],self.time_format)
			pass
			#self.assertEqual(self.df_no_gap.index.values[i],self.df_filled_single_gap.index.values[i])
			#self.assertEqual(self.df_no_gap.index.values[i],self.df_filled_double_gap.index.values[i])

	def get_stamp_for_string(self,stamp):
		    time_stamps[i] = datetime.datetime.stfptime(time_stamps[i],self.time_format)

