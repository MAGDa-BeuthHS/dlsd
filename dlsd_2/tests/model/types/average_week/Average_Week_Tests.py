import unittest

import pandas as pd
from dlsd_2.src.io.input_target_maker.ITM_Fill_Time_Gaps import ITM_Fill_Time_Gaps
from dlsd_2.src.model.types.average_week.Average_Week import Average_Week


class Average_Week_Tests(unittest.TestCase):
	def setUp(self):
		self.test_csv_dir = '/Users/ahartens/Dropbox/Work/Analysis/dlsd_2/tests/model/types/average_week/average_week_test_csvs/'
		self.file_path_thursday = self.test_csv_dir + 'avg_week_test_3_weeks_ascending_days_starting_thurs_real_dates.csv'
		self.file_path_monday = self.test_csv_dir + 'avg_week_test_3_weeks_ascending_days_starting_mon_real_dates.csv'
		self.file_path_monday_with_gaps = self.test_csv_dir + 'avg_week_test_3_weeks_ascending_days_starting_mon_real_dates_with_gaps.csv'
		self.file_path_thursday_with_gaps = self.test_csv_dir + 'avg_week_test_3_weeks_ascending_days_starting_thurs_real_dates_with_gaps.csv'


	def make_average_week_model_and_input_target_maker(self):
		self.model = Average_Week()
		self.input_target_maker = ITM_Fill_Time_Gaps()
		self.input_target_maker.set_source_file_path(self.file_path_train)
		self.input_target_maker.source_is_sql_output = False
		self.input_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')
		self.input_target_maker.set_input_sensor_idxs_and_time_offsets(None,[0]) # use all sensors, no time offset for input
		self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0],[1440,2880,4320]) 
		self.model.prepare_data_and_train_with_input_target_maker(self.input_target_maker)

	def test_source_average_week_generation(self):
		'''
			The source_average_dataset_object week should be 1 2 3 4 5 6 7 for all types 
			(all standardize to start on monday, regardless of date which source data begins on) 
		'''

		correct_value = 0
		for i in range(self.model.model_content.source_average_dataset_object.df.shape[0]):
			if (i%1440) == 0:
				correct_value = correct_value + 1
			self.assertEqual(self.model.model_content.source_average_dataset_object.df.iloc[i,0],correct_value)

class AWT_1_Source_Data_Prediction_Data_Generated_Correctly_Source_Week_Starting_On_Monday(Average_Week_Tests):
	'''
		Test CSV is 3 weeks of one sensor with no gaps . all mondays of first week are 0s, and as follows in table :
			m t w t f s s     t  f  s  s  m  t  w 
		w1 	0 1 2 3 4 5 6     2  3  4  5  6  7  8    
		w2  1 2 3 4 5 6 7     9  10 11 12 13 14 15
		w3  2 3 4 5 6 7 8     16 17 18 19 20 21 22
		1. The average week should therefore be identical to week 2
		2. The correct predictions is the average week, time offset to target time offset [1440,2880,4320] repeating over
	'''
	def setUp(self):
		super(AWT_1_Source_Data_Prediction_Data_Generated_Correctly_Source_Week_Starting_On_Monday,self).setUp()
		self.file_path_train = self.file_path_monday
		self.make_average_week_model_and_input_target_maker()

	def test_prediction_generation(self):
		''' 2. The correct predictions is the average week, time offset to target time offset [1440,2880,4320] repeating over'''

		correct_predictions = pd.read_csv(self.test_csv_dir+'weeks_3_avg_correct_predictions_if_target_begins_mon.csv',index_col=0)
		TEST_input_target_maker = ITM_Fill_Time_Gaps()
		TEST_input_target_maker.source_is_sql_output = False
		TEST_input_target_maker.set_source_file_path(self.file_path_train)
		self.model.prepare_data_and_test_with_input_target_maker(TEST_input_target_maker)
		for i in range(self.model.model_output.prediction_dataset_object.df.shape[0]):
			self.assertEqual(self.model.model_output.prediction_dataset_object.df.iloc[i,0],correct_predictions.iloc[i,0])

class AWT_2_Source_Data_Starts_Thursday_Average_Week_Starts_Monday_Then_Prediction_Matches_Target(Average_Week_Tests):
	'''
		Test CSV is 3 weeks of one sensor with no gaps, starting on thursday. and as follows in table :
			t f s s m t w     t  f  s  s  m  t  w 
		w1 	3 4 5 6 0 1 2     2  3  4  5  6  7  8
		w2  4 5 6 7 1 2 3     9  10 11 12 13 14 15
		w3  5 6 7 8 2 3 4     16 17 18 19 20 21 22
		1. test that the source_average_dataset_object should start on monday and should be 1 2 3 4 5 6 7
		2. test that prediciton_dataset_object 
	'''
	def setUp(self):
		super(AWT_2_Source_Data_Starts_Thursday_Average_Week_Starts_Monday_Then_Prediction_Matches_Target,self).setUp()
		self.file_path_train = self.file_path_thursday
		self.make_average_week_model_and_input_target_maker()

	def test_prediction_generation_target_starts_monday(self):
		correct = pd.read_csv(self.test_csv_dir+'weeks_3_avg_correct_predictions_if_target_begins_mon.csv',index_col=0)
		TEST_input_target_maker = ITM_Fill_Time_Gaps()
		TEST_input_target_maker.source_is_sql_output = False
		TEST_input_target_maker.set_source_file_path(self.file_path_monday)
		self.model.prepare_data_and_test_with_input_target_maker(TEST_input_target_maker)
		self.model.model_output.prediction_dataset_object.write_csv("/Users/ahartens/Desktop/output2.csv")
		for i in range(self.model.model_output.prediction_dataset_object.df.shape[0]):
			self.assertEqual(self.model.model_output.prediction_dataset_object.df.iloc[i,0],correct.iloc[i,0])

	def test_prediction_generation_target_starts_wednesday(self):
		correct = pd.read_csv(self.test_csv_dir+'weeks_3_avg_correct_predictions_if_target_begins_thurs.csv',index_col=0)
		TEST_input_target_maker = ITM_Fill_Time_Gaps()
		TEST_input_target_maker.source_is_sql_output = False
		TEST_input_target_maker.set_source_file_path(self.file_path_thursday)
		self.model.prepare_data_and_test_with_input_target_maker(TEST_input_target_maker)
		for i in range(self.model.model_output.prediction_dataset_object.df.shape[0]):
			self.assertEqual(self.model.model_output.prediction_dataset_object.df.iloc[i,0],correct.iloc[i,0])

class AWT_3_Same_As_1_But_With_Time_Gaps(Average_Week_Tests):
	'''
		Test CSV is 3 weeks of one sensor with Gaps! Deleted entire day 14, and 21 (two thursdays)
			m t w t f s s     t  f  s  s  m  t  w 
		w1 	0 1 2 3 4 5 6     2  3  4  5  6  7  8    
		w2  1 2 3 4 5 6 7     9  10 11 12 13 14 15
		w3  2 3 4 5 6 7 8     16 17 18 19 20 21 22
		1. The average week should therefore be 1 2 3 4 5 5 7
		2. The correct predictions is the average week, time offset to target time offset [1440,2880,4320] repeating over
	'''
	def setUp(self):
		super(AWT_3_Same_As_1_But_With_Time_Gaps,self).setUp()
		self.file_path_train = self.file_path_monday_with_gaps
		self.make_average_week_model_and_input_target_maker()

	def test_source_average_week_generation(self):
		''' 1. The source_average_dataset_object week should be 1 2 3 4 5 5 7 for all types '''
		correct_values = pd.read_csv(self.test_csv_dir+'weeks_3_avg_correct_source_data_with_gaps.csv',index_col=0)
		for i in range(self.model.model_content.source_average_dataset_object.df.shape[0]):
			self.assertEqual(self.model.model_content.source_average_dataset_object.df.iloc[i,0],correct_values.iloc[i,0])

	def test_prediction_generation(self):
		''' 2. The correct predictions is the average week, time offset to target time offset [1440,2880,4320] repeating over'''

		correct_predictions = pd.read_csv(self.test_csv_dir+'weeks_3_avg_correct_predictions_if_target_begins_thurs_with_gaps.csv',index_col=0)
		TEST_input_target_maker = ITM_Fill_Time_Gaps()
		TEST_input_target_maker.source_is_sql_output = False
		TEST_input_target_maker.set_source_file_path(self.file_path_thursday)
		self.model.prepare_data_and_test_with_input_target_maker(TEST_input_target_maker)
		for i in range(self.model.model_output.prediction_dataset_object.df.shape[0]):
			self.assertEqual(self.model.model_output.prediction_dataset_object.df.iloc[i,0],correct_predictions.iloc[i,0])