import unittest

import pandas as pd
from dlsd_2.input_target_maker.ITM_Fill_Time_Gaps_Normalized_Moving_Average import ITM_Fill_Time_Gaps_Normalized_Moving_Average
from dlsd_2.model.types.average_week.Average_Week_Content import Average_Week_Content

from dlsd_2.src.io.dataset import Dataset_From_SQL


class AWC_Tests(unittest.TestCase):
	def setUp(self):
		self.test_csv_dir = '/Users/ahartens/Dropbox/Work/Analysis/dlsd_2/tests/model/types/average_week/average_week_test_csvs/'
		self.average_week_content = Average_Week_Content()
		self.input_target_maker = ITM_Fill_Time_Gaps_Normalized_Moving_Average()
		self.input_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')

class AWC_Test_Calc_Average_Week(AWC_Tests):
	'''
		sample 3 weeks has 3 weeks and 2 sensors, odd values 1,3,5 in column 0 and even values 2,4,6 in 

		week 1 		1 	2
		week 2 		3	4
		week 3		5 	6
	'''
	def setUp(self):
		super(AWC_Test_Calc_Average_Week,self).setUp()
		sample_three_weeks_path = self.test_csv_dir + '3_weeks_odd_column_0_even_column_1.csv'
		self.sample_three_weeks_df = pd.read_csv(sample_three_weeks_path,index_col=0)
		self.avg_week = self.average_week_content._calculate_average_week_from_numpy_array(self.sample_three_weeks_df.values)

	def test_calc_average_week(self):
		''' average of first column should be 3, of second 4 '''
		for i in range(self.avg_week.shape[0]):
			self.assertEqual(self.avg_week[i,0],3)
			self.assertEqual(self.avg_week[i,1],4)

	def test_size(self):
		self.assertTrue(self.avg_week.shape[0],1440*7)



class AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Abstract(AWC_Tests):
	'''
		Input is a preaveraged week, in this case a test week where mondays are all 0s, tues all 1s etc
		Now create target data using this preaveraged week.
	'''
	def setUp(self):
		super(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Abstract,self).setUp()
		self.pre_averaged_week_starting_mon_path = self.test_csv_dir + 'avg_week_3_sensors.csv'

		self.average_week_content.set_average_data_from_csv_file_path(self.pre_averaged_week_starting_mon_path)
		self.replicate_model__set_model_content_parameters()
		self.make_correct_and_incorrect_data_with_file_paths(self.correct_file_name,self.incorrect_file_name)




class AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Abstract):
	def setUp(self):
		self.correct_file_name = 'avg_week_target_inputs(0,1440)_outputs(1440,2880,3420).csv'
		self.incorrect_file_name = 'avg_week_target_inputs(0)_outputs(1440,2880,3420).csv'
		super(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset,self).setUp()

	def make_correct_and_incorrect_data_with_file_paths(self,correct_name, incorrect_name):
		self.correct_data = pd.read_csv(self.test_csv_dir+correct_name,index_col=0)
		self.incorrect_data = pd.read_csv(self.test_csv_dir+incorrect_name,index_col=0)

	def replicate_model__set_model_content_parameters(self):
		self.input_target_maker.set_input_sensor_idxs_and_time_offsets(None,[0,1440])
		self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0],[1440,2880,4320])
		self.average_week_content.target_begin_weekday_int = 0
		self.average_week_content.target_time_offsets_list = [1440,2880,4320]
		self.average_week_content.input_time_offsets_list = [0,1440]
		self.average_week_content.num_target_rows = 7*1440*3 # define target dataset as 3 weeks long here
		self.average_week_content.target_sensor_idxs_list = [0]
		self.average_week_content.set_input_target_maker(self.input_target_maker, test=True)

	def test_correctly_created_predictions(self):
		prediction_dataset_object = self.average_week_content.make_prediction_dataset_object()
		self.assertTrue(prediction_dataset_object.df.values.tolist() == self.correct_data.values.tolist())
		self.assertFalse(prediction_dataset_object.df.values.tolist() == self.incorrect_data.values.tolist())




class AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Multiple_Sensors(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Abstract):
	def setUp(self):
		self.correct_file_name = 'avg_week_target_inputs(0,1440)_outputs(1440,2880,3420).csv'
		self.incorrect_file_name = 'avg_week_target_inputs(0)_outputs(1440,2880,3420).csv'
		super(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Multiple_Sensors,self).setUp()

	def make_correct_and_incorrect_data_with_file_paths(self,correct_name, incorrect_name):
		self.correct_data = pd.read_csv(self.test_csv_dir+correct_name,index_col=0)
		self.incorrect_data = pd.read_csv(self.test_csv_dir+incorrect_name,index_col=0)

	def replicate_model__set_model_content_parameters(self):
		self.average_week_content.target_begin_weekday_int = 0
		self.average_week_content.target_time_offsets_list = [1440,2880,4320]
		self.average_week_content.input_time_offsets_list = [0,1440]
		self.average_week_content.num_target_rows = 7*1440*3 # define target dataset as 3 weeks long here
		self.average_week_content.target_sensor_idxs_list = [0,2]

	def test_correctly_created_predictions(self):
		pass
		#print(self.average_week_content.source_average_dataset_object.df.head())
		#prediction_dataset_object = self.average_week_content.make_prediction_dataset_object()
		#print(prediction_dataset_object.df.head())
		#self.assertTrue(prediction_dataset_object.df.values.tolist() == self.correct_data.values.tolist())
		#self.assertFalse(prediction_dataset_object.df.values.tolist() == self.incorrect_data.values.tolist())




# TODO : not currently used below


class AWC_Test_Week_Start_On_N(AWC_Tests):
	def setUp(self):
		super(AWC_Test_Week_Start_On_N,self).setUp()
		pre_averaged_week_starting_mon_path = self.test_csv_dir + 'avg_week_starting_on_mon.csv'
		pre_averaged_week_starting_thurs_path = self.test_csv_dir + 'avg_week_starting_on_thurs.csv'

		self.pre_averaged_week_starting_mon = pd.read_csv(pre_averaged_week_starting_mon_path,index_col=0)
		self.pre_averaged_week_starting_thurs = pd.read_csv(pre_averaged_week_starting_thurs_path,index_col=0)

	def test_set_average_data_from_csv_file_path(self):
		self.average_week_content.set_average_data_from_csv_file_path()


class AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Different_Day_Begin(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Abstract):
	def setUp(self):
		self.correct_file_name = 'avg_week_target_inputs(0,1440)_outputs(1440,2880,3420).csv'
		self.incorrect_file_name = 'avg_week_target_inputs(0)_outputs(1440,2880,3420).csv'
		super(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Different_Day_Begin,self).setUp()

	def make_correct_and_incorrect_data_with_file_paths(self,correct_name, incorrect_name):
		pass
		
	def replicate_model__set_model_content_parameters(self):
		self.average_week_content.target_begin_weekday_int = 0
		self.average_week_content.target_time_offsets_list = [1440,2880,4320]
		self.average_week_content.input_time_offsets_list = [0]
		self.average_week_content.num_target_rows = 7*1440*3 # define target dataset as 3 weeks long here
		self.average_week_content.target_sensor_idxs_list = [0]

	def test_correctly_created_predictions(self):
		prediction_dataset_object = self.average_week_content.make_prediction_dataset_object()

		prediction_dataset_object.write_csv('/Users/ahartens/Desktop/Work/17_05_test_3_weeks.csv')
		#self.assertTrue(prediction_dataset_object.df.values.tolist() == self.correct_data.values.tolist())
		#self.assertFalse(prediction_dataset_object.df.values.tolist() == self.incorrect_data.values.tolist())

class AWC_Test_Labels(AWC_Tests):
	def setUp(self):
		super(AWC_Test_Labels,self).setUp()
		path = 'dlsd_2/tests/test_csvs/24_10_16_PZS_Belegung_oneMonth.csv'
		self.data_obj = Dataset_From_SQL()
		self.data_obj.read_csv(path)
		self.data_obj.pivot()
		self.data_obj.fill_time_gaps('%Y-%m-%d %H:%M:%S')
		self.average_week_content.calculate_average_data_from_dataset_object(self.data_obj)

	def test_calculated(self):
		pass#print(self.average_week_content.source_average_dataset_object.df)
