from dlsd_2.input_target_maker.ITM_Fill_Time_Gaps_Moving_Average import ITM_Fill_Time_Gaps_Moving_Average
from dlsd_2.model.types.average_week.Average_Week import Average_Week

import logging

logging.basicConfig(level=logging.INFO)


'''
	1. Create a model object of your choice. set parameters of the model
	2. Create a target generator object of your choice.
	3. Create a Model Handler. Pass it the model and the target generator (step 1 and 2)
		- call set_input_file_path before calling 'train' and 'test' on model handler
		- OR call helper methods 'train_with_input_file_path' and 'test_with_input_file_path'
	4. Get accuracy of predictions by calling calc_prediction_accuracy() on model_handler
'''

def main():
	file_path_train = '/Users/ahartens/Desktop/Work/pzs_oneYear_belegung.csv'
	file_path_test = '/Users/ahartens/Desktop/Work/24_10_16_PZS_Belegung_oneMonth.csv'

	# 1 Define a model : set parameters
	model = Average_Week()
	
	# 2 Define model input/target : this is done using a maker. Define parameters here (moving average, time offsets)
	train_input_and_target_maker = ITM_Fill_Time_Gaps_Moving_Average()
	train_input_and_target_maker.set_source_file_path(file_path_train)
	train_input_and_target_maker.set_moving_average_window(15)
	train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S.%f')
	train_input_and_target_maker.set_input_sensor_idxs_and_timeoffsets_lists(None,[0]) # use all sensors, no time offset for input
	train_input_and_target_maker.set_target_sensor_idxs_and_timeoffsets_lists([0],[30,45,60]) # use sensor

	# 3 Train the model
	model.set_average_data_from_csv_file_path('/Users/ahartens/Desktop/Average_Week_One_Year.csv')
	model.train_with_input_target_maker(train_input_and_target_maker)
	
	# 4 Test the model
	test_input_and_target_maker = ITM_Fill_Time_Gaps_Moving_Average()
	test_input_and_target_maker.set_source_file_path(file_path_test)
	test_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')	
	model.test_with_input_target_maker(test_input_and_target_maker)

	# 5
	accuracy_dict = model.calc_prediction_accuracy()
	model.write_target_and_predictions_to_file('/Users/ahartens/Desktop/preidctions_and_target.csv')


if __name__=="__main__":
	main()