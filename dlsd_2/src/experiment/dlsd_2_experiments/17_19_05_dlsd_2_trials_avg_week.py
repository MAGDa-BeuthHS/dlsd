import logging

from dlsd_2.input_target_maker.ITM_Fill_Time_Gaps_Moving_Average import ITM_Fill_Time_Gaps_Moving_Average

from dlsd_2.src.model.types.average_week.Average_Week import Average_Week

logging.basicConfig(level=logging.INFO)

def main():
	file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
	file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'

	# 1 define a model : set parameters
	model = Average_Week()
	
	# 2 define model input/target : this is done using a maker.
	train_input_and_target_maker = ITM_Fill_Time_Gaps_Moving_Average()
	train_input_and_target_maker.set_source_file_path(file_path_train)
	train_input_and_target_maker.set_moving_average_window(15)
	train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')
	train_input_and_target_maker.set_input_sensor_idxs_and_timeoffsets_lists(None,[0]) # use all sensors, no time offset for input
	train_input_and_target_maker.set_target_sensor_idxs_and_timeoffsets_lists([0],[30,45,60]) # use sensor

	# 3 train the model
	model.set_average_data_from_csv_file_path('/Users/ahartens/Desktop/Average_Week_One_Year.csv')
	model.prepare_data_and_train_with_input_target_maker(train_input_and_target_maker)
	
	# 4 test the model
	test_input_and_target_maker = ITM_Fill_Time_Gaps_Moving_Average()
	test_input_and_target_maker.set_source_file_path(file_path_test)
	#test_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')	
	model.prepare_data_and_test_with_input_target_maker(test_input_and_target_maker)

	# 5 write results
	accuracy_dict = model.calc_prediction_accuracy()
	model.write_target_and_predictions_to_file('/Users/ahartens/Desktop/preidctions_and_target.csv')

	print(accuracy_dict)
if __name__=="__main__":
	main()