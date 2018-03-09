import logging

from dlsd_2.src.io.input_target_maker.ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average
from dlsd_2.src.io.input_target_maker.ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors import ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors

from dlsd_2.src.model.types.neural_networks.nn_one_hidden_layer import NN_One_Hidden_Layer

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)

def main():
	file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
	file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'

	# define a model
	model = NN_One_Hidden_Layer()
	model.set_number_hidden_nodes(50)
	model.set_learning_rate(.1)
	model.set_batch_size(3)
	model.set_max_steps(1000)
	model.set_path_saved_tf_session('/Users/ahartens/Desktop/Work/dlsd_2_trials/saved_sess')
	model.set_path_tf_output('/Users/ahartens/Desktop/Work/dlsd_2_trials')	
	
	# Make train data
	train_input_and_target_maker = ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors()
	train_input_and_target_maker.set_source_file_path(file_path_train)
	train_input_and_target_maker.set_moving_average_window(15)
	train_input_and_target_maker.set_efficiency_threshold(1.0)
	train_input_and_target_maker.set_input_sensor_idxs_and_time_offsets(None,[0]) # use all sensors, no time offset for input
	train_input_and_target_maker.set_target_sensor_idxs_and_time_offsets([0],[30,45,60]) # use sensor
	train_input_and_target_maker.prepare_source_data_and_make_input_and_target()
	train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')

	# Make test data
	test_input_and_target_maker = ITM_Normalized_Moving_Average()
	test_input_and_target_maker.set_source_file_path(file_path_test)
	test_input_and_target_maker.copy_parameters_from_maker(train_input_and_target_maker)
	test_input_and_target_maker.prepare_source_data_and_make_input_and_target()

	model.train_with_prepared_input_target_maker(train_input_and_target_maker)
	model.test_with_prepared_input_target_maker(test_input_and_target_maker)

	accuracy_dict = model.calc_prediction_accuracy()
	model.write_target_and_predictions_to_file('/Users/ahartens/Desktop/preidctions_and_target.csv')

if __name__=="__main__":
	main()



