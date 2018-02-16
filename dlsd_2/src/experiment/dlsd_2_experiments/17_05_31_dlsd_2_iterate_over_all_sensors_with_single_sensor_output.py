import logging

from dlsd_2.experiment_helper.Experiment_Helper_Multiple_Models import Experiment_Helper_Multiple_Models
from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average
from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors import ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors

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
	model.set_max_steps(3)

	input_time_offsets_list = [0]
	target_time_offsets_list = [30,45,60]
	
	train_input_and_target_maker = ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors()
	train_input_and_target_maker.set_source_file_path(file_path_train)
	train_input_and_target_maker.set_moving_average_window(15)
	train_input_and_target_maker.set_efficiency_threshold(1.0) 
	train_input_and_target_maker.set_input_time_offsets_list(input_time_offsets_list)
	train_input_and_target_maker.set_target_time_offsets_list(target_time_offsets_list)
	train_input_and_target_maker.make_source_data()
	train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')

	test_input_and_target_maker = ITM_Normalized_Moving_Average()
	test_input_and_target_maker.set_source_file_path(file_path_test)
	test_input_and_target_maker.copy_parameters_from_maker(train_input_and_target_maker)
	test_input_and_target_maker.make_source_data()

	available_sensors = train_input_and_target_maker.get_source_idxs_list() # bc of type remove inefficient sensors, get available sensors

	root_path = '/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_1'
	current_exp_name = "single_layer_30_45_60"

	for i in range(0,3):
		current_sensor = available_sensors[i]
		logging.info("Starting experiment with sensor "+current_sensor)
		
		dir_hlpr = Experiment_Helper_Multiple_Models()
		dir_hlpr.setup_with_home_directory_path_and_sensor_name(root_path, current_sensor)

		train_input_and_target_maker.set_target_sensor_idxs_list([current_sensor])
		train_input_and_target_maker.make_input_and_target()
		
		test_input_and_target_maker.copy_parameters_from_maker(train_input_and_target_maker)
		test_input_and_target_maker.make_input_and_target()

		model.set_path_tf_output(dir_hlpr.get_tensorflow_dir_path())	
		model.set_path_saved_tf_session(dir_hlpr.new_tf_session_file_path_with_specifier(current_exp_name))

		model.train_with_prepared_input_target_maker(train_input_and_target_maker)
		model.test_with_prepared_input_target_maker(test_input_and_target_maker)
		
		accuracy_dict = model.calc_prediction_accuracy()
		
		model.write_target_and_predictions_to_file(dir_hlpr.new_predictions_file_path_with_specifier(current_exp_name))



if __name__=="__main__":
	main()



