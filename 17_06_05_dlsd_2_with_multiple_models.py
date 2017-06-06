from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors import ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors
from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average
from dlsd_2.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer
from dlsd_2.model.types.average_week.Average_Week import Average_Week
from dlsd_2.experiment_helper.Experiment_Helper_Multiple_Models import Experiment_Helper_Multiple_Models
import logging

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)

def main():
	file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
	file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'

	# define a model
	model_1 = NN_One_Hidden_Layer()
	model_1.set_number_hidden_nodes(50)
	model_1.set_learning_rate(.1)
	model_1.set_batch_size(3)
	model_1.set_max_steps(3)

	model_2 = NN_One_Hidden_Layer()
	model_2.set_number_hidden_nodes(50)
	model_2.set_learning_rate(.1)
	model_2.set_batch_size(3)
	model_2.set_max_steps(3)

	models = [model_1, model_2]

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

	for i in range(0,3):
		current_sensor_used_as_model_output = available_sensors[i]
		logging.info("Starting experiment with sensor "+current_sensor_used_as_model_output)
		
		dir_hlpr = Experiment_Helper_Multiple_Models()
		dir_hlpr.setup_with_home_directory_path_and_sensor_name(root_path, current_sensor_used_as_model_output)

		# set current
		train_input_and_target_maker.set_target_sensor_idxs_list([current_sensor_used_as_model_output])
		train_input_and_target_maker.make_input_and_target()
		test_input_and_target_maker.copy_parameters_from_maker(train_input_and_target_maker)
		test_input_and_target_maker.make_input_and_target()

		# write target data 
		test_df = test_input_and_target_maker.get_target_df()
		test_df.to_csv(dir_hlpr.get_target_file_path())

		# train and test
		model_prediction_accuracies = []
		for model in models:
			model.set_experiment_helper(dir_hlpr)
			model.train_with_prepared_input_target_maker(train_input_and_target_maker)
			model.test_with_prepared_input_target_maker(test_input_and_target_maker)
			model.write_predictions_using_experiment_helper()
			model_prediction_accuracies.append(model.calc_prediction_accuracy())
		
if __name__=="__main__":
	main()



