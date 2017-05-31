from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors import ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors
from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average
from dlsd_2.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer
from dlsd_2.model.types.average_week.Average_Week import Average_Week
import logging

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)

def main():
	file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
	file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'

	# define a model
	model = NN_One_Hidden_Layer()
	model.set_number_hidden_nodes(50)
	model.set_learning_rate(.1)
	model.set_batch_size(3)
	model.set_max_steps(10000)
	model.set_path_saved_session('/Users/ahartens/Desktop/Work/dlsd_2_trials/saved_sess')
	model.set_path_tf_output('/Users/ahartens/Desktop/Work/dlsd_2_trials')	

	input_time_offsets_list = [0]
	target_time_offsets_list = [30,45,60]
	
	train_input_and_target_maker = ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors()
	train_input_and_target_maker.set_source_file_path(file_path_train)
	train_input_and_target_maker.set_moving_average_window(15)
	train_input_and_target_maker.set_efficiency_threshold(1.0)
	train_input_and_target_maker.set_input_time_offsets_list(input_time_offsets_list)
	train_input_and_target_maker.set_target_time_offsets_list(target_time_offsets_list)
	train_input_and_target_maker.make_source_data()

	test_input_and_target_maker = ITM_Normalized_Moving_Average()
	test_input_and_target_maker.set_source_file_path(file_path_test)
	test_input_and_target_maker.make_source_data()

	available_sensors = train_input_and_target_maker.get_source_idxs_list() # bc of type remove inefficient sensors, get available sensors

	for current_sensor in available_sensors:
		train_input_and_target_maker.set_target_sensor_idxs_list([current_sensor])
		train_input_and_target_maker.make_input_and_target()
		
		test_input_and_target_maker.copy_parameters_from_maker(train_input_and_target_maker)
		test_input_and_target_maker.make_input_and_target()
		
		model.train_with_prepared_input_target_maker(train_input_and_target_maker)
		model.test_with_prepared_input_target_maker(test_input_and_target_maker)
		accuracy_dict = model.calc_prediction_accuracy()

if __name__=="__main__":
	main()



