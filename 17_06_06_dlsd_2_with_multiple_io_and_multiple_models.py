from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors import ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors
from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average
from dlsd_2.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer
from dlsd_2.model.types.average_week.Average_Week import Average_Week
from dlsd_2.experiment_helper.Experiment_Helper_Multiple_Models import Experiment_Helper_Multiple_Models
import logging

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)
class Experiment_Parameters:
	def set_input_time_offsets_list(self, the_list):
		self.input_time_offsets_list = the_list
	def set_target_time_offsets_list(self, the_list):
		self.target_time_offsets_list = the_list
	def set_input_sensor_idxs_list(self, the_list):
		self.input_sensor_idxs_list = the_list
	def set_target_sensor_idxs_list(self, the_list):
		self.target_sensor_idxs_list = the_list
	def set_use_adjacency_matrix(self, the_boolean):
		self.set_use_adjacency_matrix = the_boolean


def main():
	exp = Experiment()
	exp.set_experiment_root_path('/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_1')
	exp.run_experiment()

class Experiment:

	def set_experiment_root_path(self, path):
		self.root_path = path

	def run_experiment(self):
		self._define_models()
		self._define_input_and_target_makers()
		self._iterate_over_all_sensors_test_and_train_using_one_sensor_as_target()

	def _define_models(self):
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
		
		self.models = [model_1]

	def _define_input_and_target_makers(self):
		file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
		file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'
		
		self.train_input_and_target_maker = ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors()
		self.train_input_and_target_maker.set_source_file_path(file_path_train)
		self.train_input_and_target_maker.set_moving_average_window(15)
		self.train_input_and_target_maker.set_efficiency_threshold(1.0) 
		self.train_input_and_target_maker.make_source_data()
		self.train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')

		self.test_input_and_target_maker = ITM_Normalized_Moving_Average()
		self.test_input_and_target_maker.set_source_file_path(file_path_test)
		self.test_input_and_target_maker.copy_parameters_from_maker(self.train_input_and_target_maker)
		self.test_input_and_target_maker.make_source_data()


	def _iterate_over_all_sensors_test_and_train_using_one_sensor_as_target(self):
		available_sensors = self.train_input_and_target_maker.get_source_idxs_list() # bc of type remove inefficient sensors, get available sensors
		for i in range(0,3):#for self.current_sensor_used_as_model_output in available_sensors:
			self.current_sensor_used_as_model_output = available_sensors[i]
			logging.info("Starting experiment with sensor "+self.current_sensor_used_as_model_output)

			self.current_experiment_helper = Experiment_Helper_Multiple_Models()
			self.current_experiment_helper.setup_with_home_directory_path_and_sensor_name(self.root_path, self.current_sensor_used_as_model_output)

			self._set_input_target_makers_to_current_experiment_parameters()
			self._make_input_and_target()
			self._write_target_data_to_file()
			self._train_and_test_all_models()
			accuracies = self._collect_all_model_accuracies()

	def _set_input_target_makers_to_current_experiment_parameters(self):

		input_time_offsets_list = [0]
		target_time_offsets_list = [30,45,60]
	
		self.train_input_and_target_maker.set_input_time_offsets_list(input_time_offsets_list)
		self.train_input_and_target_maker.set_target_time_offsets_list(target_time_offsets_list)
		self.train_input_and_target_maker.set_target_sensor_idxs_list([self.current_sensor_used_as_model_output])
	
	def _make_input_and_target(self):
		self.train_input_and_target_maker.make_input_and_target()
		self.test_input_and_target_maker.copy_parameters_from_maker(self.train_input_and_target_maker)
		self.test_input_and_target_maker.make_input_and_target()

	def _write_target_data_to_file(self):
		test_df = self.test_input_and_target_maker.get_target_df()
		test_df.to_csv(self.current_experiment_helper.get_target_file_path())

	def _train_and_test_all_models(self):
		for model in self.models:
			model.set_experiment_helper(self.current_experiment_helper)
			model.train_with_prepared_input_target_maker(self.train_input_and_target_maker)
			model.test_with_prepared_input_target_maker(self.test_input_and_target_maker)
			model.write_predictions_using_experiment_helper()

	def _collect_all_model_accuracies(self):
		model_prediction_accuracies = []
		for model in self.models:
			model_prediction_accuracies.append(model.calc_prediction_accuracy())
		return model_prediction_accuracies

if __name__=="__main__":
	main()



