from .Experiment import *
from .experiment_helper.Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output
import os
class Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output(Experiment):

	def run_experiment(self):
		self._gather_experiment()
		self._iterate_over_all_sensors_test_and_train_using_one_sensor_as_target()

	def _iterate_over_all_sensors_test_and_train_using_one_sensor_as_target(self):
		available_sensors = self.train_input_and_target_maker.get_source_idxs_list() # bc of type remove inefficient sensors, get available sensors
		for i in range(0,3):#for self.current_sensor_used_as_model_output in available_sensors:
			self.current_sensor_used_as_model_output = available_sensors[i]
			logging.info("Starting experiment with sensor "+self.current_sensor_used_as_model_output)
			self._iterate_over_io_params()

	def _create_current_experiment_helper(self):
		self.current_experiment_helper = Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output()
		self.current_experiment_helper.set_experiment_output_path(self.root_path)
		self.current_experiment_helper.set_sensor_name(self.current_sensor_used_as_model_output)
		self.current_experiment_helper.set_io_parameters_name(self.current_io_param.name)
		self.current_experiment_helper.setup_directory()
	
	def _set_input_target_makers_to_current_model_input_output_parameters(self):
		self.train_input_and_target_maker.set_input_time_offsets_list(self.current_io_param.input_time_offsets_list)
		self.train_input_and_target_maker.set_target_time_offsets_list(self.current_io_param.target_time_offsets_list)
		self.train_input_and_target_maker.set_target_sensor_idxs_list([self.current_sensor_used_as_model_output])
	