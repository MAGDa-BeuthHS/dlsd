from .Experiment_With_K_Fold_Validation import *
from .experiment_helper.Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output
from dlsd_2.experiment.experiment_output_reader.Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation import *

import os
class Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation(Experiment_With_K_Fold_Validation):
	def __init__(self):
		super(Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation,self).__init__()

	def run_experiment(self):
		self._gather_experiment()
		self._iterate_over_all_sensors_test_and_train_using_one_sensor_as_target()
		self._calculate_accuracy_of_models()

	def _iterate_over_all_sensors_test_and_train_using_one_sensor_as_target(self):
		available_sensors = self.source_maker.get_all_sensors() # bc of type remove inefficient sensors, get available sensors
		for i in range(2):#self.current_sensor_used_as_model_output in available_sensors:
			self.current_sensor_used_as_model_output = available_sensors[i]
			self.target_for_current_sensor_written_to_file = False
			logging.info("Starting experiment with sensor "+self.current_sensor_used_as_model_output)
			self._iterate_over_k_fold_validation()

	def _create_current_experiment_helper(self):
		self.current_experiment_helper = Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output()
		self.current_experiment_helper.set_experiment_output_path(self.root_path)
		self.current_experiment_helper.set_sensor_name(self.current_sensor_used_as_model_output)
		self.current_experiment_helper.set_level_0_name('k_'+str(self.current_k))
		self.current_experiment_helper.set_io_parameters_name(self.current_io_param.name)
		self.current_experiment_helper.setup_directory()
	
	def _set_input_target_makers_to_current_model_input_output_parameters(self):
		self._modify_current_input_output_parameters_to_reflect_current_sensor()
		super(Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation,self)._set_input_target_makers_to_current_model_input_output_parameters()

	def _modify_current_input_output_parameters_to_reflect_current_sensor(self):
		print("MODIFYING TO ONLY HAVE SINGLE OUTPUT SENSOR")
		self.current_io_param.set_target_sensor_idxs_list([self.current_sensor_used_as_model_output])
		if self.current_io_param.use_single_sensor_as_input: # if single input/single output
			self.current_io_param.set_input_sensor_idxs_list([self.current_sensor_used_as_model_output])

	def define_error_calculator(self):
		return Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation()

