from .Experiment_Helper_With_K_Fold_Validation import *

class Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output(Experiment_Helper_With_K_Fold_Validation):
	'''
		|experiment_directory_path
		|---sensor_1
		    |---input_output_parameters_1
		        |---tensorflow_dir
			    |---targets.csv
		        |---predictions
		            |---model_1.csv
		            |---model_2.csv
	'''

	def __init__(self):
		super(Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output,self).__init__()
		self.sensor_name = None
	
	def set_sensor_name(self, name):
		self.sensor_name = name
		self.root_path = os.path.join(self.experiment_output_path,self.sensor_name)
		self.check_or_make_dir(self.root_path)

	def setup_directory(self):
		super(Experiment_Helper_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output,self).setup_directory()

