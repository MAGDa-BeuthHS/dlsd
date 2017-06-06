from .Experiment_Helper import *

class Experiment_Helper_Multiple_Models(Experiment_Helper):
	'''
		|Experiment location
		|---root_dir
		|   |---tensorflow_dir
			|---targets.csv
		    |---predictions
		        |---model_1.csv
		        |---model_2.csv
	'''
	def __init__(self):
		super(Experiment_Helper_Multiple_Models,self).__init__()
		self.predictions_dir_path = None

	def create_child_dirs(self):
		self.tensorflow_dir_path = self.add_directory_to_root_dir_with_name("tensorflow")
		self.prediction_dir_path = self.add_directory_to_root_dir_with_name("predictions")
	
	def get_target_file_path(self):
		return os.path.join(self.path,"target.csv")

	def make_new_model_prediction_file_path_with_model_name(self,model_name):
		filename = model_name+".csv"
		filepath = os.path.join(self.prediction_dir_path,filename)
		return filepath