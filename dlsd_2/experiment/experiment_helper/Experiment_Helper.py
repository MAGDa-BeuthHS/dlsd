import os

class Experiment_Helper:
	'''
		|Experiment location
		|---root_dir
		|------tensorflow_dir
		|------predictions_targets_dir
	'''
	def __init__(self):
		self.path = None
		self.tensorflow_dir_path = None
		self.prediction_targets_dir_path = None

	def setup_with_home_directory_path_and_sensor_name(self, home_directory_path, exp_name):
		self.check_or_make_dir(home_directory_path)
		self.set_root_dir(os.path.join(home_directory_path,exp_name))
		self.create_child_dirs()

	def set_root_dir(self, dir_name):
		self.path = dir_name
		self.add_directory_to_root_dir_with_name(self.path)

	def create_child_dirs(self):
		self.tensorflow_dir_path = self.add_directory_to_root_dir_with_name("tensorflow")
		self.prediction_targets_dir_path = self.add_directory_to_root_dir_with_name("predictions_and_targets")
		print(self.tensorflow_dir_path)

	def add_directory_to_root_dir_with_name(self, child_name):
		path = os.path.join(self.path,child_name)
		return self.check_or_make_dir(path)

	def check_or_make_dir(self, dir_name):
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		return dir_name

	def new_predictions_file_path_with_specifier(self, name):
		return os.path.join(self.prediction_targets_dir_path,name+".csv")

	def new_tf_session_file_path_with_specifier(self, name):
		new_tf_session_path = os.path.join(self.tensorflow_dir_path,"tf_session_"+name)
		return new_tf_session_path

	def get_tensorflow_dir_path(self):
		return self.tensorflow_dir_path