from dlsd_2.src.io.input_target_maker.Source_Maker import *
import pickle

class Source_Maker_For_API(Source_Maker):
	def __init__(self):
		super(Source_Maker_For_API, self).__init__()
		self.path_serialized_denormalizer_used_in_training = None
		self.path_desired_columns = None

	def prepare_source_data(self):
		self.read_data_from_csv() # replace this with however you get the data
		self.apply_modifications_defined_in_parameters()

	def read_data_from_csv(self):
		self.test = self.read_source_data(self.file_path_train)

	def apply_modifications_defined_in_parameters(self):
		self._subset_sensors_of_test_set_to_match_train()
		if self.moving_average_window is not None:
			self.test.rolling_average_with_window(self.moving_average_window)
		if self.normalize:
			self.test.denormalizer = self._load_pickle_object(self.path_serialized_denormalizer_used_in_training)
			self.test.normalize()
	
	def _load_pickle_object(self, path):
		with open(path,'rb') as f:
			return pickle.load(f)

	def _subset_sensors_of_test_set_to_match_train(self):
		desired_sensors = list(self._load_pickle_object(self.path_desired_columns))
		self.test.df = self.test.df[desired_sensors]
