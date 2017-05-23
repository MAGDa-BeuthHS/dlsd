import logging
from dlsd_2.dataset.Dataset import *

from dlsd_2.calc import error

class Model_Output:

	def __init__(self):
		self.prediction_dataset_object = Dataset()
		self.target_dataset_object = Dataset()

	def set_prediction_dataset_object_with_numpy_array(self,numpy_array):
		self.prediction_dataset_object.set_numpy_array(numpy_array)

	def set_target_dataset_object(self,dataset_object):
		self.target_dataset_object = dataset_object

	def write_target_and_predictions_to_file(self, output_file_path):
		pass # TODO

	def calc_mae(self):
		return error.mae(self.prediction_dataset_object.get_numpy_array(), self.target_dataset_object.get_numpy_array())

	def calc_mape(self):
		return error.mape(self.prediction_dataset_object.get_numpy_array(), self.target_dataset_object.get_numpy_array())
