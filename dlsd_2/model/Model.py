from .Model_Input import Model_Input
from .Model_Output import Model_Output
from .Model_Content import Model_Content

import logging

class Model:
	'''
	Two phases of model usage. Both methods require an input_target maker
		1. During train, the model_content receives the training data and builds an internal representation of the model
		2. During test, the model_content is responsible for creating a target_dataset_object
	The model_input is a wrapper for the input_dataset_object and target_dataset_object, responsible for handling input to model
	The model_output is a wrapper for a prediction_dataset_object and target_dataset_object, responsible for calculating prediction error
	'''
	def __init__(self):
		self.model_input = Model_Input()
		self.model_output = Model_Output()
		self.model_content = Model_Content()
		self.train_input_target_maker = None
		self.current_input_target_maker = None

	def train_with_input_target_maker(self,input_target_maker):
		self.train_input_target_maker = input_target_maker
		self.current_input_target_maker = input_target_maker
		self._create_and_set_input_and_target_for_model()
		self._train()

	def test_with_input_target_maker(self,input_target_maker):
		self.current_input_target_maker = input_target_maker
		self.current_input_target_maker.copy_parameters_from_maker(self.train_input_target_maker)
		self._create_and_set_input_and_target_for_model()
		self._test()

	def _create_and_set_input_and_target_for_model(self):
		self._create_input_and_target()
		self._set_model_input_and_target()

	def _create_input_and_target(self):
		self.current_input_target_maker.make_source_data()
		self.current_input_target_maker.make_input_and_target()

	def _set_model_input_and_target(self):
		self.model_input.set_input_dataset_object(self.current_input_target_maker.get_input_dataset_object())
		self.model_input.set_target_dataset_object(self.current_input_target_maker.get_target_dataset_object())

	def _train(self):
		raise NotImplementedError
	
	def _test(self):
		raise NotImplementedError

	def calc_prediction_accuracy(self):
		mape = self.model_output.calc_mape()
		mae = self.model_output.calc_mae()
		return {'mape':mape,'mae':mae}
		
	def write_target_and_predictions_to_file(self,file_path):
		pass # TODO



