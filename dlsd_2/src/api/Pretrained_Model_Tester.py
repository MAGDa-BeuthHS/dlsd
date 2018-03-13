from dlsd_2.src.experiment.Experiment import *

"""
 Test Pre-Trained Model

"""

class Pretrained_Model_Tester(Experiment):
	def test_model(self):
		self._gather_experiment()
		self.current_io_param = self.model_input_output_parameters[0]
		self.test_input_and_target_maker.set_all_sensor_idxs_and_time_offsets_using_parameters_object(self.current_io_param)
		self._create_current_experiment_helper()
		self._make_input_and_target()
		return self._test_model_with_test_data()

	def _create_input_and_target_makers(self):
		self.test_input_and_target_maker = Input_And_Target_Maker_2()
		self.test_input_and_target_maker.set_source_dataset_object(self.source_maker.test)
		self.test_input_and_target_maker.time_format = self.source_maker.time_format_test

	def _make_input_and_target(self):
		self.test_input_and_target_maker.make_input_and_target()

	def _test_model_with_test_data(self):
		model = self.models[0]
		model.set_experiment_helper(self.current_experiment_helper)
		model.test_with_prepared_input_target_maker(self.test_input_and_target_maker)
		normalized_predictions =  model.model_output.get_prediction_df()
		return self.source_maker.test.denormalizer.denormalize(normalized_predictions)

	