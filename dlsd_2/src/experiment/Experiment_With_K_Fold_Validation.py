# Alex Hartenstein 2017

import copy

from dlsd_2.experiment.Experiment import *
from dlsd_2.experiment.experiment_output_reader.Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import *
from dlsd_2.input_target_maker.Source_Maker_With_K_Fold_Validation import *

from .experiment_helper.Experiment_Helper_With_K_Fold_Validation import *


class Experiment_With_K_Fold_Validation(Experiment):
	"""
		Iterates k times splitting train and test data, 
		testing and training model for each split
	"""
	def __init__(self):
		super(Experiment_With_K_Fold_Validation, self).__init__()
		self.test_single_k = False
		self.validation_percentage = None
		self.k = None
		self.do_single_k = False

	def run_experiment(self):
		self._gather_experiment()
		self._iterate_over_k_fold_validation()

	def _prepare_source_data_and_input_target_makers(self):
		self.source_maker.prepare_source_data()
		self.source_maker.validation_percentage = self.validation_percentage
		self.source_maker.remove_validation_data()

	def _iterate_over_k_fold_validation(self):
		self._create_validation_input_target_maker() # only needs to be done once, not for each, but 
		for k in range(self.k):
			self.current_k = k
			self.validation_input_target_maker = copy.deepcopy(self.original_validation_input_target_maker) # normalizer from training data
			self._create_combined_traintest_input_target_maker()
			self._iterate_over_io_params()
			if self.do_single_k : break

	def _create_validation_input_target_maker(self):
		self.original_validation_input_target_maker = Input_And_Target_Maker_2()
		self.original_validation_input_target_maker.set_source_dataset_object(self.source_maker.validation)
		self.original_validation_input_target_maker.time_format = self.source_maker.time_format_train
	
	def _create_combined_traintest_input_target_maker(self):
		self.train_test_input_target_maker = Input_And_Target_Maker_2()
		self.train_test_input_target_maker.set_source_dataset_object(self.source_maker.train_test)
		self.train_test_input_target_maker.time_format = self.source_maker.time_format_train

	def _create_current_experiment_helper(self):
		self.current_experiment_helper = Experiment_Helper_With_K_Fold_Validation()
		self.current_experiment_helper.set_experiment_output_path(self.root_path)
		self.current_experiment_helper.set_level_0_name('k_'+str(self.current_k))
		self.current_experiment_helper.set_io_parameters_name(self.current_io_param.name)
		self.current_experiment_helper.setup_directory()

	def _create_input_and_target_to_current_io_params(self):
		self._set_up_combined_train_test_validaton_makers()
		self._prepare_combined_traintest_and_validation_inputs_and_targets()
		self._split_combined_traintest_input_target_maker_into_train_and_test()
		self._normalize_data()
		self._write_target_data_to_file()

	def _set_up_combined_train_test_validaton_makers(self):
		self.train_test_input_target_maker.set_all_sensor_idxs_and_time_offsets_using_parameters_object(self.current_io_param)
		self.validation_input_target_maker.set_all_sensor_idxs_and_time_offsets_using_parameters_object(self.current_io_param)
	
	def _prepare_combined_traintest_and_validation_inputs_and_targets(self):
		self.train_test_input_target_maker.make_input_and_target()
		self.validation_input_target_maker.make_input_and_target()

	def _split_combined_traintest_input_target_maker_into_train_and_test(self):
		splitter = Train_Test_Splitter()
		splitter.k = self.k
		splitter.current_k = self.current_k
		splitter.combined_train_test_itm = self.train_test_input_target_maker
		splitter.current_io_param = self.current_io_param
		self.train_input_and_target_maker, self.test_input_and_target_maker = splitter.split_train_and_test()
		self._set_input_target_makers_to_current_model_input_output_parameters()

	def _normalize_data(self):
		self._normalize_training_data()
		self._set_denormalizer_used_in_training_and_denormalize_itm(self.test_input_and_target_maker)
		self._set_denormalizer_used_in_training_and_denormalize_itm(self.validation_input_target_maker)

	def _normalize_training_data(self):
		self.train_input_and_target_maker.input_maker.dataset_object.normalize()
		self.train_input_and_target_maker.denormalizer_used_in_training = self.train_input_and_target_maker.input_maker.dataset_object.denormalizer
		self.train_input_and_target_maker.target_maker.dataset_object.set_denormalizer(self.train_input_and_target_maker.denormalizer_used_in_training)
		self.train_input_and_target_maker.target_maker.dataset_object.normalize()

	def _set_denormalizer_used_in_training_and_denormalize_itm(self, itm):
		itm.denormalizer_used_in_training = self.train_input_and_target_maker.denormalizer_used_in_training
		itm.input_maker.dataset_object.set_denormalizer(self.train_input_and_target_maker.denormalizer_used_in_training)
		itm.input_maker.dataset_object.normalize()
		itm.target_maker.dataset_object.set_denormalizer(self.train_input_and_target_maker.denormalizer_used_in_training)
		itm.target_maker.dataset_object.normalize()

	def _write_target_data_to_file(self):
		self._write_itm_target_data_to_file(self.test_input_and_target_maker, self.current_experiment_helper.get_target_file_path())
		self._write_itm_target_data_to_file(self.train_input_and_target_maker, self.current_experiment_helper.get_train_target_file_path())
		self._write_itm_target_data_to_file(self.validation_input_target_maker, self.current_experiment_helper.get_validation_target_file_path())


	def _train_and_test_single_model(self, model):
		model.set_experiment_helper(self.current_experiment_helper)
		model.train_with_prepared_input_target_maker(self.train_input_and_target_maker)
		self._test_model_with_input_target_maker(model, self.test_input_and_target_maker, self.current_experiment_helper.make_new_model_prediction_file_path_with_model_name)
		self._test_model_with_input_target_maker(model, self.train_input_and_target_maker, self.current_experiment_helper.make_new_model_train_prediction_file_path_with_model_name)
		self._test_model_with_input_target_maker(model, self.validation_input_target_maker, self.current_experiment_helper.make_new_model_validation_prediction_file_path_with_model_name)

	def _test_model_with_input_target_maker(self, model, itm, path_generator_func):
		model.test_with_prepared_input_target_maker(itm)
		output_path = path_generator_func(model.name)
		model.write_predictions_to_path(output_path)

	def calc_prediction_error(self):
		calc = super(Experiment_With_K_Fold_Validation,self).calc_prediction_error()
		#calc.write_concatenated_predictions_and_target_organized_by_model()


class Train_Test_Splitter:
	"""
		Can't split train/test at source_maker level because input target maker time shifts data
		If a chunk of data was missing eg k=2 in kfold validation, then the offsets would be incorrect
		Therefore have to time offset train/test data and then split into train/test data
		Train_Test_Splitter takes an input_target_maker that has already been time_offset and prepared
		It then creates two (separate train/test) input_target_makers composed of subsets of the provided 
		traintest data with a given k
	"""
	def __init__(self):
		self.k = None
		self.current_k = None
		self.combined_train_test_itm = None
		self._idxs_test = None
		self._idxs_train = None
		self._sizes = None
		self._test_input_target_maker = None
		self._train_input_target_maker = None

	def split_train_and_test(self):
		self._calc_train_test_idxs_for_current_k()
		self._create_separate_test_and_train_input_target_makers()
		return self._train_input_target_maker, self._test_input_target_maker

	def _calc_train_test_idxs_for_current_k(self):
		self._calculate_train_test_sizes()
		self._make_test_idxs()
		self._make_train_idxs_with_test_idxs(self._idxs_test)

	def _calculate_train_test_sizes(self):
		size_train_test = self.combined_train_test_itm.input_maker.dataset_object.get_number_rows()
		size_test = int(size_train_test / self.k)
		size_train = size_train_test - size_test
		self._sizes = {'train':size_train, 'test':size_test, 'all':size_train_test}

	def _make_test_idxs(self):
		idx_start = self._sizes['test'] * self.current_k
		idx_end = idx_start + self._sizes['test']
		self._idxs_test = list(range(idx_start,idx_end))

	def _make_train_idxs_with_test_idxs(self, idxs_test):
		idxs_test_train = list(range(self._sizes['all']))
		self._idxs_train = [x for x in idxs_test_train if x not in idxs_test]

	def _create_separate_test_and_train_input_target_makers(self):
		self._test_input_target_maker = self._subset_traintest_itm_with_idxs(self._idxs_test)
		self._train_input_target_maker = self._subset_traintest_itm_with_idxs(self._idxs_train)

	def _subset_traintest_itm_with_idxs(self, idxs):
		itm = Input_And_Target_Maker_2()
		itm.input_maker.dataset_object = self._subset_maker_with_idxs(self.combined_train_test_itm.input_maker, idxs)
		itm.target_maker.dataset_object = self._subset_maker_with_idxs(self.combined_train_test_itm.target_maker, idxs)
		itm.source_dataset_object = self.combined_train_test_itm.source_dataset_object
		itm.time_format = self.combined_train_test_itm.time_format
		return itm

	def _subset_maker_with_idxs(self, maker, idxs):
		df = maker.dataset_object.df
		new_df = pd.DataFrame(df.iloc[idxs,:])
		new_df.columns = df.columns.values
		new_df.index = df.index.values[idxs]
		new_d = Dataset_With_Time_Offset()
		new_d.df = new_df
		new_d.time_offsets_list = maker.dataset_object.time_offsets_list
		return new_d