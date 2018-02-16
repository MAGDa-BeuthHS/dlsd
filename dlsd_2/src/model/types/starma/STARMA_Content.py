import logging

import numpy as np
from dlsd_2.dataset.Dataset import Dataset

from dlsd_2.src.model.Model_Content import Model_Content


class STARMA_Content(Model_Content):
	
	def __init__(self):
		super(STARMA_Content,self).__init__()
		self._wa_matrices = Dataset() # contains spatial weights
		self._ts_matrix = Dataset() # contains all sensors
		self.prediction_dataset_object = Dataset()

		self._p = None
		self._q = None
		self._max_p_tlag = None
		self._max_q_tlag = None #self._get_max_tlag(q)
		self._max_tlag = max(self._max_p_tlag, self._max_q_tlag)
		self._iter = None

		self.num_target_rows = None
		self.num_target_sensors = None
		self.num_target_offsets = None
		self.input_target_maker = None

	def set_input_target_maker(self, input_target_maker, test = False):
		self.input_target_maker = input_target_maker
		self.num_target_sensors = len(input_target_maker.get_target_sensor_idxs_list())
		self.num_target_offsets = len(input_target_maker.get_target_time_offsets_list())
		self._set_parameters_from_input_target_maker() if test is False else logging.debug("Testing")

	def _set_parameters_from_input_target_maker(self):
		self.num_target_rows = self.input_target_maker.get_target_dataset_object().get_number_rows()

	def set_starma_from_csv_file_path(self,file_path):
		self._ts_matrix = Dataset()
		self._ts_matrix.read_csv(file_path,index_col=0)

	def make_prediction_dataset_object(self):
		size = [self.num_target_rows , self.num_target_offsets*self.num_target_sensors]
		array = np.zeros(size)
		self._extract_sensors_currently_being_used_as_output_from_source_data()
		self._iterate_over_target_time_offsets_replicating_average_week(array)
		self.prediction_dataset_object.set_numpy_array(array)
		return self.prediction_dataset_object

	def _extract_sensors_currently_being_used_as_output_from_source_data(self):
		self.subset_starma_array = self._ts_matrix.df[self.input_target_maker.get_target_sensor_idxs_list()].values

	def _target_time_offset_at_index(self, index):
		return self.input_target_maker.get_target_time_offsets_list()[index]
	
	def _max_input_time_offset(self):
		return max(self.input_target_maker.get_input_time_offsets_list())