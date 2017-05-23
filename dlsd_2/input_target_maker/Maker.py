from dlsd_2.dataset.Dataset import Dataset
from dlsd_2.dataset.Dataset_With_Time_Offset import Dataset_With_Time_Offset
import logging

class Maker:
	def __init__(self):
		self.selected_source_object = None # the subset of source data (ex a single sensor) in input/target
		self.selected_source_numpy_data = None
		self.dataset_object = None
		self.sensor_idxs_list = None
		self.sensor_names_list = None
		self.time_offsets_list = None
		self.top_padding = 0
		self.bottom_padding = 0

	def extract_sensor_idxs_from_source_dataset_object(self,source_dataset_object):
		if self.sensor_idxs_list is None:
			self._set_sensor_idxs_to_use_all_sensors_from_source(source_dataset_object)
		self.selected_source_numpy_data = source_dataset_object.get_numpy_columns_at_idxs(self.sensor_idxs_list)

	def make_dataset_object_with_clip_range(self, clip_range):
		if self.time_offsets_list is None:
			raise Exception('no time offsets are set')
		self.dataset_object = Dataset_With_Time_Offset()
		self.dataset_object.set_source_numpy_array(self.selected_source_numpy_data)
		self.dataset_object.set_top_padding(self.top_padding)
		self.dataset_object.set_bottom_padding(self.bottom_padding)
		self.dataset_object.set_time_offsets_list(self.time_offsets_list)
		self.dataset_object.create_time_offset_data()
		self.dataset_object.clip_ends_keep_data_between_indices(clip_range)

	def _set_sensor_idxs_to_use_all_sensors_from_source(self, source_dataset_object):
		self.sensor_idxs_list = list(range(0,source_dataset_object.get_number_columns()))

	def max_time_offset(self):
		return max(self.time_offsets_list)