from .Maker import *
from dlsd_2.dataset.Dataset_From_SQL import Dataset_From_SQL

class Input_And_Target_Maker:
	def __init__(self):
		self.source_file_path = None
		self.source_dataset_object = None
		self.input_maker = Maker()
		self.target_maker = Maker()
		self.clip_range = None
		self.source_is_sql_output = True

	def prepare_source_data_and_make_input_and_target(self):
		self.make_source_data()
		self.make_input_and_target()

	def make_source_data(self,index_col=None):
		if self.source_is_sql_output:
			self._make_source_data_from_SQL()
		else:
			self._make_source_data_from_csv(index_col=0)

	def _make_source_data_from_csv(self,index_col=None):
		if self.source_file_path is None: 
			raise Exception("Must provide a source file path for input and target and call set_source_file_path")
		self.source_dataset_object = Dataset()
		self.source_dataset_object.read_csv(self.source_file_path,index_col=index_col)

	def _make_source_data_from_SQL(self):
		if self.source_file_path is None: 
			raise Exception("Must provide a source file path")
		self.source_dataset_object = Dataset_From_SQL()
		self.source_dataset_object.read_csv(self.source_file_path)
		self.source_dataset_object.pivot()

	def set_input_sensor_idxs_and_timeoffsets_lists(self,idxs_list,offset_list):
		self.input_maker.sensor_idxs_list = idxs_list
		self.input_maker.time_offsets_list = offset_list

	def set_target_sensor_idxs_and_timeoffsets_lists(self,idxs_list,offset_list):
		self.target_maker.sensor_idxs_list = idxs_list
		self.target_maker.time_offsets_list = offset_list

	def make_input_and_target(self):
		self._calc_clip_range_so_input_target_same_size()
		self._make_input()
		self._make_target()

	def _calc_clip_range_so_input_target_same_size(self):
		'''
			Because of time offsetting, where columns are 'slid' past eachother, new NaN rows are created at the top and bottom
			(top_padding and bottom_padding, as well as offestting within the target/input)
			these NaN's must be 'clipped' : keep the data within the wrange calculated here
		'''
		top = self.target_maker.max_time_offset() + self.input_maker.max_time_offset()
		bot = -1*(self.target_maker.max_time_offset() + self.input_maker.max_time_offset())
		self.clip_range = [top,bot]

	def _make_input(self):
		self.input_maker.top_padding = self.target_maker.max_time_offset()
		self._common_make(self.input_maker)

	def _make_target(self):
		self.target_maker.bottom_padding = self.input_maker.max_time_offset()
		self._common_make(self.target_maker)

	def _common_make(self,maker):
		maker.extract_sensor_idxs_from_source_dataset_object(self.source_dataset_object)
		maker.make_dataset_object_with_clip_range(self.clip_range)

	def set_source_file_path(self,file_path):
		self.source_file_path = file_path

	def set_input_time_offsets_list(self, the_list):
		self.input_maker.time_offsets_list = the_list

	def set_target_time_offsets_list(self, the_list):
		self.target_maker.time_offsets_list = the_list

	def set_target_sensor_idxs_list(self, the_list):
		self.target_maker.sensor_idxs_list = the_list

	def get_target_time_offsets_list(self):
		return self.target_maker.time_offsets_list

	def get_input_sensor_idx_lists(self):
		return self.input_maker.sensor_idxs_list

	def get_input_time_offsets_list(self):
		return self.input_maker.time_offsets_list

	def get_input_dataset_object(self):
		return self.input_maker.dataset_object

	def get_target_dataset_object(self):
		return self.target_maker.dataset_object

	def get_input_sensor_names_list(self):
		return self.source_dataset_object.df.columns.values[self.input_maker.sensor_idxs_list]
	
	def get_target_sensor_idxs_list(self):
		return self.target_maker.sensor_idxs_list

	def get_source_idxs_list(self):
		return self.source_dataset_object.df.columns.values

	def copy_parameters_from_maker(self,mkr):
		self.set_input_sensor_idxs_and_timeoffsets_lists(mkr.input_maker.sensor_idxs_list, mkr.input_maker.time_offsets_list)
		self.set_target_sensor_idxs_and_timeoffsets_lists(mkr.target_maker.sensor_idxs_list, mkr.target_maker.time_offsets_list)
	
	def write_source_to_file(self,file_path):
		self.source_dataset_object.write_csv(file_path)
		