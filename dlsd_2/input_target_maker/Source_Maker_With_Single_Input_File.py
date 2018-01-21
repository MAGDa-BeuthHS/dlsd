from dlsd_2.input_target_maker.Source_Maker import *
import pandas as pd

class Source_Maker_With_Single_Input_File(Source_Maker):
	def __init__(self):
		super(Source_Maker_With_Single_Input_File, self).__init__()
		self.file_path_all_data = None

	def get_all_sensors(self):
		return self.all_data.get_column_names()

	def read_data_from_csv(self):
		self.all_data = Dataset()
		self.all_data.read_csv(self.file_path_all_data,index_col = 0)

	def apply_modifications_defined_in_parameters(self):
		if self.moving_average_window is not None:
			self.all_data.rolling_average_with_window(self.moving_average_window)