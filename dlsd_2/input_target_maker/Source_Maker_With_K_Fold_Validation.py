from dlsd_2.input_target_maker.Source_Maker import *
import pandas as pd

class Source_Maker_With_K_Fold_Validation(Source_Maker):
	def __init__(self):
		super(Source_Maker_With_K_Fold_Validation, self).__init__()
		self.file_path_all_data = None
		self.validation_percentage = None

	def get_all_sensors(self):
		return self.all_data.get_column_names()

	def read_data_from_csv(self):
		self.all_data = Dataset()
		self.all_data.read_csv(self.file_path_all_data,index_col = 0)

	def apply_modifications_defined_in_parameters(self):
		if self.moving_average_window is not None:
			self.all_data.rolling_average_with_window(self.moving_average_window)

	def remove_validation_data(self):
		self.calculate_train_test_validation_sizes()
		idxs_validation = range(0,self._sizes['validation'])
		idxs_all = list(range(self._sizes['all']))
		idxs_train_test = [i for i in idxs_all if i not in idxs_validation]
		self.validation = self.create_subset_of_all_data_with_idxs(idxs_validation)
		self.train_test = self.create_subset_of_all_data_with_idxs(idxs_train_test)

	def calculate_train_test_validation_sizes(self):
		size_all = self.all_data.get_number_rows()
		size_validation = int(size_all*(self.validation_percentage/100))
		self._sizes = {'validation':size_validation, 'all':size_all}

	def create_subset_of_all_data_with_idxs(self, row_range):
		d = Dataset()
		d.df = pd.DataFrame(self.all_data.df.iloc[row_range,:])
		return d