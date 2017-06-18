import pandas as pd

class Correct_Output_Table:

	def __init__(self):
		self.df = None
		self.input_columns = None
		self.target_columns = None

	def read_csv_at_path(self, file_path):
		self.df = pd.read_csv(file_path,index_col=0,header=0)

	def define_input_columns_list(self,input_columns):
		self.input_columns = input_columns

	def define_target_columns_list(self, target_columns):
		self.target_columns = target_columns

	def target_numpy_array(self):
		if (len(self.target_columns) == 1):
			return self.df.iloc[:,self.target_columns].values.reshape(-1,1)
		return self.df.iloc[:,self.target_columns].values

	def input_numpy_array(self):
		if (len(self.input_columns) == 1):
			return self.df.iloc[:,self.input_columns].values.reshape(-1,1)
		return self.df.iloc[:,self.input_columns].values