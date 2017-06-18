from .Input_And_Target_Maker import Input_And_Target_Maker


class ITM_Remove_NaNs(Input_And_Target_Maker):
	def __init__(self):
		super(ITM_Remove_NaNs,self).__init__()

	def make_source_data(self,index_col=None):
		super(ITM_Remove_NaNs,self).make_source_data()
		self.source_dataset_object.remove_any_rows_with_NaN()

	def copy_parameters_from_maker(self,mkr):
		super(ITM_Remove_NaNs,self).copy_parameters_from_maker(mkr)

	def _common_make(self,mkr):
		super(ITM_Remove_NaNs,self)._common_make(mkr)
