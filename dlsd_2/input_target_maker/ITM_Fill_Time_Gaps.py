from .Input_And_Target_Maker import Input_And_Target_Maker


class ITM_Fill_Time_Gaps(Input_And_Target_Maker):
	def __init__(self):
		super(ITM_Fill_Time_Gaps,self).__init__()
		self.time_format = None

	def set_time_format(self,time_format):
		self.time_format = time_format

	def make_source_data(self,index_col=None):
		super(ITM_Fill_Time_Gaps,self).make_source_data()
		self._check_for_time_format()
		self.source_dataset_object.fill_time_gaps_using_time_format(self.time_format)

	def make_source_data(self):

		super(ITM_Fill_Time_Gaps,self).make_source_data()
		self._check_for_time_format()
		self.source_dataset_object.fill_time_gaps_using_time_format(self.time_format)

	def _check_for_time_format(self):
		if self.time_format is None:
			raise Exception("Need to provide a time_format to fill time gaps")

	def copy_parameters_from_maker(self,mkr):
		super(ITM_Fill_Time_Gaps,self).copy_parameters_from_maker(mkr)
		self.time_format = mkr.time_format