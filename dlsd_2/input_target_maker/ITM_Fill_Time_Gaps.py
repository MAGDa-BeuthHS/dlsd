from .Input_And_Target_Maker import Input_And_Target_Maker


class ITM_Fill_Time_Gaps(Input_And_Target_Maker):
	def __init__(self):
		super(ITM_Fill_Time_Gaps,self).__init__()

	def make_source_data(self,index_col=None):
		super(ITM_Fill_Time_Gaps,self).make_source_data()
		self._check_for_time_format()
		self.source_dataset_object.fill_time_gaps_using_time_format(self.time_format)

	def copy_parameters_from_maker(self,mkr):
		super(ITM_Fill_Time_Gaps,self).copy_parameters_from_maker(mkr)
		if self.time_format is None : # in some instances training/test time_formats will differ! if training explicitly set, don't overwrite
			self.time_format = mkr.time_format

	def _common_make(self,mkr):
		super(ITM_Fill_Time_Gaps,self)._common_make(mkr)

	def _check_for_time_format(self):
		if self.time_format is None:
			raise Exception("Need to provide a time_format to fill time gaps")

