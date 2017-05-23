from .ITM_Fill_Time_Gaps import ITM_Fill_Time_Gaps
from .ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average


class ITM_Fill_Time_Gaps_Normalized_Moving_Average(ITM_Fill_Time_Gaps,ITM_Normalized_Moving_Average):
	def __init__(self):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self).__init__()

	def make_source_data(self,index_col=None):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self).make_source_data()
		
	def copy_parameters_from_maker(self,mkr):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self).copy_parameters_from_maker(mkr)
