from .ITM_Fill_Time_Gaps import ITM_Fill_Time_Gaps
from .ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average


# parent at the end comes first
class ITM_Fill_Time_Gaps_Normalized_Moving_Average(ITM_Normalized_Moving_Average,ITM_Fill_Time_Gaps):
	def __init__(self):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self).__init__()

	def make_source_data(self,index_col=None):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self).make_source_data()
		
	def copy_parameters_from_maker(self,mkr):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self).copy_parameters_from_maker(mkr)

	def _common_make(self,maker):
		super(ITM_Fill_Time_Gaps_Normalized_Moving_Average,self)._common_make(maker)
