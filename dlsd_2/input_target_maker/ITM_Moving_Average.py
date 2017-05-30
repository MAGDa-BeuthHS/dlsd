from .Input_And_Target_Maker import Input_And_Target_Maker

class ITM_Moving_Average(Input_And_Target_Maker):
	def __init__(self):
		super(ITM_Moving_Average,self).__init__()
		self.moving_average_window = None

	def make_source_data(self):
		super(ITM_Moving_Average,self).make_source_data()
		self.source_dataset_object.rolling_average_with_window(self.moving_average_window)

	def copy_parameters_from_maker(self,mkr):
		super(ITM_Moving_Average,self).copy_parameters_from_maker(mkr)
		self.set_moving_average_window(mkr.moving_average_window)

	def _common_make(self,maker):
		super(ITM_Moving_Average,self)._common_make(maker)

	def set_moving_average_window(self,moving_average_window):
		self.moving_average_window = moving_average_window
	 