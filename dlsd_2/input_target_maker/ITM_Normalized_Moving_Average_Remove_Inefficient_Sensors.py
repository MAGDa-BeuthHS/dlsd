from .ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average


class ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors(ITM_Normalized_Moving_Average):
	def __init__(self):
		super(ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors,self).__init__()
		self.efficiency_threshold = 1.0

	def make_source_data(self,index_col=None):
		super(ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors,self).make_source_data()
		self.source_dataset_object.remove_inefficient_sensors(self.efficiency_threshold)

	def copy_parameters_from_maker(self,mkr):
		super(ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors,self).copy_parameters_from_maker(mkr)

	def _common_make(self,maker):
		super(ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors,self)._common_make(maker)

	def write_sensor_efficiency_to_file(self,file_path):
		self.source_dataset_object.write_sensor_efficiency_to_file(file_path)

	def set_efficiency_threshold(self, threshold):
		self.efficiency_threshold = threshold