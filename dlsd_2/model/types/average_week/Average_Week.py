from dlsd_2.model.Model import *
from dlsd_2.model.types.average_week.Average_Week_Content import *
from dlsd_2.input_target_maker.Maker import *

class Average_Week(Model):

	def __init__(self):
		super(Average_Week,self).__init__()
		logging.info("Creating Average Week model")
		self.average_week_externally_set = False
		self.model_content = Average_Week_Content()

	def train_with_input_target_maker(self,input_target_maker):
		if self.average_week_externally_set is False:
			super(Average_Week,self).train_with_input_target_maker(input_target_maker)
		else:
			self.train_input_target_maker = input_target_maker # need to set this so parameters copied to test_input_target_maker during testing


	def _train(self):
		self.model_content.create_source_average_data_with_input_target_maker(self.current_input_target_maker)

	def _test(self):
		self.model_content.set_input_target_maker(self.current_input_target_maker)
		self.model_output.prediction_dataset_object = self.model_content.make_prediction_dataset_object()
		self.model_output.target_dataset_object = self.current_input_target_maker.get_target_dataset_object()

	def _get_day_begin_integer_from_target_dataset(self):
		first_time_stamp_string = self.current_input_target_maker.target_dataset_object.df.index.values[0,0]
		first_timestampe_datetime = datetime.datetime.strptime(first_time_stamp_string,self.current_input_target_maker.time_format)

	def set_average_data_from_csv_file_path(self,file_path):
		self.average_week_externally_set = True
		self.model_content.set_average_data_from_csv_file_path(file_path)

	def write_average_week_to_filepath(self, file_path):
		self.model_content.source_average_dataset_object.write_csv(file_path)