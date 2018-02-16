from dlsd_2.input_target_maker.Maker import *
from dlsd_2.model.types.starma.STARMA_Content import *
from pySTARMA.starma_model._model import *

from dlsd_2.src.model.Model import *


class STARMA(Model):

	def __init__(self):
		super(STARMA,self).__init__()
		logging.info("Creating STARMA model")
		self.model_content = STARMA_Content()
		self.name = "STARMA"

	def create_starma_with_source_maker(self, source_maker):
		self.model_content.create_starma_with_source_maker(source_maker)

	def _train(self):
		pass

	def _test(self):
		self.model_content.set_input_target_maker(self.current_input_target_maker)
		predictions = self.model_content.make_prediction_dataset_object()
		super(STARMA,self).set_model_output_with_predictions_numpy_array(predictions.df.values) # TODO check works

	def _get_day_begin_integer_from_target_dataset(self):
		first_time_stamp_string = self.current_input_target_maker.target_dataset_object.df.index.values[0,0]
		first_timestampe_datetime = datetime.datetime.strptime(first_time_stamp_string,self.current_input_target_maker.time_format)

	def set_average_data_from_csv_file_path(self,file_path):
		''' Single average week is provided : training doesn't have to occur '''
		self.average_week_externally_set = True
		self.model_content.set_average_data_from_csv_file_path(file_path)

	def write_average_week_to_filepath(self, file_path):
		''' After training a single average week is created. Print this out here '''
		self.model_content.source_average_dataset_object.write_csv(file_path)

