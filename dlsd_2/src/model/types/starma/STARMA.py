import datetime as dt

from dlsd_2.src.model.Model import *
from dlsd_2.src.model.types.starma.STARMA_Content import *


class STARMA(Model):
    def __init__(self):
        super(STARMA, self).__init__()
        logging.info("Creating pySTARMA model")
        self.model_content = STARMA_Content()
        self.name = "pySTARMA"

    def _build(self, source_maker):
        self.source_maker = source_maker
        self.model_content.preprocess_pystarma_model_with_source_maker(self, source_maker)

    def _train(self):
        self.model_content.fit_model()

    def _test(self):
        self.model_content.set_input_target_maker(self.current_input_target_maker)
        predictions = self.model_content.make_prediction_dataset_object()
        super(STARMA, self).set_model_output_with_predictions_numpy_array(predictions.df.values)  # TODO check works

    def _get_day_begin_integer_from_target_dataset(self):
        first_time_stamp_string = self.current_input_target_maker.target_dataset_object.df.index.values[0, 0]
        first_timestampe_datetime = dt.datetime.strptime(first_time_stamp_string,
                                                         self.current_input_target_maker.time_format)

    def set_pystarma_time_series(self, file_path):
        self.model_content.set_pystarma_time_series(file_path)

    def set_pystarma_weight_matrices(self, file_path, file_names):
        self.model_content.set_pystarma_weight_matrices(file_path, file_names)
