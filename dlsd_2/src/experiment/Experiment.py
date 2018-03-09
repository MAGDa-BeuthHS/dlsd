import logging

from dlsd_2.src.calc.error import MAE
from dlsd_2.src.experiment.Experiment_Config import *
from dlsd_2.src.experiment.experiment_output_reader.Experiment_Average_Error_Calculator import \
    Experiment_Average_Error_Calculator
from dlsd_2.src.experiment.experiment_output_reader.Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import *

from dlsd_2.src.io.input_target_maker.Input_Target_Maker import Input_And_Target_Maker_2
from .experiment_helper.Experiment_Helper import Experiment_Helper

logging.basicConfig(level=logging.DEBUG)  # filename='17_05_04_dlsd_2_trials.log',)


class Experiment:
    def __init__(self):
        self.config = Experiment_Config()
        self.source_maker = None
        self.models = []
        self.model_io_params = []
        self.current_exp_helper = None
        self.current_maker_train = None
        self.current_maker_test = None

    def set_source_maker(self, source_maker):
        self.source_maker = source_maker

    def set_models(self, models):
        self.models = models

    def set_model_io_params(self, model_io_params):
        self.model_io_params = model_io_params

    def _gather_experiment(self):
        self._define_source_maker()
        self._define_model_io_params()
        self._define_models()
        self.source_maker.prepare_source_data()

    def _define_source_maker(self):
        raise NotImplementedError

    def _define_models(self):
        raise NotImplementedError

    def _define_model_io_params(self):
        raise NotImplementedError

    def _define_error_calculator(self):
        return Experiment_Error_Calculator()

    def run_experiment(self):
        self._gather_experiment()
        for model in self.models:
            for io_param in self.model_io_params:
                self._experiment_setup(io_param)
                self._train_and_test_single_model(model)
            self._calculate_accuracy_of_models()

    def _experiment_setup(self, io_param):
        self.config.io_param_name = io_param
        self.current_exp_helper = Experiment_Helper(self.config)
        self.current_exp_helper.setup_directory()
        self.current_maker_train = Input_And_Target_Maker_2(self.source_maker.train, io_param, self.source_maker.time_format_train)
        self.current_maker_test = Input_And_Target_Maker_2(self.source_maker.test, io_param, self.source_maker.time_format_test)
        self.current_maker_train.make_input_and_target()
        self.current_maker_test.make_input_and_target()
        self._write_maker_target_data_to_file(self.current_maker_test, self.config.get_predictions_dir_path())

    def _write_maker_target_data_to_file(self, itm, output_file):
        target_df = itm.get_target_df()
        target_df.to_csv(output_file)

    def _train_and_test_single_model(self, model):
        model.set_experiment_helper(self.current_exp_helper)
        model.train_with_prepared_input_target_maker(self.current_maker_train)
        model.test_with_prepared_input_target_maker(self.current_maker_test)
        output_path = self.current_exp_helper.make_new_model_prediction_file_path_with_model_name(model.name)
        model.write_predictions_to_path(output_path)

    def _calculate_accuracy_of_models(self):
        logging.debug("calculating average error")
        analyzer = self._define_error_calculator()
        analyzer.set_root_experiment_directory(self.config.get_root_path)
        analyzer.set_analysis_functions([MAE()])
        analyzer.analyze_all_sensors()
        logging.debug("calculating average error")
        avg = Experiment_Average_Error_Calculator()
        avg.set_root_experiment_directory(self.config.get_root_path)
        avg.calculate_average()

    def _collect_all_model_accuracies(self):
        model_prediction_accuracies = []
        for model in self.models:
            model_prediction_accuracies.append(model.calc_prediction_accuracy())
        return model_prediction_accuracies

        # def create_average_week(self):
    #     self._define_source_maker()
    #     self.source_maker.prepare_source_data()
    #     model = Average_Week()
    #     model.create_average_week_with_source_maker(self.source_maker)
    #     model.write_average_week_to_filepath(PATH_AVERAGE_WEEK)
