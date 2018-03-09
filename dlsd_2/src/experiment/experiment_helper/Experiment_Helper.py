import os

from dlsd_2.src.experiment.Experiment_Config import *


class Experiment_Helper:
    '''
        |Experiment location
        |---root_dir
        |   |---tensorflow_dir
            |---targets.csv
            |---predictions
                |---model_1.csv
                |---model_2.csv
    '''

    def __init__(self, config):
        self.config = Experiment_Config(*config)

    def setup_directory(self):
        self.check_output_dirs_exist()
        self.create_io_param_dir()

    def check_output_dirs_exist(self):
        self.check_or_make_dir(self.config.get_experiment_output_path)
        self.check_or_make_dir(self.config.get_root_path)

    def create_io_param_dir(self):
        self.config.set_path_io_param(os.path.join(self.config.get_root_path, self.config.get_io_param_name))
        self.check_or_make_dir(self.config.get_path_io_param)
        self.create_child_dirs()

    def create_child_dirs(self):
        self.config.set_tensorflow_dir_path(self.add_directory_to_io_param_dir("tensorflow"))
        self.config.set_predictions_dir_path(self.add_directory_to_io_param_dir("predictions"))

    def add_directory_to_io_param_dir(self, child_name):
        path = os.path.join(config.get_path_io_param, child_name)
        return self.check_or_make_dir(path)

    def check_or_make_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name

    def new_predictions_file_path_with_specifier(self, name):
        return os.path.join(config.get_predictions_dir_path, name + ".csv")

    def new_tf_session_file_path_with_specifier(self, name):
        new_tf_session_path = os.path.join(self.config.get_tensorflow_dir_path, "tf_session_" + name)
        return new_tf_session_path

    def get_target_file_path(self):
        return os.path.join(self.config.get_path_io_param, "target.csv")

    def make_new_model_prediction_file_path_with_model_name(self, model_name):
        filename = model_name + ".csv"
        filepath = os.path.join(self.config.get_predictions_dir_path, filename)
        return filepath
