import os

class Experiment_Config:
    '''
        |Experiment location
        |---root_dir
        |   |---tensorflow_dir
            |---targets.csv
            |---predictions
                |---model_1.csv
                |---model_2.csv
    '''

    def __init__(self):
        self.io_param_name = None
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.experiment_output_path = os.path.dirname(os.path.abspath(__file__))
        self.path_io_param = os.path.dirname(os.path.abspath(__file__))
        self.tensorflow_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.predictions_dir_path = os.path.dirname(os.path.abspath(__file__))

    def get_root_path(self):
        return self.root_path

    def set_root_path(self, root_path):
        self.root_path = root_path

    def get_experiment_output_path(self):
        return self.experiment_output_path

    def set_experiment_output_path(self, path):
        self.experiment_output_path = path
        self.set_root_path(path)

    def get_io_param_name(self):
        return self.io_param_name

    def set_io_param_name(self, io_param_name):
        self.io_param_name = io_param_name

    def get_path_io_param(self):
        return self.path_io_param

    def set_path_io_param(self, path_io_param):
        self.path_io_param = path_io_param

    def get_tensorflow_dir_path(self):
        return self.tensorflow_dir_path

    def set_tensorflow_dir_path(self, tensorflow_dir_path):
        self.tensorflow_dir_path = tensorflow_dir_path

    def get_predictions_dir_path(self):
        return self.predictions_dir_path

    def set_predictions_dir_path(self, predictions_dir_path):
        self.predictions_dir_path = predictions_dir_path