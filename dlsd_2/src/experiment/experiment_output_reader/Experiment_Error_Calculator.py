import os
import pandas as pd
import numpy as np


class Experiment_Error_Calculator:
    def __init__(self):
        self.experiment_output_reader = None
        self.output_path = None
        self.analysis_function = None
        self.col_num = None
        self.mae_tables = []
        self.current_table = None

    def set_experiment_output_reader(self, experiment_output_reader):
        self.experiment_output_reader = experiment_output_reader

    def set_output_path(self, output_path):
        self.output_path = output_path

    def set_analysis_function(self, function):
        self.analysis_function = function

    def do_analysis(self):
        self._create_analysis_output_directory()
        self._iterate_models_from_reader()

    def _create_analysis_output_directory(self):
        self.set_output_path(self._check_or_make_dir(os.path.join(self.output_path, self.analysis_function.name)))

    def _iterate_models_from_reader(self):
        model_names = self.experiment_output_reader.get_model_names()
        for model_name in model_names:
            self._write_analysis_file(model_name)

    def _write_analysis_file(self, model_name):
        dicts_for_model = self.experiment_output_reader.get_predictions_and_target_for_model_name(model_name)
        self._create_empty_tables_for_current_model(dicts_for_model)
        for i in range(1, len(dicts_for_model)):
            mae = self._calculate_mae(dicts_for_model[i])
            self.current_table.iloc[i - 1, :] = mae.values  # i-1 because first in list is the target(made by reader)
        path = os.path.join(self.output_path, model_name + ".csv")
        self.current_table.to_csv(path)

    def _create_empty_tables_for_current_model(self, dicts_for_model):
        self.col_num = dicts_for_model[0]['df'].shape[1]
        columns = dicts_for_model[0]['df'].columns.values
        index = self.experiment_output_reader.get_io_param_names()

        self.current_table = pd.DataFrame(np.zeros((len(dicts_for_model) - 1, self.col_num)))
        try:
            self.current_table.columns = columns
            self.current_table.index = index
        except:
            print(self.current_table.shape)
            print(self.experiment_output_reader.__dict__)
            print(index)
            print("TABLE DID NOT FIT")
            pass

    def _calculate_mae(self, dict_for_model):
        prediction = dict_for_model['df']
        target = dict_for_model['target']
        l = self._get_length_of_array(prediction, target)
        try:
            mae = self.analysis_function.calc_error_with_prediction_and_target(prediction[0:l], target.values[
                                                                                           0:l])  # sometimes problems here
        except:
            print("CALCULATING MAE DID NOT WORK")
            print(dict_for_model.keys())
            mae = pd.DataFrame(np.zeros([1, 7]))
        return mae

    def _get_length_of_array(self, preds, targ):
        if preds.shape[0] < targ.shape[0]:
            return preds.shape[0]
        return targ.shape[0]

    def _check_or_make_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
