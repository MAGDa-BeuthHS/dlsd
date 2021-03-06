import pandas as pd

from dlsd_2.src.calc.error import *
from dlsd_2.src.io.dataset.Dataset import Dataset


class Model_Output:

    def __init__(self):
        self.prediction_dataset_object = Dataset()
        self.target_dataset_object = Dataset()

    def set_prediction_dataset_object_with_numpy_array(self, numpy_array):
        self.prediction_dataset_object.set_numpy_array(numpy_array)
        self.prediction_dataset_object.df.columns = self.target_dataset_object.df.columns.values

    def set_target_dataset_object(self, dataset_object):
        self.target_dataset_object = dataset_object

    def get_prediction_df(self):
        self.prediction_dataset_object.df.index = self.target_dataset_object.df.index.values[
                                                  0:self.prediction_dataset_object.df.shape[0]]
        return self.prediction_dataset_object.df

    def calc_mae(self):
        my_error = mae(self.prediction_dataset_object.df, self.target_dataset_object.df)
        return my_error

    def calc_mape(self):
        return mape(self.prediction_dataset_object.df, self.target_dataset_object.df)

    def make_target_and_predictions_df(self):
        array = np.concatenate((self.target_dataset_object.df.values, self.prediction_dataset_object.df.values), axis=1)
        new_df = pd.DataFrame(array)
        new_df.index = self.target_dataset_object.df.index.values
        names_target = ["target_%d" % x for x in self.target_dataset_object.df.columns.values]
        names_predict = ["predict_%d" % x for x in self.prediction_dataset_object.df.columns.values]
        print(self.target_dataset_object.index.values)
        print("IN MODEL OUTPUT")
        new_df.columns = names_target + names_predict
        return new_df

    def fill_time_gaps_in_target_and_predictions_using_time_format(self, time_format):
        self.target_dataset_object.fill_time_gaps(time_format)
        self.fill_time_gaps_in_predictions_using_time_format(time_format)

    def fill_time_gaps_in_predictions_using_time_format(self, time_format):
        print(self.prediction_dataset_object.df.shape)
        print(self.target_dataset_object.df.shape)
        self.prediction_dataset_object.df.index = self.target_dataset_object.df.index.values[
                                                  0:self.prediction_dataset_object.df.shape[0]]
        self.prediction_dataset_object.fill_time_gaps(time_format)
