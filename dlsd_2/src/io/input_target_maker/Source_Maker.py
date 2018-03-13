from dlsd_2.src.io.dataset import Dataset
from dlsd_2.src.io.dataset import Dataset_From_SQL


class Source_Maker:
    def __init__(self):
        self.file_path_train = None
        self.file_path_test = None
        self.is_sql_output = True
        self.normalize = False
        self.moving_average_window = 15
        self.remove_inefficient_sensors_below_threshold = None
        self.fill_time_gaps = False
        self.remove_nans = False
        self.time_format_train = None
        self.time_format_test = None
        self.train = None
        self.test = None

    def read_source_data(self, file_path):
        if self.is_sql_output:
            d = Dataset_From_SQL()
            d.read_csv(file_path)
            d.pivot()
        else:
            d = Dataset()
            d.read_csv(file_path, index_col=0)
        return d

    def read_data_from_csv(self):
        self.train = self.read_source_data(self.file_path_train)
        self.test = self.read_source_data(self.file_path_test)

    def apply_parameters(self):
        if self.moving_average_window is not None:
            self.train.rolling_average_with_window(self.moving_average_window)
            self.test.rolling_average_with_window(self.moving_average_window)
        if self.normalize:
            self.train.normalize()
            self.test.denormalizer = self.train.denormalizer
            self.test.normalize()
        if self.remove_inefficient_sensors_below_threshold is not None:
            self.train.remove_inefficient_sensors(self.remove_inefficient_sensors_below_threshold)
            self._match_test_and_train_sensors()
        if self.fill_time_gaps:
            self.train.fill_time_gaps(self.time_format_train)
            self.test.fill_time_gaps(self.time_format_test)

    def _match_test_and_train_sensors(self):
        desired_sensors = self.train.df.columns.values
        self.test.df = self.test.df[desired_sensors]

    def prepare_source_data(self):
        self.read_data_from_csv()
        self.apply_parameters()
