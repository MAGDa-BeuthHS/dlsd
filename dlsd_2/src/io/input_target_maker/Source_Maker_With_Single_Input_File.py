from dlsd_2.src.io.dataset.Dataset import Dataset
from dlsd_2.src.io.input_target_maker.Source_Maker import Source_Maker


class Source_Maker_With_Single_Input_File(Source_Maker):
    def __init__(self):
        super(Source_Maker_With_Single_Input_File, self).__init__()
        self.file_path_all_data = None

    def read_data_from_csv(self):
        self.all_data = Dataset()
        self.all_data.read_csv(self.file_path_all_data, index_col=0)

    def apply_parameters(self):
        if self.moving_average_window is not None:
            self.all_data.rolling_average_with_window(self.moving_average_window)

    def get_all_sensors(self):
        return self.all_data.get_column_names()
