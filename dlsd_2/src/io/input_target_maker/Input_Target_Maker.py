from .Maker import *


class Input_And_Target_Maker_2:
    def __init__(self, source_dataset_object, io_param, time_format, clip_range=None):
        self.source_dataset_object = source_dataset_object
        self.io_param = io_param
        self.time_format = time_format
        self.clip_range = clip_range
        self.input_maker = Maker()
        self.target_maker = Maker()

    def set_all_sensor_idxs_and_time_offsets(self, params):
        self.io_param = params
        print("IN HERE : set_all_sensor_idxs_and_time_offsets_using_parameters_object")
        print(params.input_sensor_idxs_list)
        self.set_input_sensor_idxs_list(params.input_sensor_idxs_list)
        self.set_input_time_offsets_list(params.input_time_offsets_list)
        self._check_input_target_sensor_idxs(self.input_maker)
        self.set_target_time_offsets_list(params.target_time_offsets_list)
        self.set_target_sensor_idxs_list(params.target_sensor_idxs_list)
        self._check_input_target_sensor_idxs(self.target_maker)

    def make_input_and_target(self):
        self._calc_clip_range()
        self._make_input()
        self._make_target()
        self._set_input_target_row_names()
        print(self.input_maker.dataset_object.df.shape)

    def _calc_clip_range(self):
        '''
            Because of time offsetting, where columns are 'slid' past each other, new NaN rows are created at the top and bottom
            (top_padding and bottom_padding, as well as offsetting within the target/input)
            these NaN's must be 'clipped' : keep the data within the range calculated here
        '''
        top = self.target_maker.max_time_offset() + self.input_maker.max_time_offset()
        bot = -1 * (self.target_maker.max_time_offset() + self.input_maker.max_time_offset())
        self.clip_range = [top, bot]

    def _make_input(self):
        self._check_input_target_sensor_idxs(self.input_maker)
        print()
        self.input_maker.top_padding = self.target_maker.max_time_offset()
        self.input_maker.extract_sensor_and_row_names(self.source_dataset_object)
        if self.io_param.adjacency_matrix is not None:
            self.input_maker.multiply_by_adjacency_of_target_sensor_including_target_sensor(
                self.io_param.adjacency_matrix, self.target_maker.sensor_idxs_list,
                self.io_param.include_output_sensor_in_adjacency)
        self.input_maker.make_dataset_object_with_clip_range(self.clip_range)

    def _make_target(self):
        self._check_input_target_sensor_idxs(self.target_maker)
        self.target_maker.bottom_padding = self.input_maker.max_time_offset()
        self.target_maker.extract_sensor_and_row_names(self.source_dataset_object)
        self.target_maker.make_dataset_object_with_clip_range(self.clip_range)

    def _check_input_target_sensor_idxs(self, maker):
        if maker.sensor_idxs_list is None:
            maker.sensor_idxs_list = list(self.source_dataset_object.df.columns.values)
        print(len(maker.sensor_idxs_list))

    def _set_input_target_row_names(self):
        source_row_names = self.source_dataset_object.df.index.values
        clipped_row_names = source_row_names[0:len(source_row_names) - self.target_maker.bottom_padding - max(
            self.target_maker.time_offsets_list)]
        self.input_maker.dataset_object.df.index = clipped_row_names
        self.target_maker.dataset_object.df.index = clipped_row_names

    ## ----------------------------------------------------------

    def set_time_format(self, time_format):
        self.time_format = time_format

    def get_source_idxs_list(self):
        return self.source_dataset_object.df.columns.values

    def get_input_dataset_object(self):
        return self.input_maker.dataset_object

    def set_source_file_path(self, file_path):
        self.source_file_path = file_path

    def set_source_dataset_object(self, dataset_object):
        self.source_dataset_object = dataset_object
        self.denormalizer_used_in_training = self.source_dataset_object.denormalizer

    def get_input_sensor_names_list(self):
        return self.source_dataset_object.df.columns.values[self.input_maker.sensor_idxs_list]

    def get_input_sensor_idx_lists(self):
        return self.input_maker.sensor_idxs_list

    def set_input_sensor_idxs_list(self, the_list):
        self.input_maker.sensor_idxs_list = the_list

    def get_input_time_offsets_list(self):
        return self.input_maker.time_offsets_list

    def set_input_time_offsets_list(self, the_list):
        self.input_maker.time_offsets_list = the_list

    def set_input_sensor_idxs_and_time_offsets(self, idxs_list, offset_list):
        self.input_maker.sensor_idxs_list = idxs_list
        self.input_maker.time_offsets_list = offset_list

    def get_target_dataset_object(self):
        return self.target_maker.dataset_object

    def get_target_sensor_idxs_list(self):
        return self.target_maker.sensor_idxs_list

    def set_target_sensor_idxs_list(self, the_list):
        self.target_maker.sensor_idxs_list = the_list

    def get_target_time_offsets_list(self):
        return self.target_maker.time_offsets_list

    def set_target_time_offsets_list(self, the_list):
        self.target_maker.time_offsets_list = the_list

    def set_target_sensor_idxs_and_time_offsets(self, idxs_list, offset_list):
        self.target_maker.sensor_idxs_list = idxs_list
        self.target_maker.time_offsets_list = offset_list

    def get_target_df(self):
        if self.denormalizer_used_in_training is not None:
            return self.denormalizer_used_in_training.denormalize(self.target_maker.dataset_object.df)
        return self.target_maker.dataset_object.df

    def print(self):
        print("INPUT MAKER : ")
        self.input_maker.print()
        print("TARGET MAKER")
        self.target_maker.print()
