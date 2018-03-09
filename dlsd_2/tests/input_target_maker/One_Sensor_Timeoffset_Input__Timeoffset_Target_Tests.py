from .Input_And_Target_Maker_Tests import *


class One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests(Input_And_Target_Maker_Tests):
    def setUp(self):
        self.correct_output_dir = 'dlsd_2/tests/input_target_maker/itm_test_csvs/correct_output/One_Sensor_Timeoffset_Input__Timeoffset_Target/'
        super(One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests, self).setUp()


class One_Sensor_Timeoffset_Input__Timeoffset_Target_Small_Tests(One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/small.csv'
        super(One_Sensor_Timeoffset_Input__Timeoffset_Target_Small_Tests, self).setUp()
        self.input_time_offsets_list = [0, 1]
        self.target_time_offsets_list = [1, 2]

    def define_correct_output_input_target_columns(self):
        self.correct_output_reader.define_input_columns_list([0, 1])
        self.correct_output_reader.define_target_columns_list([2, 3])

    def test_input_target_made_correctly_small_sensor_1(self):
        self.set_correct_output_file_path('small_sensor_1_input_0_1_target_1_2.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], self.input_time_offsets_list)
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], self.target_time_offsets_list)
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_small_sensor_2(self):
        self.set_correct_output_file_path('small_sensor_2_input_0_1_target_1_2.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([1], self.input_time_offsets_list)
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([1], self.target_time_offsets_list)
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_small_sensor_3(self):
        self.set_correct_output_file_path('small_sensor_3_input_0_1_target_1_2.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([2], self.input_time_offsets_list)
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([2], self.target_time_offsets_list)
        self.make_input_and_target_and_check_if_correct()


class One_Sensor_Timeoffset_Input__Timeoffset_Target_Medium_Tests(One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/medium.csv'
        super(One_Sensor_Timeoffset_Input__Timeoffset_Target_Medium_Tests, self).setUp()
        self.input_time_offsets_list = [0, 5]
        self.target_time_offsets_list = [1, 2, 3]

    def define_correct_output_input_target_columns(self):
        self.correct_output_reader.define_input_columns_list([0, 1])
        self.correct_output_reader.define_target_columns_list([2, 3, 4])

    def test_input_target_made_correctly_sensor_1_shift_10(self):
        self.set_correct_output_file_path('medium_sensor_1_input_0_5_target_1_2_3.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], self.input_time_offsets_list)
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], self.target_time_offsets_list)
        self.make_input_and_target_and_check_if_correct()


class One_Sensor_Timeoffset_Input__Timeoffset_Target_Large_Tests_Simple(
    One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
        super(One_Sensor_Timeoffset_Input__Timeoffset_Target_Large_Tests_Simple, self).setUp()
        self.input_time_offsets_list = [0, 1, 2]
        self.target_time_offsets_list = [1, 2, 3]

    def define_correct_output_input_target_columns(self):
        self.correct_output_reader.define_input_columns_list([0, 1, 2])
        self.correct_output_reader.define_target_columns_list([3, 4, 5])

    def test_input_target_made_correctly_sensor_1_shift_10(self):
        self.set_correct_output_file_path('large_sensor_1_input_1_2_3_target_1_2_3.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], self.input_time_offsets_list)
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], self.target_time_offsets_list)
        self.make_input_and_target_and_check_if_correct()


class One_Sensor_Timeoffset_Input__Timeoffset_Target_Large_Tests_Simple2(
    One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
        super(One_Sensor_Timeoffset_Input__Timeoffset_Target_Large_Tests_Simple2, self).setUp()
        self.input_time_offsets_list = [0, 5]
        self.target_time_offsets_list = [15, 25, 35]

    def define_correct_output_input_target_columns(self):
        self.correct_output_reader.define_input_columns_list([0, 1])
        self.correct_output_reader.define_target_columns_list([2, 3, 4])

    def test_input_target_made_correctly_sensor_1_shift_10(self):
        self.set_correct_output_file_path('large_sensor_1_input_0_5_target_15_25_35.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], self.input_time_offsets_list)
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], self.target_time_offsets_list)
        self.make_input_and_target_and_check_if_correct()
