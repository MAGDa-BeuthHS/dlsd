from .Input_And_Target_Maker_Tests import *


class One_Sensor_Input__One_Target_Tests(Input_And_Target_Maker_Tests):
    def setUp(self):
        self.correct_output_dir = 'dlsd_2/tests/input_target_maker/itm_test_csvs/correct_output/One_Sensor_Input__One_Target/'
        super(One_Sensor_Input__One_Target_Tests, self).setUp()

    def define_correct_output_input_target_columns(self):
        self.correct_output_reader.define_input_columns_list([0])
        self.correct_output_reader.define_target_columns_list([1])


class One_Sensor_Input__One_Target_Small_Tests(One_Sensor_Input__One_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/small.csv'
        super(One_Sensor_Input__One_Target_Small_Tests, self).setUp()

    def test_input_target_made_correctly_small_sensor_1(self):
        self.set_correct_output_file_path('sensor_1_offset_2.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], [2])
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_small_sensor_2(self):
        self.set_correct_output_file_path('sensor_2_offset_2.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([1], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([1], [2])
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_small_sensor_3(self):
        self.set_correct_output_file_path('sensor_3_offset_2.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([2], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([2], [2])
        self.make_input_and_target_and_check_if_correct()


class One_Sensor_Input__One_Target_Large_Tests(One_Sensor_Input__One_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
        super(One_Sensor_Input__One_Target_Large_Tests, self).setUp()

    def test_input_target_made_correctly_sensor_1_shift_10(self):
        self.set_correct_output_file_path('large_sensor_1_offset_10.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], [10])
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_sensor_2_shift_10(self):
        self.set_correct_output_file_path('large_sensor_2_offset_10.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([1], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([1], [10])
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_sensor_3_shift_10(self):
        self.set_correct_output_file_path('large_sensor_3_offset_10.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([2], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([2], [10])
        self.make_input_and_target_and_check_if_correct()


class One_Sensor_Input__One_Target_Large_Test_Offsets(One_Sensor_Input__One_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
        super(One_Sensor_Input__One_Target_Large_Test_Offsets, self).setUp()

    def test_input_target_made_correctly_sensor_3_shift_40(self):
        self.set_correct_output_file_path('large_sensor_1_offset_40.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], [40])
        self.make_input_and_target_and_check_if_correct()

    def test_input_target_made_correctly_sensor_3_shift_5(self):
        self.set_correct_output_file_path('large_sensor_1_offset_5.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0], [0])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0], [5])
        self.make_input_and_target_and_check_if_correct()
