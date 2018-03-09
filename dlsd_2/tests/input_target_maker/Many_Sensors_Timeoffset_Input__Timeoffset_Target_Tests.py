from .Input_And_Target_Maker_Tests import *


class Many_Sensors_Timeoffset_Input__Timeoffset_Target_Tests(Input_And_Target_Maker_Tests):
    def setUp(self):
        self.correct_output_dir = 'dlsd_2/tests/input_target_maker/itm_test_csvs/correct_output/Many_Sensors_Timeoffset_Input__Timeoffset_Target/'
        super(Many_Sensors_Timeoffset_Input__Timeoffset_Target_Tests, self).setUp()


class Many_Sensor_Time_Offset_Timeoffset_Sensor_Target(Many_Sensors_Timeoffset_Input__Timeoffset_Target_Tests):
    def setUp(self):
        self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
        super(Many_Sensor_Time_Offset_Timeoffset_Sensor_Target, self).setUp()

    def define_correct_output_input_target_columns(self):
        self.correct_output_reader.define_input_columns_list([0, 1, 2, 3])
        self.correct_output_reader.define_target_columns_list([4, 5, 6, 7])

    def test_input_target_made_correctly_sensor_1_shift_10(self):
        self.set_correct_output_file_path('large_s_1_3_offset_0_5_input_s_1_3_offset_5_10_target.csv')
        self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0, 2], [0, 5])
        self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0, 2], [5, 10])
        self.make_input_and_target_and_check_if_correct()
