from .Input_And_Target_Maker_Tests import *


class Many_Sensors_Input__One_Target_Tests(Input_And_Target_Maker_Tests):
	def setUp(self):
		self.correct_output_dir = 'dlsd_2/tests/input_target_maker/itm_test_csvs/correct_output/Many_Sensors_Input__One_Target/'
		super(Many_Sensors_Input__One_Target_Tests,self).setUp()

class Many_Sensor_Input__One_Target_Small_Test_All_Sensors(Many_Sensors_Input__One_Target_Tests):
	def setUp(self):
		self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/small.csv'
		super(Many_Sensor_Input__One_Target_Small_Test_All_Sensors,self).setUp()

	def define_correct_output_input_target_columns(self):
		self.correct_output_reader.define_input_columns_list([0,1,2])
		self.correct_output_reader.define_target_columns_list([3,4,5])

	def test_input_target_made_correctly_small_sensor_1(self):
		self.set_correct_output_file_path('small_sensors_0_1_2_target_offset_2.csv')
		self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0,1,2],[0])
		self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0,1,2],[2])
		self.make_input_and_target_and_check_if_correct()

class Many_Sensor_input__One_Target_Large_Tests(Many_Sensors_Input__One_Target_Tests):
	def setUp(self):
		self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
		super(Many_Sensor_input__One_Target_Large_Tests,self).setUp()

	def define_correct_output_input_target_columns(self):
		self.correct_output_reader.define_input_columns_list([0,1])
		self.correct_output_reader.define_target_columns_list([2,3])

	def test_input_target_made_correctly_sensor_1_shift_10(self):
		self.set_correct_output_file_path('large_sensor__input_s_1_2_no_offset__target_s_1_2_offset_20.csv')
		self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0,1],[0])
		self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0,1],[20])
		self.make_input_and_target_and_check_if_correct()

class Many_Sensor_Time_Offset_One_Target(Many_Sensors_Input__One_Target_Tests):
	def setUp(self):
		self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
		super(Many_Sensor_input__One_Target_Large_Tests,self).setUp()

	def define_correct_output_input_target_columns(self):
		self.correct_output_reader.define_input_columns_list([0,1])
		self.correct_output_reader.define_target_columns_list([2,3])

	def test_input_target_made_correctly_sensor_1_shift_10(self):
		self.set_correct_output_file_path('large_sensor__input_s_1_2_no_offset__target_s_1_2_offset_20.csv')
		self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0,1],[0])
		self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0,1],[20])
		self.make_input_and_target_and_check_if_correct()


class Many_Sensor_Time_Offset_One_Sensor_Target(Many_Sensors_Input__One_Target_Tests):
	def setUp(self):
		self.source_file_path = 'dlsd_2/tests/input_target_maker/itm_test_csvs/large.csv'
		super(Many_Sensor_Time_Offset_One_Sensor_Target,self).setUp()

	def define_correct_output_input_target_columns(self):
		self.correct_output_reader.define_input_columns_list([0,1,2])
		self.correct_output_reader.define_target_columns_list([3])

	def test_input_target_made_correctly_sensor_1_shift_10(self):
		self.set_correct_output_file_path('large_s_1_2_3_nooffset_input_s_1_offset_20_target.csv')
		self.input_target_maker.set_input_sensor_idxs_and_time_offsets([0,1,2],[0])
		self.input_target_maker.set_target_sensor_idxs_and_time_offsets([0],[20])
		self.make_input_and_target_and_check_if_correct()