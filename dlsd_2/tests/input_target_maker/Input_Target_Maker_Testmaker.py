from .Many_Sensors_Input__One_Target_Tests import *
from .Many_Sensors_Input__Timeoffset_Target_Tests import *
from .Many_Sensors_Timeoffset_Input__Timeoffset_Target_Tests import *
from .One_Sensor_Input__One_Target_Tests import *
from .One_Sensor_Input__Timeoffset_Target_Tests import *
from .One_Sensor_Timeoffset_Input__Timeoffset_Target_Tests import *

'''
	If add a new Model_Input class
	1. create new ##New_Model_Input##_tests.py file with tests
	2. add tests to add_all_tests_to_suite 
	Done

	There are 6 cases
	# 		Input 					Target
	# 1		one						single 			x
	# 2		one						time_offset 	x
	# 3		one 	time_offset 	single 			
	# 4		one 	time_offset 	time_offset 	x
	# 		many 					single
	# 5		many 					time_offset 	
	# 		many  	time_offset 	single 			
	# 6 	many  	time_offset 	time_offset 	

'''


class Input_Target_Maker_Testmaker:

    def make_tests(self):
        logging.info('Making all INPUT_TARGET tests')
        self.test_suite = unittest.TestSuite()
        self.add_all_tests_to_suite()
        return self.test_suite

    def add_fast_tests_to_suite(self, test_suite):
        logging.info('Adding all INPUT_TARGET tests to test suite')
        self.test_suite = test_suite

        self._1_one_sensor_INPUT__one_TARGET_tests()
        self._2_one_sensor_INPUT__timeoffset_TARGET_tests()
        # self._3_one_sensor_timeoffset_INPUT__one_TARGET__tests()
        self._4_one_sensor_timeoffset_INPUT__timeoffset_TARGET__tests()
        self._5_many_sensors_INPUT__one_TARGET__tests()
        self._6_many_sensors_timeoffset_INPUT__timeoffset_TARGET__tests()

    def _1_one_sensor_INPUT__one_TARGET_tests(self):
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__One_Target_Small_Tests))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__One_Target_Large_Tests))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__One_Target_Large_Test_Offsets))

    def _2_one_sensor_INPUT__timeoffset_TARGET_tests(self):
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__Timeoffset_Target_Small_Tests))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__Timeoffset_Target_Large_Tests))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__One_Target_Large_Test_Offsets))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Input__Timeoffset_Target_Large_Tests_Simple))

    #def _3_one_sensor_timeoffset_INPUT__one_TARGET__tests(self):
        #self.test_suite.addTest(unittest.makeSuite(One_Sensor_In_One_Out_No_Time_Offset))

    def _4_one_sensor_timeoffset_INPUT__timeoffset_TARGET__tests(self):
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Timeoffset_Input__Timeoffset_Target_Small_Tests))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Timeoffset_Input__Timeoffset_Target_Medium_Tests))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Timeoffset_Input__Timeoffset_Target_Large_Tests_Simple))
        self.test_suite.addTest(unittest.makeSuite(One_Sensor_Timeoffset_Input__Timeoffset_Target_Large_Tests_Simple2))

    def _5_many_sensors_INPUT__one_TARGET__tests(self):
        self.test_suite.addTest(unittest.makeSuite(Many_Sensor_Input__One_Target_Small_Test_All_Sensors))
        self.test_suite.addTest(unittest.makeSuite(Many_Sensor_input__One_Target_Large_Tests))
        self.test_suite.addTest(unittest.makeSuite(Many_Sensor_Time_Offset_One_Sensor_Target))
        self.test_suite.addTest(unittest.makeSuite(Many_Sensor_Time_Offset_Timeoffset_Sensor_Target))

    def _6_many_sensors_timeoffset_INPUT__timeoffset_TARGET__tests(self):
        self.test_suite.addTest(unittest.makeSuite(Many_Sensor_Time_Offset_Timeoffset_Sensor_Target))
