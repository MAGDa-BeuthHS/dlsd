import unittest
from .average_week.Average_Week_Content_Tests import *
from .average_week.Average_Week_Tests import *

class Types_Testmaker:
	def add_all_tests_to_suite(self,test_suite):
		self.add_fast_tests_to_suite()
		self.add_slow_tests_to_suite()

	def add_fast_tests_to_suite(self,test_suite):
		test_suite.addTest(unittest.makeSuite(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset))
		test_suite.addTest(unittest.makeSuite(AWC_Test_Opening_Preaveraged_Week_And_Creating_Target_Dataset_Multiple_Sensors))
		test_suite.addTest(unittest.makeSuite(AWT_1_Source_Data_Prediction_Data_Generated_Correctly_Source_Week_Starting_On_Monday))
		test_suite.addTest(unittest.makeSuite(AWT_2_Source_Data_Starts_Thursday_Average_Week_Starts_Monday_Then_Prediction_Matches_Target))
		test_suite.addTest(unittest.makeSuite(AWT_3_Same_As_1_But_With_Time_Gaps))

	def add_slow_tests_to_suite(self,test_suite):
		test_suite.addTest(unittest.makeSuite(AWC_Test_Calc_Average_Week))
		test_suite.addTest(unittest.makeSuite(Average_Week_Tests))
