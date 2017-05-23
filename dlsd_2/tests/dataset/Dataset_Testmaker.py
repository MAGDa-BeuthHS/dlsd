import unittest
import logging

from .dataset_helpers.Time_Gap_Filler_Tests import Time_Gap_Filler_Tests

class Dataset_Testmaker:
	
	def make_tests(self):
		logging.info('Making Dataset Module tests')
		test_suite = unittest.TestSuite()
		self.add_all_tests_to_suite(test_suite)
		return test_suite 

	def add_fast_tests_to_suite(self,test_suite):
		test_suite.addTest(unittest.makeSuite(Time_Gap_Filler_Tests))
