import unittest
import logging

from .input_target_maker.Input_Target_Maker_Testmaker import Input_Target_Maker_Testmaker
from .dataset.Dataset_Testmaker import Dataset_Testmaker
from .model.Model_Testmaker import Model_Testmaker

class Testmaker:
	def make_tests(self):
		logging.info('Making all tests')
		test_suite = unittest.TestSuite()
		self.add_all_tests_to_suite(test_suite)
		return test_suite

	def make_fast_tests(self):
		logging.info('Making Fast tests')
		test_suite = unittest.TestSuite()
		self.add_fast_tests_to_suite(test_suite)
		return test_suite
		
	def add_all_tests_to_suite(self,test_suite):
		#input_target_maker_testmaker = Input_Target_Maker_Testmaker()
		#input_target_maker_testmaker.add_all_tests_to_suite(test_suite)

		#dataset_testmaker = Dataset_Testmaker()
		#dataset_testmaker.add_all_tests_to_suite(test_suite)

		model_testmaker = Model_Testmaker()
		model_testmaker.add_all_tests_to_suite(test_suite)

	def add_fast_tests_to_suite(self,test_suite):
		input_target_maker_testmaker = Input_Target_Maker_Testmaker()
		input_target_maker_testmaker.add_fast_tests_to_suite(test_suite)

		dataset_testmaker = Dataset_Testmaker()
		dataset_testmaker.add_fast_tests_to_suite(test_suite)

		model_testmaker = Model_Testmaker()
		model_testmaker.add_fast_tests_to_suite(test_suite)