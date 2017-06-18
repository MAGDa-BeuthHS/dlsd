
import unittest

from .types.Types_Testmaker import Types_Testmaker

class Model_Testmaker:
	def add_all_tests_to_suite(self,test_suite):
		types_testmaker = Types_Testmaker()
		types_testmaker.add_all_tests_to_suite(test_suite)

	def add_fast_tests_to_suite(self,test_suite):
		types_testmaker = Types_Testmaker()
		types_testmaker.add_fast_tests_to_suite(test_suite)

	def add_slow_tests_to_suite(self,test_suite):
		types_testmaker = Types_Testmaker()
		types_testmaker.add_slow_tests_to_suite(test_suite)
