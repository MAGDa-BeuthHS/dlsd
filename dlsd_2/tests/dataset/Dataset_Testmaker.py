import unittest

from .dataset_helpers.Inefficient_Sensors_Remover_Tests import Inefficient_Sensors_Remover_Tests
from .dataset_helpers.Time_Gap_Filler_Tests import Time_Gap_Filler_Tests


class Dataset_Testmaker:

    def add_fast_tests_to_suite(self, test_suite):
        test_suite.addTest(unittest.makeSuite(Time_Gap_Filler_Tests))
        test_suite.addTest(unittest.makeSuite(Inefficient_Sensors_Remover_Tests))
