import logging
import unittest
from dlsd_2.tests.Testmaker import Testmaker

logging.basicConfig(level=logging.DEBUG)
testmaker = Testmaker()
test_suites = testmaker.make_fast_tests()
test_runner = unittest.TextTestRunner()
logging.info('Running all tests of DLSD_2')
test_runner.run(test_suites)