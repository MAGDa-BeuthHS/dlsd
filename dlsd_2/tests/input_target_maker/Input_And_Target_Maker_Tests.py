import logging
import unittest

import numpy as np

from dlsd_2.tests.input_target_maker.Input_Target_Correct_Output_Table import Correct_Output_Table


class Input_And_Target_Maker_Tests(unittest.TestCase):
    def setUp(self):
        self.input_target_maker = Input_And_Target_Maker()
        self.input_target_maker.source_file_path = self.source_file_path
        self.correct_output_reader = Correct_Output_Table()
        self.define_correct_output_input_target_columns()

    def define_correct_output_input_target_columns(self):
        raise NotImplementedError("Please Implement this method")

    def set_correct_output_file_path(self, correct_output_file_name):
        correct_output_file_path = self.correct_output_dir + correct_output_file_name
        self.correct_output_reader.read_csv_at_path(correct_output_file_path)

    def make_input_and_target_and_check_if_correct(self):
        self.input_target_maker.source_is_sql_output = False
        self.input_target_maker.make_source_data(index_col=0)
        self.input_target_maker.make_input_and_target()
        self.debug_print()
        self.assertTrue(np.allclose(self.input_target_maker.input_maker.dataset_object.df.values,
                                    self.correct_output_reader.input_numpy_array()))
        self.assertTrue(np.allclose(self.input_target_maker.target_maker.dataset_object.df.values,
                                    self.correct_output_reader.target_numpy_array()))

    def debug_print(self):
        logging.debug("Created INPUT")
        logging.debug(self.input_target_maker.input_maker.dataset_object.df.values[:, :])
        logging.debug("Created TARGET")
        logging.debug(self.input_target_maker.target_maker.dataset_object.df.values[:, :])
        # logging.debug("Stacked created input + target")
        # logging.debug(np.hstack([self.input_target_maker.input_dataset_object.df.values,self.input_target_maker.target_dataset_object.df.values])[0:5,:])
        logging.debug("CORRECT")
        logging.debug(self.correct_output_reader.df.tail())
