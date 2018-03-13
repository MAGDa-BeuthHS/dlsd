from dlsd_2.src.io.input_target_maker.Model_Input_Output_Parameters import Model_Input_Output_Parameters
from dlsd_2.src.model.types.neural_networks.LSTM.LSTM_One_Hidden_Layer import LSTM_One_Hidden_Layer
from dlsd_2.src.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer
from dlsd_2.src.io.input_target_maker.Source_Maker_With_K_Fold_Validation import *
from dlsd_2.src.model.types.average_week.Average_Week import Average_Week
from dlsd_2.src.experiment.Experiment_With_K_Fold_Validation import *
from dlsd_2.src.api.Source_Maker_For_API import *
from dlsd_2.src.api.Pretrained_Model_Tester import *
import logging

"""
	Call test_model_with_data_at_path for api
"""

def test_model_with_data_at_path(path):
	tester = Test_LSTM_For_API()
	tester.path_data_to_test = path
	tester.set_experiment_root_path('/alex/experiment_output/lstm_for_api')
	return tester.test_model()

'''
	The following is for training and testing the model. 
	Api calls (above) use the saved tensorflow seission created below. 
	If changes are made, make sure hardcoded paths above reflect those below
'''


logging.basicConfig(level=logging.INFO)
PATH_TEST_DATA = '/alex/scratch_target.csv'#all_fixed.csv'
PATH_DATA = '/alex/Repair/all_fixed.csv'
PATH_ADJACENCY = '/alex/data_other/Time_Adjacency_Matrix.csv'
PATH_OUTPUT = '/alex/experiment_output/lstm_for_api'
PATH_SERIALIZED_DENORMALIZER_USED_IN_TRAINING = os.path.join(PATH_OUTPUT,'denormalizer_used_in_training.pickle')
PATH_DESIRED_COLUMNS = os.path.join(PATH_OUTPUT,'path_desired_columns.csv')

def train_model():
	exp = LSTM()
	exp.set_experiment_root_path(PATH_OUTPUT)
	exp.run_experiment()
	save_necessary_data_to_restore(exp)

def test_saved_model(path):
	tester = Test_LSTM()
	tester.set_experiment_root_path(PATH_OUTPUT)
	return tester.test_model()

def save_necessary_data_to_restore(exp):
	with open(PATH_SERIALIZED_DENORMALIZER_USED_IN_TRAINING, 'wb') as f:
		pickle.dump(exp.models[0].train_input_target_maker.denormalizer_used_in_training,f)
	with open(PATH_DESIRED_COLUMNS, 'wb') as f:
		columns = exp.train_input_and_target_maker.source_dataset_object.df.columns.values
		pickle.dump(columns,f)

class LSTM(Experiment):
	def _define_source_maker(self):
		source_maker = Source_Maker()
		source_maker.file_path_train = PATH_DATA
		source_maker.file_path_test = PATH_TEST_DATA
		set_parameters_for_source_maker(source_maker)
		self.set_source_maker(source_maker)

	def _define_models(self):
		self.add_model(model())

	def _define_model_input_output_parameters(self):
		self.set_input_output_parameters_list([io_param()])

class Test_LSTM(Pretrained_Model_Tester):
	def _define_source_maker(self):
		source_maker = Source_Maker_For_API()
		source_maker.file_path_train = PATH_TEST_DATA
		source_maker.path_serialized_denormalizer_used_in_training = PATH_SERIALIZED_DENORMALIZER_USED_IN_TRAINING
		source_maker.path_desired_columns = PATH_DESIRED_COLUMNS
		set_parameters_for_source_maker(source_maker)
		self.set_source_maker(source_maker)

	def _define_models(self):
		self.add_model(model())

	def _define_model_input_output_parameters(self):
		self.set_input_output_parameters_list([io_param()])

class Test_LSTM_For_API(Test_LSTM):
	def __init__(self):
		super(Test_LSTM_For_API, self).__init__()
		self.path_data_to_test = None

	def _define_source_maker(self):
		path_output = '/alex/experiment_output/lstm_for_api'
		source_maker = Source_Maker_For_API()
		source_maker.file_path_train = self.path_data_to_test
		source_maker.path_serialized_denormalizer_used_in_training = os.path.join(path_output,'denormalizer_used_in_training.pickle')
		source_maker.path_desired_columns = os.path.join(path_output,'path_desired_columns.csv')
		set_parameters_for_source_maker(source_maker)
		self.set_source_maker(source_maker)



def model():
	m = LSTM_One_Hidden_Layer()
	m.name = "lstm_model"
	m.set_number_hidden_nodes(50)
	m.set_learning_rate(.01)
	m.set_batch_size(256)
	m.set_num_epochs(30)
	m.fill_output_timegaps = False
	return m

def io_param():
	io = Model_Input_Output_Parameters()
	io.name = "all_in_all_out"
	io.set_target_time_offsets_list([2,3,6,9,12,15,18])
	io.set_input_time_offsets_list(list(range(0,7)))
	return io

def set_parameters_for_source_maker(source_maker):
	source_maker.normalize = True
	source_maker.moving_average_window = 3
	source_maker.remove_inefficient_sensors_below_threshold = 1.0
	source_maker.time_format_train = '%Y-%m-%d %H:%M:%S'
	source_maker.time_format_test = '%Y-%m-%d %H:%M:%S'
	source_maker.is_sql_output = False


if __name__=="__main__":
	test_model_with_data_at_path(PATH_TEST_DATA)



