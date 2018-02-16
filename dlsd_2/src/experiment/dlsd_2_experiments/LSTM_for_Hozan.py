import logging

from dlsd_2.experiment.Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation import *
from dlsd_2.input_target_maker.Source_Maker_With_K_Fold_Validation import *
from dlsd_2.model.types.neural_networks.LSTM.LSTM_One_Hidden_Layer import LSTM_One_Hidden_Layer

logging.basicConfig(level=logging.INFO)#filename='17_05_04_dlsd_2_trials.log',)

PATH_DATA = '/hartensa/data_other/hozan_S1186_for_model.csv'
PATH_OUTPUT = '/hartensa/experiment_output/hozan_experiment_2'
TARGET_TIME_OFFSETS = [1,2,3,6]
INPUT_TIME_OFFSETS = [0,3,4,5]


def main():
	exp = LSTM_Fixed_Data()
	exp.k = 5
	exp.validation_percentage = 10
	exp.set_experiment_root_path(PATH_OUTPUT)
	#exp._calculate_accuracy_of_models()
	exp.run_experiment()

class LSTM_Fixed_Data(Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation):
	def _define_source_maker(self):
		source_maker = Source_Maker_With_K_Fold_Validation()
		source_maker.file_path_all_data = PATH_DATA
		source_maker.normalize = True
		source_maker.moving_average_window = 3
		source_maker.remove_inefficient_sensors_below_threshold = 1.0
		source_maker.time_format_train = '%Y-%m-%d %H:%M:%S'
		source_maker.time_format_test = '%Y-%m-%d %H:%M:%S'
		self.set_source_maker(source_maker)

	def _define_models(self):
		model = LSTM_One_Hidden_Layer()
		model.name = "lstm"
		model.set_number_hidden_nodes(50)
		model.set_learning_rate(.1)
		model.set_batch_size(20)
		model.set_num_epochs(50)
		model.fill_output_timegaps = False
		self.add_model(model)

	def _define_model_input_output_parameters(self):
		io = Model_Input_Output_Parameters()
		io.name = "LSTM"
		io.set_target_time_offsets_list(TARGET_TIME_OFFSETS)
		io.set_input_time_offsets_list(INPUT_TIME_OFFSETS)
		self.set_input_output_parameters_list([io])

if __name__=="__main__":
	main()



