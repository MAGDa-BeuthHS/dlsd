from dlsd_2.model.types.neural_networks.LSTM.LSTM_One_Hidden_Layer import LSTM_One_Hidden_Layer

from dlsd_2.input_target_maker.Source_Maker_With_K_Fold_Validation import *
from dlsd_2.model.types.average_week.Average_Week import Average_Week
from dlsd_2.experiment.Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation import *
import logging

logging.basicConfig(level=logging.INFO)#filename='17_05_04_dlsd_2_trials.log',)

PATH_DATA = '/hartensa/Repair/all_fixed.csv'
PATH_ADJACENCY = '/hartensa/data_other/Time_Adjacency_Matrix.csv'
PATH_OUTPUT = '/hartensa/experiment_output/lstm_experiment'
PATH_AVERAGE_WEEK = '/hartensa/data_other/Average_Week_One_Year_Fixed.csv'


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
		model = Average_Week()
		model.name = "Average_Week"
		model.set_average_data_from_csv_file_path(PATH_AVERAGE_WEEK)
		self.add_model(model)

		model = LSTM_One_Hidden_Layer()
		model.name = "lstm_one_hidden_layer_content_with_time_adjacency_matrix_17_09_28_(0,31,1)"
		model.set_number_hidden_nodes(50)
		model.set_learning_rate(.1)
		model.set_batch_size(20)
		model.set_num_epochs(1)
		model.fill_output_timegaps = False
		self.add_model(model)

	def _define_model_input_output_parameters(self):
		adjacency_path = PATH_ADJACENCY
		adj_matrix = Adjacency_Matrix()
		adj_matrix.set_matrix_from_file_path(adjacency_path)
		
		io_1 = Model_Input_Output_Parameters()
		io_2 = Model_Input_Output_Parameters()
		io_3 = Model_Input_Output_Parameters()
		io_4 = Model_Input_Output_Parameters()

		all_ios = [io_1,io_2,io_3,io_4]

		io_1.name = "LSTM_single"
		io_2.name = "LSTM_nn"
		io_3.name = "LSTM_nn+"
		io_4.name = "LSTM_all"

		io_1.use_single_sensor_as_input = True

		io_2.adjacency_matrix = adj_matrix
		io_3.adjacency_matrix = adj_matrix

		io_2.include_output_sensor_in_adjacency = False

		target_time_offsets = [2,3,6,9,12,15,18]
		input_time_offsets_for_sequential_input = list(range(0,7))
		for io in all_ios:
			io.set_target_time_offsets_list(target_time_offsets)
			io.set_input_time_offsets_list(input_time_offsets_for_sequential_input)
		self.set_input_output_parameters_list([io_4])

if __name__=="__main__":
	main()



