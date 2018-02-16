import logging

from dlsd_2.experiment.Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation import *

from dlsd_2.src.model.types.neural_networks.nn_one_hidden_layer import NN_One_Hidden_Layer

logging.basicConfig(level=logging.INFO)#filename='17_05_04_dlsd_2_trials.log',)

# PATH_TRAIN = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
# PATH_TEST = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'
# PATH_ADJACENCY = '/Users/ahartens/Desktop/Work/AdjacencyMatrix_repaired.csv'
# PATH_OUTPUT = '/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_4'

PATH_TRAIN = '/hartensa/data_sql/16_11_25_PZS_Belegung_augustFull.csv'
PATH_TEST = '/hartensa/data_sql/16_11_25_PZS_Belegung_September_Full.csv'
PATH_ADJACENCY = '/hartensa/data_other/Time_Adjacency_Matrix.csv'
PATH_OUTPUT = '/hartensa/experiment_output/ffnn_average_week'
PATH_AVERAGE_WEEK = '/hartensa/data_other/Average_Week_One_Year.csv'

class Experiment_17_06_09_Redo_December_Experiment(Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation):
	def _define_source_maker(self):
		source_maker = Source_Maker()
		source_maker.file_path_train = PATH_TRAIN
		source_maker.file_path_test = PATH_TEST
		source_maker.normalize = True
		source_maker.moving_average_window = 15
		source_maker.remove_inefficient_sensors_below_threshold = 1.0
		source_maker.time_format_train = '%Y-%m-%d %H:%M:%S'
		source_maker.time_format_test = '%Y-%m-%d %H:%M:%S'
		self.set_source_maker(source_maker)

	def _define_models(self):
		model = NN_One_Hidden_Layer()
		model.name = "ffnn_50_hidden_nodes"
		model.set_number_hidden_nodes(50)
		model.set_learning_rate(.1)
		model.set_batch_size(20)
		model.set_num_epochs(2)
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

		target_time_offsets = [10,15,30,45,60,75,90]
		input_time_offsets_for_sequential_input = list(range(0,1))
		for io in all_ios:
			io.set_target_time_offsets_list(target_time_offsets)
			io.set_input_time_offsets_list(input_time_offsets_for_sequential_input)
		self.set_input_output_parameters_list([io_1, io_4])

def main():
	exp = Experiment_17_06_09_Redo_December_Experiment()
	exp.set_experiment_root_path(PATH_OUTPUT)
	exp.run_experiment()

if __name__=="__main__":
	main()



