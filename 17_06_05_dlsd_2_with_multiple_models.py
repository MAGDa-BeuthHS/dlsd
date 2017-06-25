from dlsd_2.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer
from dlsd_2.model.types.average_week.Average_Week import Average_Week
from dlsd_2.experiment.Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import *
import logging

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)

class Experiment_17_06_09_Redo_December_Experiment(Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output):
	def _define_source_maker(self):
		source_maker = Source_Maker()
		source_maker.file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
		source_maker.file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'
		source_maker.normalize = True
		source_maker.moving_average_window = 50
		source_maker.remove_inefficient_sensors_below_threshold = 1.0
		source_maker.time_format_train = '%Y-%m-%d %H:%M:%S'
		source_maker.time_format_test = '%Y-%m-%d %H:%M:%S'
		self.set_source_maker(source_maker)

	def _define_models(self):
		model_1 = NN_One_Hidden_Layer()
		model_1.name = "single_hidden_layer_50"
		model_1.set_number_hidden_nodes(50)
		model_1.set_learning_rate(.1)
		model_1.set_batch_size(3)
		model_1.set_max_steps(10)
		
		model_2 = Average_Week()
		model_2.set_average_data_from_csv_file_path('/Users/ahartens/Desktop/Average_Week_One_Year.csv')
		
		self.add_model(model_1)
		self.add_model(model_2)

	def _define_model_input_output_parameters(self):
		adjacency_path = '/Users/ahartens/Desktop/Work/AdjacencyMatrix_repaired.csv'
		adj_matrix = Adjacency_Matrix()
		adj_matrix.set_matrix_from_file_path(adjacency_path)
		
		io_1 = Model_Input_Output_Parameters()
		io_2 = Model_Input_Output_Parameters()
		io_3 = Model_Input_Output_Parameters()
		io_4 = Model_Input_Output_Parameters()
		io_5 = Model_Input_Output_Parameters()
		io_6 = Model_Input_Output_Parameters()
		io_7 = Model_Input_Output_Parameters()
		io_8 = Model_Input_Output_Parameters()

		all_ios = [io_1,io_2,io_3,io_4,io_5,io_6,io_7,io_8]

		io_1.name = "FFNN_single"
		io_2.name = "FFNN_nn"
		io_3.name = "FFNN_nn+"
		io_4.name = "FFNN_all"
		io_5.name = "mFFNN_single"
		io_6.name = "mFFNN_nn"
		io_7.name = "mFFNN_nn+"
		io_8.name = "mFFNN_all"

		io_1.use_single_sensor_as_input = True
		io_5.use_single_sensor_as_input = True

		io_2.adjacency_matrix = adj_matrix
		io_3.adjacency_matrix = adj_matrix
		io_6.adjacency_matrix = adj_matrix
		io_7.adjacency_matrix = adj_matrix

		io_2.include_output_sensor_in_adjacency = False
		io_6.include_output_sensor_in_adjacency = False

		target_time_offsets = [15,30,45,60]
		for io in all_ios:
			io.set_target_time_offsets_list(target_time_offsets)

		input_time_offsets_for_sequential_input = [0,10,15,30]
		for i in range(4,8):
		 	io = all_ios[i]
		 	io.set_input_time_offsets_list(input_time_offsets_for_sequential_input)

		self.set_input_output_parameters_list(all_ios)



def main():
	experiment_path = '/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_1'

	exp = Experiment_17_06_09_Redo_December_Experiment()
	exp.set_experiment_root_path(experiment_path)
	exp.run_experiment()


if __name__=="__main__":
	main()



