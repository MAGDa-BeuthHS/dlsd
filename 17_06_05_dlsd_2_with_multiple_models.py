from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors import ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors
from dlsd_2.input_target_maker.ITM_Normalized_Moving_Average import ITM_Normalized_Moving_Average
from dlsd_2.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer
from dlsd_2.model.types.average_week.Average_Week import Average_Week
from dlsd_2.experiment.Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import *
import logging

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)

class Experiment_17_06_09_Redo_December_Experiment(Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output):

	def _define_models(self):
		model_1 = NN_One_Hidden_Layer()
		model_1.set_number_hidden_nodes(50)
		model_1.set_learning_rate(.1)
		model_1.set_batch_size(3)
		model_1.set_max_steps(5000)
		self.add_model(model_1)

		model_2 = Average_Week()
		model_2.set_average_data_from_csv_file_path('/Users/ahartens/Desktop/Average_Week_One_Year.csv')
		self.add_model(model_2)

	def _define_model_input_output_parameters(self):
		io_1 = Model_Input_Output_Parameters()
		io_1.name = "all_sensor_in_one_sensor_out_(15_30_45)"
		io_1.set_target_time_offsets_list([15,30,45])
		self.add_input_output_parameter(io_1)

	def _define_input_and_target_makers(self):
		file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
		file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'
		
		train_input_and_target_maker = ITM_Normalized_Moving_Average_Remove_Inefficient_Sensors()
		train_input_and_target_maker.set_source_file_path(file_path_train)
		train_input_and_target_maker.set_moving_average_window(15)
		train_input_and_target_maker.set_efficiency_threshold(1.0) 
		train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')

		test_input_and_target_maker = ITM_Normalized_Moving_Average()
		test_input_and_target_maker.set_source_file_path(file_path_test)

		self.set_train_and_test_input_target_maker_and_create_source_data(train_input_and_target_maker, test_input_and_target_maker)

def main():
	exp = Experiment_17_06_09_Redo_December_Experiment()
	exp.set_experiment_root_path('/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_1')
	exp.run_experiment()



if __name__=="__main__":
	main()



