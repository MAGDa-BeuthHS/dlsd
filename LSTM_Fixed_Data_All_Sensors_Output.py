from dlsd_2.model.types.neural_networks.LSTM.LSTM_One_Hidden_Layer import LSTM_One_Hidden_Layer

from dlsd_2.input_target_maker.Source_Maker_With_K_Fold_Validation import *
from dlsd_2.model.types.average_week.Average_Week import Average_Week
from dlsd_2.experiment.Experiment_With_K_Fold_Validation import *
import logging

logging.basicConfig(level=logging.INFO)#filename='17_05_04_dlsd_2_trials.log',)

PATH_DATA = '/hartensa/Repair/all_fixed.csv'
#PATH_DATA = '/hartensa/Repair/first_500_fixed.csv'
PATH_ADJACENCY = '/hartensa/data_other/Time_Adjacency_Matrix.csv'
PATH_OUTPUT = '/hartensa/experiment_output/lstm_experiment_all_sensors_as_output_fixed_data'
PATH_AVERAGE_WEEK = '/hartensa/data_other/Average_Week_One_Year_Fixed.csv'


def main():
	exp = LSTM_Fixed_Data()
	exp.k = 5
	exp.validation_percentage = 10
	exp.set_experiment_root_path(PATH_OUTPUT)
	exp.run_experiment()
	#exp._calculate_accuracy_of_models()


class LSTM_Fixed_Data(Experiment_With_K_Fold_Validation):
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
		# model = Average_Week()
		# model.name = "Average_Week"
		# model.set_average_data_from_csv_file_path(PATH_AVERAGE_WEEK)
		# self.add_model(model)

		model = LSTM_One_Hidden_Layer()
		model.name = "lstm_100"
		model.set_number_hidden_nodes(100)
		model.set_learning_rate(.08)
		model.set_batch_size(256)
		model.set_num_epochs(20)
		model.fill_output_timegaps = False
		self.add_model(model)

	def _define_model_input_output_parameters(self):		
		io = Model_Input_Output_Parameters()
		io.set_input_time_offsets_list(list(range(0,6)))
		io.set_target_time_offsets_list([2,3,6,9,12,15,18])
		self.set_input_output_parameters_list([io])

if __name__=="__main__":
	main()



