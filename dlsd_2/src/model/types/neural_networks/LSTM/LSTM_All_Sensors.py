from dlsd_2.src.experiment.Experiment_With_K_Fold_Validation import *
from dlsd_2.src.io.input_target_maker.Model_Input_Output_Parameters import *
from dlsd_2.src.io.input_target_maker.Source_Maker_With_K_Fold_Validation import *
from dlsd_2.src.model.types.average_week.Average_Week import Average_Week
from dlsd_2.src.model.types.neural_networks.LSTM.LSTM_One_Hidden_Layer import LSTM_One_Hidden_Layer
from dlsd_2.src.model.types.neural_networks.nn_one_hidden_layer import NN_One_Hidden_Layer

logging.basicConfig(level=logging.INFO)  # filename='17_05_04_dlsd_2_trials.log',)

PATH_DATA = '/alex/Repair/all_fixed.csv'
PATH_ADJACENCY = '/alex/data_other/Time_Adjacency_Matrix.csv'
PATH_OUTPUT = '/alex/experiment_output/lstm_avg_week_and_ffnn_with_all_sensors'
PATH_AVERAGE_WEEK = '/alex/data_other/Average_Week_One_Year_Fixed.csv'


def main():
    exp = LSTM_Fixed_Data()
    exp.k = 5
    exp.validation_percentage = 10
    exp.set_experiment_root_path(PATH_OUTPUT)
    exp.run_experiment()


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
        model = Average_Week()
        model.name = "Average_Week"
        model.set_average_data_from_csv_file_path(PATH_AVERAGE_WEEK)
        self.add_model(model)

        model = LSTM_One_Hidden_Layer()
        model.name = "lstm_model"
        model.set_number_hidden_nodes(50)
        model.set_learning_rate(.01)
        model.set_batch_size(256)
        model.set_num_epochs(30)
        model.fill_output_timegaps = False
        self.add_model(model)

        model = NN_One_Hidden_Layer()
        model.name = "ffnn_model"
        model.set_number_hidden_nodes(50)
        model.set_learning_rate(.01)
        model.set_batch_size(256)
        model.set_num_epochs(30)
        model.fill_output_timegaps = False
        self.add_model(model)

    def _define_model_input_output_parameters(self):
        io = Model_Input_Output_Parameters()
        io.name = "all_in_all_out"
        io.set_target_time_offsets_list([2, 3, 6, 9, 12, 15, 18])
        io.set_input_time_offsets_list(list(range(0, 7)))
        self.set_input_output_parameters_list([io])


if __name__ == "__main__":
    main()
