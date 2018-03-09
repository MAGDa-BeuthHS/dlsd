import logging

from dlsd_2.src.experiment.Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import \
    Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output
from dlsd_2.src.io.input_target_maker.Input_Target_Maker import Input_And_Target_Maker_2
from dlsd_2.src.model.types.neural_networks.nn_one_hidden_layer.NN_One_Hidden_Layer import NN_One_Hidden_Layer

logging.basicConfig(level=logging.DEBUG)  # filename='17_05_04_dlsd_2_trials.log',)


class Experiment_17_06_09_Redo_December_Experiment(
    Experiment_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output):

    def _define_models(self):
        model_1 = NN_One_Hidden_Layer()
        model_1.set_number_hidden_nodes(50)
        model_1.set_learning_rate(.1)
        model_1.set_batch_size(3)
        model_1.set_max_steps(3)
        self.models = [model_1]

    def _define_input_and_target_makers(self):
        file_path_train = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_augustFull.csv'
        file_path_test = '/Users/ahartens/Desktop/Work/16_11_25_PZS_Belegung_September_Full.csv'

        self.train_input_and_target_maker = Input_And_Target_Maker_2()
        self.train_input_and_target_maker.set_source_file_path(file_path_train)
        self.train_input_and_target_maker.set_moving_average_window(15)
        self.train_input_and_target_maker.set_efficiency_threshold(1.0)
        self.train_input_and_target_maker.make_source_data()
        self.train_input_and_target_maker.set_time_format('%Y-%m-%d %H:%M:%S')

        self.test_input_and_target_maker = Input_And_Target_Maker_2()
        self.test_input_and_target_maker.set_source_file_path(file_path_test)
        self.test_input_and_target_maker.copy_parameters_from_maker(self.train_input_and_target_maker)
        self.test_input_and_target_maker.make_source_data()


def main():
    exp = Experiment_17_06_09_Redo_December_Experiment()
    exp.set_experiment_root_path('/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_1')
    exp.run_experiment()


if __name__ == "__main__":
    main()
