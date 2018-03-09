from dlsd_2.src.experiment.experiment_output_reader import Experiment_Output_Reader
from dlsd_2.src.experiment.experiment_output_reader import Many_Sensor_Directory_Helper
from dlsd_2.src.experiment.experiment_output_reader.Experiment_Error_Calculator import Experiment_Error_Calculator
from dlsd_2.src.experiment.experiment_output_reader.Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output import \
    Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output


class Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output_With_K_Fold_Validation(
    Experiment_Error_Calculator_For_Iterate_Over_All_Sensors_Using_One_Sensor_As_Output):
    def __init__(self):
        self.directory_helper = Many_Sensor_Directory_Helper()
        self.reader = Experiment_Output_Reader()
        self.list_of_functions = None

    def set_root_experiment_directory(self, path):
        self.directory_helper.set_root_experiment_directory(path)

    def set_analysis_functions(self, list_of_functions):
        self.analysis_functions = list_of_functions

    def analyze_all_sensors(self):
        self.directory_helper.prepare_list_of_all_sensors_to_analyze()
        self.directory_helper.prepare_list_of_level_0_dirs()
        self._iterate_over_all_sensors_calculating_k_fold_error()
        self._iterate_over_all_sensors_calculating_avg_error()

    def _iterate_over_all_sensors_calculating_k_fold_error(self):
        for sensor in self.directory_helper.all_sensor_dirs:
            for level_0_dir in self.directory_helper.level_0_dirs:
                self.reader = Experiment_Output_Reader()
                self._extract_data_from_level_0_dir_in_current_sensor(level_0_dir, sensor)
                self._iterate_over_error_functions()

    def _extract_data_from_level_0_dir_in_current_sensor(self, level_0_dir, sensor):
        self.directory_helper.set_current_sensor_directory_with_sensor(sensor)
        self.directory_helper.set_current_level_0_directory_with_level_0(level_0_dir)
        self.reader.set_experiment_output_directory(self.directory_helper.current_level_0_dir)
        self.reader.extract_data()

    def _do_analysis_with_error_function(self, error_function):
        analyzer = Experiment_Error_Calculator()
        analyzer.set_experiment_output_reader(self.reader)
        analyzer.set_output_directory_path(self.directory_helper.make_current_error_directory())
        analyzer.set_analysis_function(error_function)
        analyzer.do_analysis()

    def _iterate_over_all_sensors_calculating_avg_error(self):
        for sensor in self.directory_helper.all_sensor_dirs:
            self.directory_helper.set_current_sensor_directory_with_sensor(sensor)
            from dlsd_2.src.experiment.experiment_output_reader.Experiment_Average_Error_Calculator import \
                Experiment_Average_Error_Calculator
            avg = Experiment_Average_Error_Calculator()
            avg.set_root_experiment_directory(self.directory_helper.current_sensor_dir)
            avg.calculate_average()
