import os


class Many_Sensor_Directory_Helper:
    def __init__(self):
        self.root_experiment_directory = None
        self.all_sensor_dirs = None
        self.current_sensor_dir = None

    def set_root_experiment_directory(self, path):
        self.root_experiment_directory = path

    def prepare_list_of_all_sensors_to_analyze(self):
        all_subdirs = next(os.walk(self.root_experiment_directory))[1]
        self.all_sensor_dirs = [i for i in all_subdirs if "error" not in i]

    def set_current_sensor_directory_with_sensor(self, sensor):
        self.current_sensor_dir = os.path.join(self.root_experiment_directory, str(sensor))

    def make_current_error_directory(self):
        path = self.get_current_error_directory()
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_current_error_directory(self):
        return os.path.join(self.current_sensor_dir, "error")

    def _check_or_make_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name


class Many_Sensor_Directory_Helper_With_K_Fold_Validation(Many_Sensor_Directory_Helper):
    def prepare_list_of_level_0_dirs(self):
        first_sensor = self.all_sensor_dirs[0]
        path_first_sensor = os.path.join(self.root_experiment_directory, first_sensor)
        subdirs_in_sensor_dir = next(os.walk(path_first_sensor))[1]
        self.level_0_dirs = [i for i in subdirs_in_sensor_dir if "error" not in i]

    def set_current_level_0_directory_with_level_0(self, level_0_dir):
        self.current_level_0_dir = os.path.join(self.current_sensor_dir, level_0_dir)

    def get_current_error_directory(self):
        return os.path.join(self.current_level_0_dir, "error")
