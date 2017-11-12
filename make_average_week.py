# Alex Hartenstein 2017

from dlsd_2.input_target_maker.Source_Maker_With_K_Fold_Validation import *
from dlsd_2.model.types.average_week.Average_Week import Average_Week

#PATH_DATA = '/hartensa/Repair/all_fixed.csv'
#PATH_AVERAGE_WEEK = '/hartensa/data_other/Average_Week_One_Year_Fixed.csv'
PATH_AVERAGE_WEEK = '/Users/ahartens/Desktop/Average_Week_One_Year_Fixed.csv'
PATH_DATA = '/Users/ahartens/Desktop/all_fixed.csv'

def main():
	print("starting")
	source_maker = create_source_maker()
	create_average_week_with_source_maker(source_maker)
	print("done")

def create_source_maker():
	source_maker = Source_Maker_With_K_Fold_Validation()
	source_maker.file_path_all_data = PATH_DATA
	source_maker.normalize = True
	source_maker.moving_average_window = 3
	source_maker.remove_inefficient_sensors_below_threshold = 1.0
	source_maker.time_format_train = '%Y-%m-%d %H:%M:%S'
	source_maker.time_format_test = '%Y-%m-%d %H:%M:%S'
	source_maker.prepare_source_data()
	return source_maker

def create_average_week_with_source_maker(source_maker):
	model = Average_Week()
	model.create_average_week_with_source_maker(source_maker)
	model.write_average_week_to_filepath(PATH_AVERAGE_WEEK)

if __name__ == "__main__":
	main()