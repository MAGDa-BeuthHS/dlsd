# Alex Hartenstein 05.2017

How to use :
	1. Create a Model object of your choice. (set parameters of the model, eg hidden level number, if necessary)
	2. Create an Input_And_Target_Maker(ITM) for the training data. Set parameters of the training ITM. Three most important methods to call are :
		- ITM.set_source_file_path(file_path)
		- ITM.set_input_sensor_idxs_and_timeoffsets_lists([sensor_list],[time_offset_list])
		- ITM.set_target_sensor_idxs_and_timeoffsets_lists([sensor_list],[time_offset_list])
	3. Call Model.train_with_input_target_maker(ITM) on the model, passing it the ITM from (2)
	4. Create an ITM for the test data. It is only necessary to set the source file path (ITM.set_source_file_path()), as all other parameters are automatically copied from the training ITM. (opt. ITM.set_time_format if timestamp format differs from the training data)
	4. Call M.test_with_input_target_maker(ITM) on the model, passing it the ITM from (4)
	5. Call M.calc_prediction_accuracy() to get a dictionary of mae, mape, etc or M.write_target_and_predictions_to_file() on the model to get results



#TODOs

1. AVERAGE_WEEK: decide what to do with not starting at 00:00:00 timestamp : should probably just average everything then readjust to start monday at midnight
2. row/column names when printing
3. figure out denormalizer in neural network model. does the model input need it?
