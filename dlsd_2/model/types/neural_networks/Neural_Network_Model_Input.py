from dlsd_2.model.Model_Input import *
import numpy as np

class Neural_Network_Model_Input(Model_Input):		

	def __init__(self):
		super(Neural_Network_Model_Input, self).__init__()
		logging.debug("\tAdding Neural_Network_Model_Input")

	def fill_feed_dict(self, input_placeholder, target_placeholder, batch_size):
		indices = np.random.choice(self.get_number_datapoints(),batch_size,replace = False)
		input_batch = self.input_dataset_object.get_numpy_rows_at_idxs(indices)
		target_batch = self.target_dataset_object.get_numpy_rows_at_idxs(indices)
		feed_dict = {
			input_placeholder:input_batch, 
			target_placeholder:target_batch
			}
		return feed_dict
