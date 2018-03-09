from dlsd_2.src.model.types.neural_networks.Neural_Network_Model_Input import *


class LSTM_One_Hidden_Layer_Input(Neural_Network_Model_Input):

    def __init__(self):
        super(LSTM_One_Hidden_Layer_Input, self).__init__()
        logging.debug("\tAdding Neural_Network_Model_Input")

    def _fill_feed_dict_with_indices(self, indices, input_placeholder, target_placeholder, batch_size):
        input_batch = self.input_dataset_object.get_numpy_rows_at_idxs(indices).reshape(
            self._get_input_shape(batch_size))
        target_batch = self.target_dataset_object.get_numpy_rows_at_idxs(indices)
        feed_dict = {
            input_placeholder: input_batch,
            target_placeholder: target_batch
        }
        return feed_dict

    def _get_input_shape(self, batch_size):
        shape = [batch_size, self.get_number_input_time_offsets(), -1]
        return shape
