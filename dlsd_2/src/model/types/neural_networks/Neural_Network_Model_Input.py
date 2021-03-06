import logging

import numpy as np

from dlsd_2.src.model.Model_Input import Model_Input


class Neural_Network_Model_Input(Model_Input):

    def __init__(self):
        super(Neural_Network_Model_Input, self).__init__()
        logging.debug("\tAdding Neural_Network_Model_Input")
        self.status = None
        self.idxs_train = None
        self.idxs_test = None
        self.idxs_validation = None

    def fill_feed_dict(self, input_placeholder, target_placeholder, batch_size):
        indices = np.random.choice(self.get_number_datapoints(), batch_size, replace=False)
        return self._fill_feed_dict_with_indices(indices, input_placeholder, target_placeholder, batch_size)

    def fill_feed_dict_in_order(self, input_placeholder, target_placeholder, batch_size, i):
        batch_idxs = list(range(i * batch_size, (i + 1) * batch_size))
        return self._fill_feed_dict_with_indices(batch_idxs, input_placeholder, target_placeholder, batch_size)

    def _fill_feed_dict_with_indices(self, indices, input_placeholder, target_placeholder, batch_size):
        input_batch = self.input_dataset_object.get_numpy_rows_at_idxs(indices)
        target_batch = self.target_dataset_object.get_numpy_rows_at_idxs(indices)
        feed_dict = {
            input_placeholder: input_batch,
            target_placeholder: target_batch
        }
        return feed_dict

    def fill_last_feed_dict_in_order(self, input_placeholder, target_placeholder, batch_size, num_batches):
        batch_idxs = list(range(num_batches * batch_size, self.input_dataset_object.get_number_rows()))
        return self._fill_feed_dict_with_indices(batch_idxs, input_placeholder, target_placeholder, len(batch_idxs))
