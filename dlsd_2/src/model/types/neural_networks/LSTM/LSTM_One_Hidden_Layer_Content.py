import logging

class LSTM_One_Hidden_Layer_Content(NN_One_Hidden_Layer_Content):
    def __init__(self, data, target, number_hidden_nodes, learning_rate, number_rnn_steps=None):
        self.num_rnn_steps = number_rnn_steps
        super(LSTM_One_Hidden_Layer_Content, self).__init__(data, target, number_hidden_nodes, learning_rate)

    @tf_attributeLock
    def prediction(self):
        logging.info("Adding LSTM Prediction nodes to the graph")
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.n_hidden, state_is_tuple = True)
        outputs,last_states = tf.nn.dynamic_rnn(cell = cell, inputs = self.data_placeholder, dtype = tf.float32)
        last_output = outputs[:,self.num_rnn_steps-1,:]
        # outputs contains an tensor with shape ( batch size, rnn_sequence_length , n_hidden)
        # only the rnn layers are connected! to create the output layer of proper size need a new activation function!
        # activation function for output layer
        with tf.name_scope('outputLayer'):
            weights = tf.Variable(tf.truncated_normal((self.n_hidden,self.n_output),stddev=0.1), name="lay2_weights")
            bias = tf.Variable(tf.constant(0.1,shape=[self.n_output]), name="lay2_bias")
            out_layer2 = tf.nn.sigmoid(tf.matmul(last_output,weights)+bias, name = "lay2_output")
        return out_layer2

    @tf_attributeLock
    def error(self):
        logging.debug("Adding Error nodes to the graph")
        # using l2 norm (sum of) square error
        final_error = tf.square(tf.subtract(self.target_placeholder,self.prediction),name="myError")
        #tf.histogram_summary("final_error",final_error)
        mean = tf.reduce_mean(final_error,0)
        #tf.histogram_summary("mean_error",mean)
        return final_error
