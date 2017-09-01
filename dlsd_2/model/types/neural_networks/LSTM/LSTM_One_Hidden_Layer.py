from dlsd_2.model.types.neural_networks.Neural_Network_Model import *
from .LSTM_One_Hidden_Layer_Content import LSTM_One_Hidden_Layer_Content
from .LSTM_One_Hidden_Layer_Input import LSTM_One_Hidden_Layer_Input


class LSTM_One_Hidden_Layer(Neural_Network_Model):

	def __init__(self):
		super(LSTM_One_Hidden_Layer,self).__init__()
		logging.info("Creating Neural Network with One Hidden Layer")
		self.name = "NN_One_Hidden_Layer"

	def define_model_input(self):
		print("HERE DEFINING LSTM MODEL INPUT")
		self.model_input = LSTM_One_Hidden_Layer_Input()


	def _build_model(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			print(self.model_input.get_number_inputs())
			self.input_pl = tf.placeholder(tf.float32, shape=self._get_input_shape(),name="input_placeholder")
			self.target_pl = tf.placeholder(tf.float32,shape=[None,self.model_input.get_number_targets()],name="target_placeholder")
			self.model_content = LSTM_One_Hidden_Layer_Content(data = self.input_pl,
																target = self.target_pl,
																number_hidden_nodes = self.number_hidden_nodes,
																learning_rate = self.learning_rate,
																number_rnn_steps = self.model_input.get_number_input_time_offsets())
			#self.summary_op = tf.merge_all_summaries()
			self.saver = tf.train.Saver()

	def _get_input_shape(self):
		# input tensor must be of shape : batch_size x num_timesteps x num_features
		batch_size = None
		num_timesteps = self.model_input.get_number_input_time_offsets()
		num_features = self.model_input.get_number_inputs()/num_timesteps
		return [batch_size, num_timesteps, num_features]
