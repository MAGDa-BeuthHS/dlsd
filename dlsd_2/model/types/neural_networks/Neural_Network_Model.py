from dlsd_2.model.Model import *
from dlsd_2.model.Model_Output import Model_Output
from dlsd_2.model.types.neural_networks.Neural_Network_Model_Input import *

import tensorflow as tf

class Neural_Network_Model(Model):

	def __init__(self):
		super(Neural_Network_Model, self).__init__()
		logging.debug("\tAdding Neural_Network_Model")
		self.model_input = Neural_Network_Model_Input()
		self.model_output = Model_Output()

	def set_number_hidden_nodes(self,number_hidden_nodes):
		self.number_hidden_nodes = number_hidden_nodes

	def set_learning_rate(self,learning_rate):
		self.learning_rate = learning_rate

	def set_max_steps(self,max_steps):
		self.max_steps = max_steps
		self.test_step = 100

	def set_batch_size(self,batch_size):
		self.batch_size = batch_size

	def set_path_saved_session(self, path):
		self.path_saved_session = path

	def set_path_tf_output(self,path):
		self.path_tf_output = path

	def set_denormalizer_from_input_dataset(self):
		''' Test data should be denormalized exactly the same as the training dataset :
			this is only called during training '''
		self.model_input.set_denormalizer_from_input_dataset()
		self.model_output.set_denormalizer_from_input_dataset()

	def _build_model(self):
		raise NotImplementedError

	def _train(self):
		self.set_denormalizer_from_input_dataset() # test data should be normalized using same max value as training
		self._build_model()
		with tf.Session(graph = self.graph) as sess:
			self.sess = sess
			summary_writer = tf.train.SummaryWriter(self.path_tf_output, sess.graph)
			sess.run(tf.initialize_all_variables())
			for step in range(self.max_steps):
			    feed_dict = self.model_input.fill_feed_dict(self.input_pl,self.target_pl,self.batch_size)
			    loss_value,predicted = sess.run([self.model_content.optimize,self.model_content.prediction],feed_dict = feed_dict)
			    if(step%self.test_step == 0):
			        mean = sess.run(self.model_content.evaluation,feed_dict = feed_dict)
			        logging.info("Training step : %d of %d"%(step,self.max_steps))
			        logging.info("Mean test error is %f"%self.model_input.denormalizer.denormalize(mean))
			self.saver.save(sess,self.path_saved_session) 

	def _test(self):
		self._build_model()
		with tf.Session(graph = self.graph) as sess:
			self.saver.restore(sess,self.path_saved_session)
			logging.info("Restored session for testing")
			feed_dict = {self.input_pl : self.model_input.get_all_input_as_numpy_array(), 
						self.target_pl : self.model_input.get_all_target_as_numpy_array()}
			prediction = sess.run(self.model_content.prediction,feed_dict=feed_dict)

			self.model_output.set_prediction_dataset_object_with_numpy_array(prediction)
			self.model_output.set_target_dataset_object(self.model_input.get_target_dataset_object())
