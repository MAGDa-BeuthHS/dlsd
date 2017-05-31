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
		self.number_hidden_nodes = None
		self.learning_rate = None 
		self.max_steps = None
		self.batch_size = None 
		self.path_saved_session = None
		self.path_tf_output = None

	def set_number_hidden_nodes(self,number_hidden_nodes):
		self.number_hidden_nodes = number_hidden_nodes

	def set_learning_rate(self,learning_rate):
		self.learning_rate = learning_rate

	def set_max_steps(self,max_steps):
		self.max_steps = max_steps
		self.test_step = 1000

	def set_batch_size(self,batch_size):
		self.batch_size = batch_size

	def set_path_saved_session(self, path):
		self.path_saved_session = path

	def set_path_tf_output(self,path):
		self.path_tf_output = path

	def _build_model(self):
		raise NotImplementedError

	def _train(self):
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
			        print(mean)
			        logging.info("Training step : %d of %d"%(step,self.max_steps))
			        #logging.info("Mean test error is %f"%self.train_input_target_maker.denormalizer_used_in_training.denormalize(mean))
			self.saver.save(sess,self.path_saved_session) 

	def _test(self):
		self._build_model()
		with tf.Session(graph = self.graph) as sess:
			self.saver.restore(sess,self.path_saved_session)
			logging.info("Restored session for testing")
			feed_dict = {self.input_pl : self.model_input.get_all_input_as_numpy_array(), 
						self.target_pl : self.model_input.get_all_target_as_numpy_array()}
			prediction = sess.run(self.model_content.prediction,feed_dict=feed_dict)
			super(Neural_Network_Model,self).set_model_output_with_predictions_numpy_array(prediction)

	def get_target_and_predictions_df(self):
		targs_preds = super(Neural_Network_Model,self).get_target_and_predictions_df()
		return self.train_input_target_maker.denormalizer_used_in_training.denormalize(targs_preds)