import tensorflow as tf
from dlsd_2.model.types.neural_networks.Neural_Network_Model_Input import *

from dlsd_2.src.model.Model import *


class Neural_Network_Model(Model):

	def __init__(self):
		super(Neural_Network_Model, self).__init__()
		logging.debug("\tAdding Neural_Network_Model")
		self.number_hidden_nodes = None
		self.learning_rate = None 
		self.max_steps = None
		self.test_step = 5000
		self.batch_size = None 
		self.path_saved_tf_session = None
		self.path_tf_output = None
		self.epochs = None
		self.use_epochs = True
		self.predictions = None
		self.use_cross_validation = False
		self.fill_output_timegaps = True
	
	def define_model_input(self):
		self.model_input = Neural_Network_Model_Input()

	def set_number_hidden_nodes(self,number_hidden_nodes):
		self.number_hidden_nodes = number_hidden_nodes

	def set_learning_rate(self,learning_rate):
		self.learning_rate = learning_rate

	def set_dont_use_epochs_instead_max_steps(self,max_steps):
		self.max_steps = max_steps
		self.use_epochs = False

	def set_num_epochs(self,num_epochs):
		self.epochs = num_epochs

	def set_batch_size(self,batch_size):
		self.batch_size = batch_size

	def set_path_saved_tf_session(self, path):
		self.path_saved_tf_session = path

	def set_path_tf_output(self,path):
		self.path_tf_output = path

	def _build_model(self):
		raise NotImplementedError

	def _train(self):
		if self.use_cross_validation : self.model_input.status = 'train'

		if self.use_epochs:
			self._train_with_epochs_and_ordered_feed_dict()
		else:
			self._train_with_fixed_number_of_steps_and_random_feed_dict()

	def _train_with_epochs_and_ordered_feed_dict(self):
		self._build_model()
		with tf.Session(graph = self.graph) as sess:
			self.sess = sess
			summary_writer = tf.summary.FileWriter(self.path_tf_output, sess.graph)
			sess.run(tf.global_variables_initializer())
			self._iterate_over_epochs(sess)
			self.saver.save(sess,self.path_saved_tf_session)

	def _iterate_over_epochs(self, sess):
		for epoch in range(self.epochs):
			logging.info("EPOCH #"+str(epoch))
			self._iterate_over_data_in_batches(sess)

	def _iterate_over_data_in_batches(self, sess):
		num_batches = int(self.model_input.get_number_datapoints()/self.batch_size)
		for step in range(num_batches):
		    feed_dict = self.model_input.fill_feed_dict_in_order(self.input_pl,self.target_pl,self.batch_size,step)
		    loss_value,predicted = sess.run([self.model_content.optimize,self.model_content.prediction],feed_dict = feed_dict)
		    self._print_progress(sess,step, feed_dict)

	def _print_progress(self, sess, step, feed_dict):
		if(step%self.test_step == 0):
			mean = sess.run(self.model_content.evaluation,feed_dict = feed_dict)
			print(mean)
			logging.info("Training step : %d"%(step))
			#logging.info("Mean test error is %f"%self.train_input_target_maker.denormalizer_used_in_training.denormalize(mean))

	def _train_with_fixed_number_of_steps_and_random_feed_dict(self):
		self._build_model()
		with tf.Session(graph = self.graph) as sess:
			self.sess = sess
			summary_writer = tf.summary.FileWriter(self.path_tf_output, sess.graph)
			sess.run(tf.global_variables_initializer())
			for step in range(self.max_steps):
			    feed_dict = self.model_input.fill_feed_dict_in_order(self.input_pl,self.target_pl,self.batch_size, step)
			    loss_value,predicted = sess.run([self.model_content.optimize,self.model_content.prediction],feed_dict = feed_dict)
			    if(step%self.test_step == 0):
			        mean = sess.run(self.model_content.evaluation,feed_dict = feed_dict)
			        logging.info("Training step : %d of %d"%(step,self.max_steps))
			        #logging.info("Mean test error is %f"%self.train_input_target_maker.denormalizer_used_in_training.denormalize(mean))
			self.saver.save(sess,self.path_saved_tf_session)

	def _test_all_data_at_once(self):
		self._build_model()
		with tf.Session(graph = self.graph) as sess:
			self.saver.restore(sess,self.path_saved_tf_session)
			logging.info("Restored session for testing")
			feed_dict = {self.input_pl : self.model_input.get_all_input_as_numpy_array(), 
						self.target_pl : self.model_input.get_all_target_as_numpy_array()}
			prediction = sess.run(self.model_content.prediction,feed_dict=feed_dict)
			super(Neural_Network_Model,self).set_model_output_with_predictions_numpy_array(prediction)

	def _test(self):
		self._build_model()
		self.predictions = None
		with tf.Session(graph = self.graph) as sess:
			self.saver.restore(sess,self.path_saved_tf_session)
			print(sess.run(self.model_content.global_step))
			logging.info("Restored session for testing")
			self._iterate_over_data_in_batches_for_testing(sess)
		print(self.predictions.shape)
		super(Neural_Network_Model,self).set_model_output_with_predictions_numpy_array(self.predictions)

	def _iterate_over_data_in_batches_for_testing(self, sess):
		num_batches = int(self.model_input.get_number_datapoints()/self.batch_size)
		for step in range(num_batches):
			feed_dict = self.model_input.fill_feed_dict_in_order(self.input_pl, self.target_pl, self.batch_size, step)
			predicted = sess.run(self.model_content.prediction, feed_dict = feed_dict)
			self._add_predicted_to_predictions(predicted)
		self._test_last_batch(sess, num_batches)

	def _test_last_batch(self, sess, num_batches):
		if num_batches*self.batch_size < self.model_input.get_number_datapoints() :
			feed_dict = self.model_input.fill_last_feed_dict_in_order(self.input_pl, self.target_pl, self.batch_size, num_batches)
			predicted = sess.run(self.model_content.prediction, feed_dict = feed_dict)
			self._add_predicted_to_predictions(predicted)

	def _add_predicted_to_predictions(self, predicted):
		if self.predictions is None:
			self.predictions = np.copy(predicted)
		else:
			self.predictions = np.concatenate((self.predictions,predicted), axis = 0)

	def get_target_and_predictions_df(self):
		if self.fill_output_timegaps : self.model_output.fill_time_gaps_in_target_and_predictions_using_time_format(self.current_input_target_maker.time_format)
		targs_preds = super(Neural_Network_Model,self).get_target_and_predictions_df()
		targs_preds = self.train_input_target_maker.denormalizer_used_in_training.denormalize(targs_preds)
		return targs_preds

	def get_prediction_df(self):
		if self.fill_output_timegaps : self.model_output.fill_time_gaps_in_predictions_using_time_format(self.current_input_target_maker.time_format)
		preds = self.model_output.get_prediction_df()
		preds = self.train_input_target_maker.denormalizer_used_in_training.denormalize(preds)
		return preds

	def set_experiment_helper(self, experiment_helper):
		super(Neural_Network_Model,self).set_experiment_helper(experiment_helper)
		self.set_path_tf_output(self.experiment_helper.get_tensorflow_dir_path())	
		self.set_path_saved_tf_session(self.experiment_helper.new_tf_session_file_path_with_specifier(self.name))

