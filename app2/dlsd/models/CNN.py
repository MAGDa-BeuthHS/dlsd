from .decorators import tf_attributeLock 
from dlsd import Common as c
import tensorflow as tf



'''
	Convolutional Neural network
	Input : stack of images
	Output : value -1 or 1 if image contains ROI

	first 
'''

class CNN:
	def __init__(self,images_pl,targets_pl,fc_n_hidden,learningRate):

		self.fc_n_hidden = fc_n_hidden
		self.learningRate = learningRate
      	#self.n_input = int(self.images_pl.get_shape()[1])
        #self.n_output = int(self.targets_pl.get_shape()[1])

		
		self.images = images_pl
		self.targets = targets_pl

		self.prediction
		self.error
		self.optimize
		self.evaluate



	@tf_attributeLock
	def prediction(self):
		c.debugInfo(__name__,"Adding Prediction nodes to the graph")
		
		x_image = tf.reshape(self.images,[-1,128,128,1])

		n_features_1 = 64
		n_features_2 = 128
		n_features_3 = 256
		n_features_4 = 256
		n_features_5 = 512
		n_features_6 = 512
		n_features_7 = 512
		n_features_8 = 512

		ks = 3
		
		''' Convolution 1 '''
		with tf.name_scope("conv1"):
			kernel = _weight_variable('weights', shape=[ks,ks,1,n_features_1], stddev=.1)
			biases = _bias_variable('biases',[n_features_1])
			conv1 = tf.nn.relu(_conv2d(x_image,kernel,'conv1')+biases,name="conv1_output")
			_activation_summary(conv1)

		''' Pool 1 '''
		pool1 = _max_pool_2x2(conv1,'pool1')

		''' Convolution 2 '''
		with tf.name_scope("conv2"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_1,n_features_2], stddev=.1)
			biases = _bias_variable('biases',[n_features_2])
			conv2 = tf.nn.relu(_conv2d(pool1,kernel,'conv2')+biases,name="conv2_output")
			_activation_summary(conv2)
		
		''' Pool 2 '''
		pool2 = _max_pool_2x2(conv2,'pool2')

		

		''' Pool 3 '''
		pool3 = _max_pool_2x2(pool2,'pool3')


		''' Pool 4 '''
		pool4 = _max_pool_2x2(pool3,'pool4')

		'''
			Fully Connected Layers
			first calculate input size : image 
		'''
		with tf.name_scope("fc_inputLayer"):
			# flatten images : per row is an image with x*y*conv2Output size
			reshape = tf.reshape(pool4,[ -1,8*8*n_features_2])
			dim = reshape.get_shape()[1].value
			weights = _weight_variable('weights',
						shape=[dim,self.fc_n_hidden])
			bias = _bias_variable('bias',shape=[self.fc_n_hidden])
			fc_output_1 = tf.nn.sigmoid(tf.matmul(reshape,weights)+bias)
		with tf.name_scope("fc_readoutLayer"):
			weights = _weight_variable('weights',shape=[self.fc_n_hidden,1])
			bias = _bias_variable('bias',shape=[1])
			readout = tf.nn.sigmoid(tf.matmul(fc_output_1,weights)+bias)
		return readout
	@tf_attributeLock
	def predictionFull(self):
		c.debugInfo(__name__,"Adding Prediction nodes to the graph")
		
		x_image = tf.reshape(self.images,[-1,128,128,1])

		n_features_1 = 64
		n_features_2 = 128
		n_features_3 = 256
		n_features_4 = 256
		n_features_5 = 512
		n_features_6 = 512
		n_features_7 = 512
		n_features_8 = 512

		ks = 3
		
		''' Convolution 1 '''
		with tf.name_scope("conv1"):
			kernel = _weight_variable('weights', shape=[ks,ks,1,n_features_1], stddev=.1)
			biases = _bias_variable('biases',[n_features_1])
			conv1 = tf.nn.relu(_conv2d(x_image,kernel,'conv1')+biases,name="conv1_output")
			_activation_summary(conv1)

		''' Pool 1 '''
		pool1 = _max_pool_2x2(conv1,'pool1')

		''' Convolution 2 '''
		with tf.name_scope("conv2"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_1,n_features_2], stddev=.1)
			biases = _bias_variable('biases',[n_features_2])
			conv2 = tf.nn.relu(_conv2d(pool1,kernel,'conv2')+biases,name="conv2_output")
			_activation_summary(conv2)
		
		''' Pool 2 '''
		pool2 = _max_pool_2x2(conv2,'pool2')

		''' Convolution 3 '''
		with tf.name_scope("conv3"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_2,n_features_3], stddev=.1)
			biases = _bias_variable('biases',[n_features_3])
			conv3 = tf.nn.relu(_conv2d(pool2,kernel,'conv3')+biases,name="conv3_output")
			_activation_summary(conv3)

		''' Convolution 4 '''
		with tf.name_scope("conv4"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_3,n_features_4], stddev=.1)
			biases = _bias_variable('biases',[n_features_4])
			conv4 = tf.nn.relu(_conv2d(conv3,kernel,'conv4')+biases,name="conv4_output")
			_activation_summary(conv4)

		''' Pool 3 '''
		pool3 = _max_pool_2x2(conv4,'pool3')

		''' Convolution 5 '''
		with tf.name_scope("conv5"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_4,n_features_5], stddev=.1)
			biases = _bias_variable('biases',[n_features_5])
			conv5 = tf.nn.relu(_conv2d(pool3,kernel,'conv5')+biases,name="conv5_output")
			_activation_summary(conv5)

		''' Convolution 6 '''
		with tf.name_scope("conv6"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_5,n_features_6], stddev=.1)
			biases = _bias_variable('biases',[n_features_6])
			conv6 = tf.nn.relu(_conv2d(conv5,kernel,'conv6')+biases,name="conv6_output")
			_activation_summary(conv6)

		''' Pool 4 '''
		pool4 = _max_pool_2x2(conv6,'pool4')

		''' Convolution 7 '''
		with tf.name_scope("conv7"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_6,n_features_7], stddev=.1)
			biases = _bias_variable('biases',[n_features_7])
			conv7 = tf.nn.relu(_conv2d(pool4,kernel,'conv7')+biases,name="conv7_output")
			_activation_summary(conv7)

		''' Convolution 8 '''
		with tf.name_scope("conv8"):
			kernel = _weight_variable('weights', shape=[ks,ks,n_features_7,n_features_8], stddev=.1)
			biases = _bias_variable('biases',[n_features_8])
			conv8 = tf.nn.relu(_conv2d(conv7,kernel,'conv8')+biases,name="conv8_output")
			_activation_summary(conv8)
		'''
			Fully Connected Layers
			first calculate input size : image 
		'''
		with tf.name_scope("fc_inputLayer"):
			# flatten images : per row is an image with x*y*conv2Output size
			reshape = tf.reshape(conv8,[ -1,8*8*n_features_8])
			dim = reshape.get_shape()[1].value
			weights = _weight_variable('weights',
						shape=[dim,self.fc_n_hidden])
			bias = _bias_variable('bias',shape=[self.fc_n_hidden])
			fc_output_1 = tf.nn.sigmoid(tf.matmul(reshape,weights)+bias)
		with tf.name_scope("fc_readoutLayer"):
			weights = _weight_variable('weights',shape=[self.fc_n_hidden,1])
			bias = _bias_variable('bias',shape=[1])
			readout = tf.nn.sigmoid(tf.matmul(fc_output_1,weights)+bias)
		return readout

	@tf_attributeLock
	def error(self):
		c.debugInfo(__name__,"Adding Error nodes to the graph")
		# using l2 norm (sum of) square error
		#error_op = tf.square(tf.sub(self.targets,self.prediction),name="error")
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(self.prediction), reduction_indices=[1]))


		tf.histogram_summary("error",cross_entropy)
		return cross_entropy

	@tf_attributeLock
	def optimize(self):
		c.debugInfo(__name__,"Adding Optimize nodes to the graph")
		optimizer = tf.train.GradientDescentOptimizer(self.learningRate,name="gradientDescentOptimzier")
		global_step = tf.Variable(0,name='global_step',trainable=False)
		optimizer_op = optimizer.minimize(self.error,global_step=global_step,name="minimizeGradientDescent")

	@tf_attributeLock
	def evaluate(self):
		c.debugInfo(__name__,"Adding Evaluation nodes to the graph")
		predictions = self.prediction
		rounded = tf.round(predictions)
		correct_prediction = tf.equal(rounded,self.targets)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		return accuracy,correct_prediction,predictions,rounded



def _weight_variable(name,shape,stddev=0.1):
	return tf.Variable(tf.truncated_normal(shape, stddev=stddev),name=name)


def _bias_variable(name,shape):
	return tf.Variable(tf.constant(0.1, shape=shape),name=name)


def _activation_summary(x):
	tf.histogram_summary(x.op.name+'/activations',x)
	tf.scalar_summary(x.op.name+'/sparsity',tf.nn.zero_fraction(x))

def _conv2d(x,W,name):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name=name)

def _max_pool_2x2(x,name):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1], padding='SAME',name=name)
