from .decorators import tf_attributeLock
import tensorflow as tf

'''
    Simplest deep neural network that can be created

    Model is created in three steps for clarity.
    1. prediction : forward path through the neural network through a single hidden 
        layer to make an inference using a given input
    2. optimize : add derivative nodes to the graph and minimize the error using gradient descent
    3. error : compare prediction and target values
    tf_attributeLock decorator ensures that 1. operation nodes are only added to the graph
    once and that 2. nodes contained within are given a variable_scope name

    Alex Hartenstein 14/10/2016
'''


class SimpleNeuralNetwork:
    '''
        @ param data        tensorflow placeholder to hold input data
        @ param target      tensorflow placeholder to hold (true) output data (target value)
        @ param number_hidden_nodes
        @ param learning_rate
    '''
    def __init__(self, data, target, number_hidden_nodes, learning_rate):
        # data and target are placeholders
        self.data = data
        self.target = target
        # define hyperparameters of network
        self.n_input = int(self.data.get_shape()[1])
        self.n_hidden = number_hidden_nodes
        self.n_output = int(self.target.get_shape()[1])
        self.learningRate = learning_rate
        # reference operation attributes of model
        self.prediction
        self.optimize
        self.error
    
    
    '''

    '''
    @tf_attributeLock
    def prediction(self):
        with tf.name_scope('layer1'):
            weights = tf.Variable(tf.truncated_normal((self.n_input,self.n_hidden),stddev=0.1), name="lay1_weights")
            bias = tf.Variable(tf.constant(0.1,shape=[self.n_hidden]), name = "lay1_bias")
            out_layer1 = tf.nn.sigmoid(tf.matmul(self.data,weights)+bias, name = "lay1_output")
        with tf.name_scope('layer2'):
            weights = tf.Variable(tf.truncated_normal((self.n_hidden,self.n_output),stddev=0.1), name="lay2_weights")
            bias = tf.Variable(tf.constant(0.1,shape=[self.n_output]), name="lay2_bias")
            out_layer2 = tf.nn.sigmoid(tf.matmul(out_layer1,weights)+bias, name = "lay2_output")
        return out_layer2
       
    @tf_attributeLock
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learningRate, name = "gradientDescent")
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer_op = optimizer.minimize(self.error,global_step = global_step,name="minimizeGradientDescent")
        return optimizer_op
    
    @tf_attributeLock
    def error(self):
        final_error = tf.square(tf.sub(self.target,self.prediction),name="myError")
        tf.histogram_summary("final_error",final_error)
        return final_error