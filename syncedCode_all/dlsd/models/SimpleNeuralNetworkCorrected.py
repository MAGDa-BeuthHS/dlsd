from .decorators import tf_attributeLock
import tensorflow as tf
from dlsd import Common as c

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

==============================================================================
'''


class SimpleNeuralNetwork:
    
    def __init__(self, data, target, number_hidden_nodes, learning_rate):
        '''
            Args : 
                data :                      tensorflow placeholder to hold input data
                target :                    tensorflow placeholder to hold (true) output data (target value)
                number_hidden_nodes : 
                learning_rate :

        '''
        # data and target are placeholders
        self.data = data
        self.target = target
        # define hyperparameters of network
        self.n_input = int(self.data.get_shape()[1])
        self.n_hidden = number_hidden_nodes
        self.n_output = int(self.target.get_shape()[1])
        self.learningRate = learning_rate

        c.debugInfo(__name__,"#input : %d   #hidden : %d   #output : %d   learningRate : %.2f"%(self.n_input,self.n_hidden,self.n_output,self.learningRate))
        # reference operation attributes of model
        self.prediction
        self.optimize
        self.error
        self.evaluation

 
    @tf_attributeLock
    def prediction(self):
        c.debugInfo(__name__,"Adding Prediction nodes to the graph")
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
        c.debugInfo(__name__,"Adding Optimize nodes to the graph")
        optimizer = tf.train.GradientDescentOptimizer(self.learningRate, name = "gradientDescent")
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer_op = optimizer.minimize(self.error,global_step = global_step,name="minimizeGradientDescent")
        return optimizer_op
    
    @tf_attributeLock
    def error(self):
        c.debugInfo(__name__,"Adding Error nodes to the graph")
        # using l2 norm (sum of) square error
        final_error = tf.square(tf.sub(self.target,self.prediction),name="myError")
        tf.histogram_summary("final_error",final_error)
        mean = tf.reduce_mean(final_error,0)
        tf.histogram_summary("mean_error",mean)
        return final_error

    @tf_attributeLock
    def evaluation(self):
        c.debugInfo(__name__,"Adding Evaluation nodes to the graph")
        # using l2 norm (sum of) square error
        final_error = tf.abs(tf.sub(self.target,self.prediction,name="myEvaluationError"))
        tf.histogram_summary("evaluation_final_error",final_error)
        mean = tf.reduce_mean(final_error)
        tf.scalar_summary("evaluation_mean_error",mean)
        return mean
    