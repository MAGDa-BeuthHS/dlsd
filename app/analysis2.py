from dlsd.dataset import dataset_generators as dsg
from dlsd.dataset import dataset_helpers as dsh

from dlsd.models import SimpleNeuralNetwork as nn
from dlsd import Common as c
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import argparse

class config_inference:
    def __init__(self):
        self.data = dsg.makeData_allSensorsInOneOutWithTimeOffset('/Users/ahartens/Desktop/Temporary/24_10_16_wideTimeSeriesBelegung_naDropped.csv',
                                        remakeData =False,
                                        splitTrain = False)
        self.output_dir = '/Users/ahartens/Desktop/tf'
        self.save_path = os.path.join(self.output_dir,"model.ckpt")
        self.batch_size = self.data.getNumberTestPoints()
        self.n_hidden = 40
        self.learningRate = 0

class config_allInOneOut:
    def __init__(self):
        self.data = dsg.makeData_allSensorsInOneOutWithTimeOffset('/Users/ahartens/Desktop/Temporary/24_10_16_wideTimeSeriesBelegung_naDropped.csv',remakeData =False)
        self.output_dir = '/Users/ahartens/Desktop/tf'
        self.max_steps = 10000
        self.batch_size = 10
        self.learningRate = 0.3
        self.n_hidden = 40
        self.save_path = os.path.join(self.output_dir,"model.ckpt")


class config_allInAllOut:
    # remake data from SQL output
    def __init__(self):
        self.data = dsg.makeData_allSensorsInAllOutWithTimeOffset(inputFilePath = '/Users/ahartens/Desktop/Work/24_10_16_PZS_Belegung_limited.csv',
            remakeData =True,
            outputFilePath ='/Users/ahartens/Desktop/Temporary/16_10_26_wideTimeSeriesBelegung.csv',
            saveOutputFile = True)
        self.output_dir = '/Users/ahartens/Desktop/tf'
        self.max_steps = 1000
        self.batch_size = 10
        self.learningRate = 0.3
        self.n_hidden = 40
    

def makeCommandLineArgs():
    parser = argparse.ArgumentParser(description='Run a neural network')
    parser.add_argument('-r','--restore', help='Restore from a file',required=False)
    parser.add_argument('-v','--verbose',help='Print error loggin messages', required=False)
    args = parser.parse_args()
    return args
     

if __name__ == "__main__":
    # set debugInfo verbose variable
    args = makeCommandLineArgs()

    if (args.verbose != None):
        c.verbose = True

    if (args.restore != None):
        c.debugInfo(__name__,"Restoring saved tf session")
        config = config_inference()
    else:   
        # create a FullDataSet object containing train/test data as well as next_batch() method
        config = config_allInOneOut()           
      
    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[config.batch_size,config.data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[config.batch_size,config.data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        c.debugInfo(__name__,"Creating neural network")
        nn = nn.SimpleNeuralNetwork(pl_input,pl_output,config.n_hidden,config.learningRate)
        
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()


    with tf.Session(graph = graph) as sess:

        summary_writer = tf.train.SummaryWriter(config.output_dir, sess.graph)
        
        if (args.restore != None):
            saver.restore(sess,config.save_path)
            c.debugInfo(__name__,"Restored session")
            myFeedDict = dsg.fill_feed_dict(config.data.test,
                                        pl_input,
                                        pl_output,
                                        config.batch_size)
            prediction = sess.run(nn.prediction,feed_dict=myFeedDict)
            output = pd.DataFrame(np.empty((config.data.getNumberTestPoints(),2)))
            output.iloc[:,0]=dsh.denormalizeData(config.data.test.outputData,config.data.max_value)
            output.iloc[:,1]=dsh.denormalizeData(prediction,config.data.max_value)
            output.columns=["true","prediction"]
            output.to_csv('/Users/ahartens/Desktop/output.csv',index=False)
        
        else:
            sess.run(tf.initialize_all_variables())

            for step in range(config.max_steps):
                myFeedDict = dsg.fill_feed_dict(config.data.train,
                                           pl_input,
                                           pl_output,
                                           config.batch_size)
                loss_value,summary_str,predicted = sess.run([nn.optimize,summary_op,nn.prediction],feed_dict = myFeedDict)
                if(step%100 == 0):
                    print(dsh.denormalizeData(predicted,config.data.max_value))
                    summary_writer.add_summary(summary_str)
                    summary_writer.flush()
                    
                    myFeedDict = dsg.fill_feed_dict(config.data.test,
                                           pl_input,
                                           pl_output,
                                           config.batch_size)
                    mean = sess.run(nn.evaluation,feed_dict = myFeedDict)
                    print(dsh.denormalizeData(mean,config.data.max_value))
                    c.debugInfo(__name__,"Training step : %d of %d"%(step,config.max_steps))
                
            save_path = saver.save(sess, config.save_path)

            c.debugInfo(__name__,"Model saved in file: %s" % save_path)
