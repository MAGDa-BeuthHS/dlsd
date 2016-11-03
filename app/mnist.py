from dlsd.dataset import dataset_generators as dsg
from dlsd.dataset import dataset_helpers as dsh
from dlsd.models import SimpleNeuralNetwork as models
from dlsd import Common as c
import tensorflow as tf
import os
import pandas as pd
import numpy as np

''' ------------------------------------------------------------------------------------------------
    Configuration Objects containing the data to do training/inference on 
    Data is stored in a FullDataSet object with both train/test data 
    ------------------------------------------------------------------------------------------------ '''
class config:
    inputFile = '/Users/ahartens/Desktop/Temporary/mnist_train.csv'
    output_dir = '/Users/ahartens/Desktop/tf'
    max_steps = 10000
    batch_size = 9
    learningRate = 0.3
    n_hidden = 50
    save_path = os.path.join(output_dir,"model.ckpt")

class config_mnist_training(config):
    # remake data from SQL output
    def __init__(self):
        self.data = makeData(config.inputFile)

class config_restore(config):
    # remake data from SQL output
    def __init__(self):
        self.data = makeInferenceData(config.inputFile)
        config.batch_size = self.data.getNumberTestPoints()
        


''' ------------------------------------------------------------------------------------------------
    Create Neural Network and perform training 
    ------------------------------------------------------------------------------------------------ '''

def main():
    # get command line arguments
    args = c.makeCommandLineArgs()

    # show debug information or not
    if (args.verbose != None): c.verbose = True

    # restore data if command line argument exists
    if (args.restore != None):
        c.debugInfo(__name__,"Restoring saved tf session")
        config = config_restore()
    
    # create a FullDataSet object containing train/test data as well as next_batch() method
    else: config = config_mnist_training()
    
    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[config.batch_size,config.data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[config.batch_size,config.data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        c.debugInfo(__name__,"Creating neural network")
        nn = models.MNISTNeuralNetwork(pl_input,pl_output,config.n_hidden,config.learningRate)
        
        # create summary operation and saver
        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

    # start session
    with tf.Session(graph = graph) as sess:

        summary_writer = tf.train.SummaryWriter(config.output_dir, sess.graph)
        
        # if restoring data perform inference exactly once and save output to csv
        if (args.restore != None):
            saver.restore(sess,config.save_path)
            c.debugInfo(__name__,"Restored session")
            myFeedDict = dsg.fill_feed_dict(config.data.test,
                                        pl_input,
                                        pl_output,
                                        config.batch_size)
            # do prediction on all data points (no training/test separation)
            mean, counts, predictions,targets = sess.run(nn.evaluation,feed_dict = myFeedDict)
            # create csv with correct label in column one and predicted label in column two
            output = pd.DataFrame(np.empty((config.data.getNumberTestPoints(),2)))
            output.iloc[:,0]=targets.reshape(-1,1)
            output.iloc[:,1]=predictions.reshape(-1,1)
            output.columns=["true","prediction"]
            output.to_csv('/Users/ahartens/Desktop/mnistoutput.csv',index=False)
        
        else:
            sess.run(tf.initialize_all_variables())

            for step in range(config.max_steps):
                myFeedDict = dsg.fill_feed_dict(config.data.train,
                                           pl_input,
                                           pl_output,
                                           config.batch_size)
                # train network by running optimize
                loss_value,summary_str = sess.run([nn.optimize,summary_op],feed_dict = myFeedDict)
                
                if(step%100 == 0):
                    summary_writer.add_summary(summary_str)
                    summary_writer.flush()
                    
                    myFeedDict = dsg.fill_feed_dict(config.data.test,
                                           pl_input,
                                           pl_output,
                                           config.batch_size)
                    mean, counts, predictions,targets = sess.run(nn.evaluation,feed_dict = myFeedDict)
                    print(mean)
                    print(counts)
                    print(predictions)
                    print(targets)
                    c.debugInfo(__name__,"Training step : %d of %d"%(step,config.max_steps))
            
            # save variables to file for later restore
            save_path = saver.save(sess, config.save_path)
            c.debugInfo(__name__,"Model saved in file: %s" % save_path)        




''' ------------------------------------------------------------------------------------------------
    Make Data (need to make one-hot-vector for output and concatenate)
    ------------------------------------------------------------------------------------------------ '''

def prepareDataFromFile(inputFilePath):
    '''
        Args:
            inputFilePath : path to mnist data. In first column is target value (0-9), 
                            following 784 columns are pixel values
        Return :
            final_df :      columns 0-784 are pixel values min/max normalized, 
                            last 10 columns are one-hot-vector of target output
    '''
    data_df = pd.read_csv(inputFilePath,sep=",")

    # get pixel values (first column is target value) and normalize them
    pixels = data_df.iloc[:,1:data_df.shape[1]].values
    max_value = np.amax(pixels)
    pixels = ((pixels/max_value)*.99) + 0.01

    # make one hot vectors of value
    target = data_df.iloc[:,0]
    target_array = np.zeros((data_df.shape[0],10))+.01
    indices = np.arange(0,data_df.shape[0],dtype=np.int).reshape(1,-1)
    target_array[indices,target] = .99

    final_df = pd.DataFrame(np.concatenate((pixels,target_array),1))
    
    return final_df

def makeData(inputFilePath):
    '''
        Args :
            inputFilePath : path to mnist data. 
        Return :
            theData :   FullDataSet object from dataset_helpers containing two DataSet 
                                objects containing two numpy arrays(input/target)
    '''
    # get (true value) targets and dataframe (input | output on each row as one hot vector from file)
    final_df = prepareDataFromFile(inputFilePath)

    # split data into train and test components
    train_df, test_df = dsh.splitDataToTrainAndTest(final_df,0.8)

    # output is the last ten columns of each row
    index_of_Output = final_df.shape[1]-10

    # set input dfs
    train_input = train_df.iloc[:,0:index_of_Output]
    test_input = test_df.iloc[:,0:index_of_Output]
    
    # set output dfs
    train_output = train_df.iloc[:,index_of_Output:final_df.shape[1]]
    test_output = test_df.iloc[:,index_of_Output:final_df.shape[1]]
    
    # create data wrapper containing input/output for train and test
    theData = dsh.FullDataSet(trainInput = train_input.values,
                            trainOutput = train_output.values,
                            testInput = test_input.values,
                            testOutput = test_output.values)
    theData.toString()
    
    return theData

def makeInferenceData(inputFilePath):
    '''
        Same as makeData except does NOT split train/test data
    '''
    final_df = prepareDataFromFile(inputFilePath)

    index_of_Output = final_df.shape[1]-10
    train_input = final_df.iloc[:,0:index_of_Output]
    test_input = final_df.iloc[:,0:index_of_Output]
    
    train_output = final_df.iloc[:,index_of_Output:final_df.shape[1]]
    test_output = final_df.iloc[:,index_of_Output:final_df.shape[1]]
    
    theData = dsh.FullDataSet(trainInput = train_input.values,
                            trainOutput = train_output.values,
                            testInput = test_input.values,
                            testOutput = test_output.values)
    theData.toString()
    return theData


if __name__ == "__main__":
    main()
