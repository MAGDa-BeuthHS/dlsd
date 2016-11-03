from dlsd.dataset import dataset_generators as dsg
from dlsd.dataset import dataset_helpers as dsh

from dlsd.models import SimpleNeuralNetwork as model
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
    ''' Parent class configuration object. change file paths here '''
    n_hidden = 200
    learningRate = 0.01
    output_dir = '/Users/ahartens/Desktop/tf'
    path_sqlFile = '/Users/ahartens/Desktop/Work/24_10_16_PZS_Belegung_oneMonth.csv'
    path_inputFile = '/Users/ahartens/Desktop/Work/16_11_2_PZS_Belegung_oneMonth_oneTimeAsOutput_timeOffset15.csv'
    save_path = os.path.join(output_dir,"model.ckpt")
    max_steps = 10000
    batch_size = 3

class config_dataFromPreparedFile(config):
    ''' If csv file of data (already prepared in previous step, from SQL) exists use this config '''
    def __init__(self):
        self.data = makeData(config.path_inputFile)

class config_restoreForTesting(config):
    ''' If want to run model on every data point (no test/train split) using a restored tf session'''
    def __init__(self):
        self.data = makeData(config.path_inputFile, splitTrain = False)
        config.batch_size = self.data.getNumberTestPoints()

class config_dataFromSQL(config):
    ''' Prepare data from from SQL and save output for further training. Prerequisite for other configs'''
    def __init__(self):
        self.data = makeData(inputFilePath = config.path_sqlFile, remakeData =True, outputFilePath = config.path_inputFile, saveOutputFile = True)




''' ------------------------------------------------------------------------------------------------
    Create Neural Network and perform training 
    ------------------------------------------------------------------------------------------------ '''

def main():   
    # get command line arguments
    args = c.makeCommandLineArgs()

    # show debug information or not
    if (args.verbose != None): c.verbose = True


    # Create configuration object containing data and hyperparameters
    # restore data if command line argument exists
    if (args.restore != None): config = config_restoreForTesting()
    # prepare data for neural network from SQL input file
    elif(args.makeData != None): config = config_dataFromSQL()
    # create a FullDataSet object containing train/test data as well as next_batch() method
    else: config = config_dataFromPreparedFile()           
      

    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[config.batch_size,config.data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[config.batch_size,config.data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        c.debugInfo(__name__,"Creating neural network")
        nn = model.SimpleNeuralNetwork(pl_input,pl_output,config.n_hidden,config.learningRate)
        
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
                #print(myFeedDict)

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




''' ------------------------------------------------------------------------------------------------
    Prepare data for training
    ------------------------------------------------------------------------------------------------ '''

def makeData(inputFilePath,
                remakeData = False,
                outputFilePath = "",
                saveOutputFile = False, 
                timeOffset = 15,
                splitTrain = True,
                trainTestFraction =.8):
    '''
        Args : 
            inputFilePath :         Path to csv file 26_8_16_PZS_Belgugn_All_Wide_NanOmitec.csv or similar
            remakeData :            Boolean : if True then inputFilePath refers to an SQL output file and the data is remade
            outputFilePath :        Path to outputfile if saveOutputFile is True
            timeOffset :            Int : number of minutes that 
        
        Return :
            theData :       FullDataSet object from dataset_helpers containing two DataSet 
                            objects containing two numpy arrays(input/target), contains next_batch() function!
    '''
    # remake data from SQL output and min/max normalize it
    if (remakeData == True):
        c.debugInfo(__name__,"Processing data from an SQL file")
        data_df, max_value = dsh.normalizeData(stn.sqlToNumpy_allSensorsInAllOutWithTimeOffset(inputFilePath,
                                                    saveOutputFile = saveOutputFile,
                                                    outputFilePath = outputFilePath,
                                                    timeOffset=timeOffset))
    # open file and min/max normalize data
    else:
        c.debugInfo(__name__,"Opening preprocessed data file %s"%inputFilePath)
        data_df, max_value = dsh.normalizeData(pd.read_csv(inputFilePath))
    

    # first half of data is input
    indexOutputBegin = int((data_df.shape[1])/2)

    # define index of single output sensor (the output is at some time in the future)
    outputSensorIndex = 0

    if (splitTrain == True):
        train_df, test_df = dsh.splitDataToTrainAndTest(data_df,trainTestFraction)
        c.debugInfo(__name__,"train_df (%d,%d)\ttest_df (%d,%d)"%(train_df.shape[0],train_df.shape[1],test_df.shape[0],test_df.shape[1]))
        c.debugInfo(__name__,"Single output sensor at index %d, sensor name : %s"%(outputSensorIndex,data_df.columns.values[outputSensorIndex]))
        
        train_input = train_df.iloc[:,0:indexOutputBegin]
        train_output = train_df.iloc[:,indexOutputBegin+outputSensorIndex]

        test_input = test_df.iloc[:,0:indexOutputBegin]
        test_output = test_df.iloc[:,indexOutputBegin+outputSensorIndex]

        c.debugInfo(__name__,"Making FullDataSet object containing train/test data")
        # create FullDataSet object with appropriate data
        theData = dsh.FullDataSet(trainInput = train_input.values,
                                    trainOutput = train_output.values.reshape(-1,1),
                                    testInput = test_input.values,
                                    testOutput = test_output.values.reshape(-1,1))
    # Don't split data into train/test (only for testing)
    else:
        test_input = data_df.iloc[:,0:indexOutputBegin]
        test_output = data_df.iloc[:,indexOutputBegin+outputSensorIndex]        
        c.debugInfo(__name__,"Making FullDataSet object with only test data")
        # create FullDataSet object with appropriate data
        theData = dsh.FullDataSet(trainInput = np.empty(test_input.shape),
                                    trainOutput = np.empty(test_output.reshape(-1,1).shape),
                                    testInput = test_input.values,
                                    testOutput = test_output.values.reshape(-1,1))
    theData.max_value = max_value
    theData.toString()

    return theData





if __name__ == "__main__":
    main()