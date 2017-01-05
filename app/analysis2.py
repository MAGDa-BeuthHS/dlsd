from dlsd.dataset import dataset_helpers as dsh
from dlsd.dataset import dataset_sqlToNumpy as stn

from dlsd.models import SimpleNeuralNetwork as model
from dlsd import Common as c
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import time


''' ------------------------------------------------------------------------------------------------
    Configuration Objects containing the data to do training/inference on 
    Data is stored in a FullDataSet object with both train/test data 
    ------------------------------------------------------------------------------------------------ '''

class Configuration:
    ''' Parent class configuration object. change file paths here '''
    n_hidden = 60
    learningRate = 0.01
    max_steps = 20000
    batch_size = 20
    test_step = 5000
    timeOffset = 15

    def setPathNames(args):
        path = args.outputDirectory
        Configuration.path_outputDir = args.outputDirectory
        Configuration.path_TFDir = os.path.join(path,'tensorFlow')
        Configuration.path_TFoutput = os.path.join(Configuration.path_TFDir,time.strftime("%Y_%m_%d__%H_%M"))        
        Configuration.path_sqlFile = args.pathToSQLFile
        Configuration.path_preparedData = args.preparedData
        Configuration.path_specifiedSensors = os.path.join(path,'sensorsUsed.csv' if args.specifiedSensors == None else args.specifiedSensors)
        Configuration.path_predictionOutputs = os.path.join(path,'AllPredictions.csv' if args.predictionOutput == None else args.predictionOutput)
        Configuration.path_savedSession = os.path.join(path,"model.ckpt")
        print(args.adjacencyMatrixPath)
        print('IS THE PATH')
        Configuration.path_adjacency = None if args.adjacencyMatrixPath is None else args.adjacencyMatrixPath
        print(Configuration.path_adjacency)
        if not os.path.exists(Configuration.path_TFDir):
            os.makedirs(Configuration.path_TFDir)
        if not os.path.exists(Configuration.path_TFoutput):
            os.makedirs(Configuration.path_TFoutput)

    def makeTrackedPredictions(self):
        '''
            TrackedPredictions is an empty dataframe to be filled with predictions (one column per training step)
            First column contains the desired target values
            Is only called if -tp in command line arguments
        '''
        self.trackedPredictions = pd.DataFrame(np.zeros((self.data.getNumberTestPoints(),1+int(Configuration.max_steps/Configuration.test_step))))
        self.trackedPredictions.iloc[:,0]= dsh.denormalizeData(self.data.test.outputData,self.data.max_value)

class config_train_dataFromPreparedFile(Configuration):
    ''' If csv file of data (already prepared in previous step, from SQL) exists use this config '''
    def __init__(self):
        self.data = makeData(path_preparedData = Configuration.path_preparedData, path_adjacencyMatrix=Configuration.path_adjacency)

class config_train_dataFromSQL(Configuration):
    ''' Prepare data from from SQL and save output for further training. Prerequisite for other configs'''
    def __init__(self):
        self.data = makeData(path_sqlFile = Configuration.path_sqlFile, path_preparedData = Configuration.path_preparedData, path_sensorsList = Configuration.path_specifiedSensors, path_adjacencyMatrix=Configuration.path_adjacency)

class config_restoreSess_dataFromPreparedFile(Configuration):
    ''' If want to run model on every data point (no test/train split) using a restored tf session'''
    def __init__(self):
        self.data = makeData(path_preparedData = Configuration.path_preparedData, splitTrain = False, path_adjacencyMatrix=Configuration.path_adjacency)
        Configuration.makeTrackedPredictions(self)

class config_restoreSess_dataFromSQL(Configuration):
    ''' Use when wish to test a restored tensorflow session on a new dataset (data is in SQL output form)'''
    def __init__(self):
        specifiedSensors = pd.read_csv(Configuration.path_specifiedSensors,header=None).values
        self.data = makeData(path_sqlFile = Configuration.path_sqlFile, splitTrain = False, specifiedSensorsArray = specifiedSensors, path_adjacencyMatrix=Configuration.path_adjacency)
        Configuration.makeTrackedPredictions(self)

''' ------------------------------------------------------------------------------------------------
    Create Neural Network and perform training 
    ------------------------------------------------------------------------------------------------ '''

def main(args):   
    # show debug information or not
    if (args.verbose != None): c.verbose = True

    Configuration.setPathNames(args)

    # Create configuration object containing data and hyperparameters
    if (args.restoreSess != None):
        if (args.pathToSQLFile != None): config = config_restoreSess_dataFromSQL()
        else: config = config_restoreSess_dataFromPreparedFile()
    else:
        if (args.pathToSQLFile != None): config = config_train_dataFromSQL()
        else: config = config_train_dataFromPreparedFile() 
    
        if (args.trackPredictions != None) : config_track = config_restoreSess_dataFromPreparedFile() 


    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[None,config.data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[None,config.data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        c.debugInfo(__name__,"Creating neural network")
        nn = model.SimpleNeuralNetwork(pl_input,pl_output,config.n_hidden,config.learningRate)
        
        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

    with tf.Session(graph = graph) as sess:

        summary_writer = tf.train.SummaryWriter(config.path_TFoutput, sess.graph)
        
        if (args.restoreSess != None):
            saver.restore(sess,config.path_savedSession)
            c.debugInfo(__name__,"Restored session")
            test_DataPrintOutput(nn,sess,pl_input,pl_output,config,fileName = config.path_predictionOutputs)
        
        else:
            sess.run(tf.initialize_all_variables())
            

            for step in range(config.max_steps):
                myFeedDict = config.data.train.fill_feed_dict(
                                           pl_input,
                                           pl_output,
                                           Configuration.batch_size)

                loss_value,summary_str,predicted = sess.run([nn.optimize,summary_op,nn.prediction],feed_dict = myFeedDict)
                if(step%Configuration.test_step == 0):
                    if (args.trackPredictions != None): test_allDataAppendToDf(nn,sess,pl_input,pl_output,config_track,int(step/config.test_step)+1)
                    #c.debugInfo(__name__,dsh.denormalizeData(predicted,config.data.max_value))
                    summary_writer.add_summary(summary_str)
                    summary_writer.flush()
                    
                    myFeedDict = config.data.test.fill_feed_dict(
                                           pl_input,
                                           pl_output,
                                           config.batch_size)
                    mean = sess.run(nn.evaluation,feed_dict = myFeedDict)
                    c.debugInfo(__name__,"Training step : %d of %d"%(step,config.max_steps))
                    c.debugInfo(__name__,"Mean test error is %f"%dsh.denormalizeData(mean,config.data.max_value))
                
            path_savedSession = saver.save(sess, config.path_savedSession)
            
            # save tracked predictions
            if (args.trackPredictions != None): config_track.trackedPredictions.to_csv(os.path.join(config_track.path_outputDir,"trackedPredictions.csv"),index=False)
            
            c.debugInfo(__name__,"Model saved in file: %s" % path_savedSession)



''' ------------------------------------------------------------------------------------------------
    Prepare data for training
    ------------------------------------------------------------------------------------------------ '''

def makeData(path_sqlFile=None,
                path_preparedData = None,
                timeOffset = 15,
                splitTrain = True,
                trainTestFraction =.8,
                specifiedSensorsArray = None,
                path_sensorsList = None,
                path_adjacencyMatrix=None):
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
    print("THE TIME OFFSET IS %d"%timeOffset)
    if (path_sqlFile is not None):
        c.debugInfo(__name__,"Processing data from an SQL file %s"%path_sqlFile)
        data_df, max_value = dsh.normalizeData(stn.sqlToNumpy_allSensorsInAllOutWithTimeOffset(path_sqlFile,
                                                    outputFilePath = path_preparedData,
                                                    timeOffset=Configuration.timeOffset,
                                                    sensorsOutputPath = path_sensorsList,
                                                    specifiedSensors = specifiedSensorsArray))
    # If no SQL data then open file and min/max normalize data
    else:
        c.debugInfo(__name__,"Opening preprocessed data file %s"%path_preparedData)
        data_df, max_value = dsh.normalizeData(pd.read_csv(path_preparedData))
        print("OUT")

    # first half of data is input
    indexOutputBegin = int((data_df.shape[1])/2)

    # define index of single output sensor (the output is at some time in the future)
    outputSensorIndex = 0

    # add in the adjacency matrix
    if (path_adjacencyMatrix is not None):
        c.debugInfo(__name__,"Found an adjacency matrix : multiplying it in!")
        # 182-185,281 are missing from adjacency matrix!! remove them! tell max, this needs to be changed!
        data_df=pd.DataFrame(data_df.iloc[:,5:data_df.shape[1]].values,columns=data_df.columns.values[5:data_df.shape[1]])
        indexOutputBegin = indexOutputBegin-5

        # list of sensors columns that we are using
        desired = data_df.columns.values[0:indexOutputBegin]

        # read adjacency matrix
        adjMatrix_orig = pd.read_csv(path_adjacencyMatrix)

        # adjacency matrix csv has headers as type string, with columns 0,1 actual strings : rename all columns as ints!
        sensorsList = list(adjMatrix_orig.columns.values[2:adjMatrix_orig.shape[1]].astype(np.int64))
        columns = [0,1]+sensorsList
        adjMatrix_orig.columns = columns

        # remove all columns (sensors) that we don't want, leaving only sensors that are desired
        removed = adjMatrix_orig[desired]

        # get row index of single sensor being used for output (as a string) : this row is the adjacency!
        indexForSensorInMatrix = np.where(adjMatrix_orig.iloc[:,1]==data_df.columns.values[outputSensorIndex])[0]
        adjacencyForOutputSensor = removed.iloc[indexForSensorInMatrix,:].values

        data_df.iloc[:,0:indexOutputBegin] = data_df.iloc[:,0:indexOutputBegin].values*adjacencyForOutputSensor

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


''' ------------------------------------------------------------------------------------------------
    Testing : Methods to Restore from session and Track Progress 
    ------------------------------------------------------------------------------------------------ '''

def test_DataPrintOutput(nn,sess,pl_input,pl_output,config,fileName):
    ''' 
        Use after a session restore to use neural network for inference against all data points 
        (non randomized) and print output
        Args : 
            nn :            A neural network class which contains a prediction attribute
            sess :          A tensorflow session
            config :        A config class, in this case the config used for restore (all data in test, none in training)
            fileName :      Optional : default is outputDir/AllPredictions.csv

    '''
    prediction = test_nonRandomizedPrediction(nn,sess,pl_input,pl_output,config)
    output = pd.DataFrame(np.empty((config.data.getNumberTestPoints(),2)))
    output.iloc[:,0]=dsh.denormalizeData(config.data.test.outputData,config.data.max_value)
    output.iloc[:,1]=dsh.denormalizeData(prediction,config.data.max_value)
    output.columns=["true","prediction"]
    c.debugInfo(__name__,"Printing prediction output to %s"%fileName)
    output.to_csv(os.path.join(config.path_outputDir,fileName),index=False)

def test_allDataAppendToDf(nn,sess,pl_input,pl_output,config,i,fileName="AllPredictionsOverTime.csv"):
    '''
        config_tracking contains a dataframe containing output against time

    '''
    prediction = test_nonRandomizedPrediction(nn,sess,pl_input,pl_output,config)
    config.trackedPredictions.iloc[:,i]=dsh.denormalizeData(prediction,config.data.max_value)

def test_nonRandomizedPrediction(nn,sess,pl_input,pl_output,config):
    ''' 
        Use to do prediction with the model using *non randomized* test data (meaning indices of datapoints
        is unchanged)
    '''
    myFeedDict = {
                pl_input : config.data.test.inputData,
                pl_output : config.data.test.outputData,
            }
    prediction = sess.run(nn.prediction,feed_dict=myFeedDict)
    return prediction


if __name__ == "__main__":
    args = c.makeCommandLineArgs()

    main(args)