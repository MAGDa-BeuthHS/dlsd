from dlsd.dataset import dataset_helpers as dsh
from dlsd.dataset import dataset_sqlToNumpy as stn

from dlsd.models import SimpleNeuralNetwork as model
from dlsd import debugInfo
from dlsd import makeCommandLineArgs

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import time

class Configuration:
    ''' Parent class configuration object. change file paths here '''
    n_hidden = 60
    learningRate = 0.01
    max_steps = 20000
    batch_size = 20
    test_step = 5000
    timeOffset = 15
    def __init__(self,args):
        self.setPathNames(args)

    def setPathNames(self,args):
        path = args.outputDirectory
        self.path_outputDir = args.outputDirectory
        self.path_TFDir = os.path.join(path,'tensorFlow')
        self.path_TFoutput = os.path.join(self.path_TFDir,time.strftime("%Y_%m_%d__%H_%M"))        
        self.path_sqlFile = args.pathToSQLFile
        self.path_preparedData = args.preparedData
        self.path_specifiedSensors = os.path.join(path,'sensorsUsed.csv' if args.specifiedSensors == None else args.specifiedSensors)
        self.path_predictionOutputs = os.path.join(path,'AllPredictions.csv' if args.predictionOutput == None else args.predictionOutput)
        self.path_savedSession = os.path.join(path,"model.ckpt")
        self.path_adjacency = None if args.adjacencyMatrixPath is None else args.adjacencyMatrixPath
        if not os.path.exists(self.path_TFDir):
            os.makedirs(self.path_TFDir)
        if not os.path.exists(self.path_TFoutput):
            os.makedirs(self.path_TFoutput)

    def makeTrackedPredictions(self):
        '''
            TrackedPredictions is an empty dataframe to be filled with predictions (one column per training step)
            First column contains the desired target values
            Is only called if -tp in command line arguments
        '''
        self.trackedPredictions = pd.DataFrame(np.zeros((self.data.getNumberTestPoints(),1+int(self.max_steps/self.test_step))))
        self.trackedPredictions.iloc[:,0]= dsh.denormalizeData(self.data.test.outputData,self.data.max_value)

    ''' If csv file of data (already prepared in previous step, from SQL) exists use this config '''
    def train_dataFromPreparedFile(self,dataPrepareMethod):
        self.data = makeData(path_preparedData = self.path_preparedData, path_adjacencyMatrix=self.path_adjacency,prepareData_function=dataPrepareFunction)

    ''' Prepare data from from SQL and save output for further training. Prerequisite for other configs'''
    def train_dataFromSQL(self,dataPrepareMethod):
        self.data = makeData(path_sqlFile = self.path_sqlFile, path_preparedData = self.path_preparedData, path_sensorsList = self.path_specifiedSensors, path_adjacencyMatrix=self.path_adjacency,prepareData_function=dataPrepareFunction)

    ''' If want to run model on every data point (no test/train split) using a restored tf session'''
    def restoreSess_dataFromPreparedFile(self,dataPrepareMethod):
        self.data = makeData(path_preparedData = self.path_preparedData, splitTrain = False, path_adjacencyMatrix=self.path_adjacency,prepareData_function=dataPrepareFunction)
        self.makeTrackedPredictions(self)

    ''' Use when wish to test a restored tensorflow session on a new dataset (data is in SQL output form)'''
    def restoreSess_dataFromSQL(self,dataPrepareMethod):
        specifiedSensors = pd.read_csv(self.path_specifiedSensors,header=None).values
        self.data = makeData(path_sqlFile = self.path_sqlFile, splitTrain = False, specifiedSensorsArray = specifiedSensors, path_adjacencyMatrix=self.path_adjacency,prepareData_function=dataPrepareFunction)
        self.makeTrackedPredictions()

''' ------------------------------------------------------------------------------------------------
    Create Neural Network and perform training 
    ------------------------------------------------------------------------------------------------ '''


def main(args):   
    # show debug information or not
    if (args.verbose != None): verbose = True

    config = Configuration(args)

    timeOffsets = [5,10,15,30,45]
    sequential = list(range(0,5))

    indexOutputSensor = 1
    method = 4
    methods = [pd_1_singleInput,
            pd_2_allInput,
            pd_3_adjacency_withSelf,
            pd_4_adj_noSelf,
            pd_1s_singleInput,
            pd_2s_allInput,
            pd_3s_adjacency_withSelf,
            pd_4s_adj_noSelf]

    data_df, max_value = makeDataFromSQL(path_sqlFile = self.path_sqlFile,
                                        path_preparedData = self.path_preparedData)

    for i in range(1,len(methods)):
        makeData(prepareData_function=dataPrepareFunction)

    # Create configuration object containing data and hyperparameters
    if (args.restoreSess != None):
        if (args.pathToSQLFile != None): config.restoreSess_dataFromSQL()
        else: config.restoreSess_dataFromPreparedFile()
    else:
        if (args.pathToSQLFile != None): config.train_dataFromSQL()
        else: config.train_dataFromPreparedFile() 
    
        if (args.trackPredictions != None) : 
            config_track = Configuration(args)
            config_track.restoreSess_dataFromPreparedFile() 

def trainNetwork(args,config):


    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[None,config.data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[None,config.data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        debugInfo(__name__,"Creating neural network")
        nn = model.SimpleNeuralNetwork(pl_input,pl_output,config.n_hidden,config.learningRate)
        
        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

    with tf.Session(graph = graph) as sess:

        summary_writer = tf.train.SummaryWriter(config.path_TFoutput, sess.graph)
        
        if (args.restoreSess != None):
            saver.restore(sess,config.path_savedSession)
            debugInfo(__name__,"Restored session")
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
                    #debugInfo(__name__,dsh.denormalizeData(predicted,config.data.max_value))
                    summary_writer.add_summary(summary_str)
                    summary_writer.flush()
                    
                    myFeedDict = config.data.test.fill_feed_dict(
                                           pl_input,
                                           pl_output,
                                           config.batch_size)
                    mean = sess.run(nn.evaluation,feed_dict = myFeedDict)
                    debugInfo(__name__,"Training step : %d of %d"%(step,config.max_steps))
                    debugInfo(__name__,"Mean test error is %f"%dsh.denormalizeData(mean,config.data.max_value))
                
            path_savedSession = saver.save(sess, config.path_savedSession)
            
            # save tracked predictions
            if (args.trackPredictions != None): config_track.trackedPredictions.to_csv(os.path.join(config_track.path_outputDir,"trackedPredictions.csv"),index=False)
            
            debugInfo(__name__,"Model saved in file: %s" % path_savedSession)

def makeDataFromSQL(path_sqlFile=None, path_preparedData = None,specifiedSensorsArray = None):
    # remake data from SQL output and min/max normalize it
    if (path_sqlFile is not None):
        debugInfo(__name__,"Processing data from an SQL file %s"%path_sqlFile)
        data_df,_ = stn.sqlToNumpy_pivotAndSmooth(path_sqlFile,specifiedSensorsArray)
        data_df, max_value = dsh.normalizeData(data_df)
    # If no SQL data then open file and min/max normalize data
    else:
        debugInfo(__name__,"Opening preprocessed data file %s"%path_preparedData)
        data_df, max_value = dsh.normalizeData(pd.read_csv(path_preparedData))
    return data_df, max_value

def fillDataSetObject(data_df, max_value,
                timeOffsets = None,
                splitTrain = True,
                trainTestFraction =.8,
                specifiedSensorsArray = None,
                path_sensorsList = None,
                path_adjacencyMatrix=None,
                prepareData_function):
    '''
        Args : 
            inputFilePath :         Path to csv file 26_8_16_PZS_Belgugn_All_Wide_NanOmitecsv or similar
            remakeData :            Boolean : if True then inputFilePath refers to an SQL output file and the data is remade
            outputFilePath :        Path to outputfile if saveOutputFile is True
            timeOffset :            Int : number of minutes that 
        
        Return :
            theData :       FullDataSet object from dataset_helpers containing two DataSet 
                            objects containing two numpy arrays(input/target), contains next_batch() function!
    '''

    # define index of single output sensor (the output is at some time in the future)
    outputSensorIndex = 0
    adjacencyForOutputSensor = None
    # add in the adjacency matrix
    if (path_adjacencyMatrix is not None):
        debugInfo(__name__,"Found an adjacency matrix : multiplying it in!")
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

        #data_df.iloc[:,0:indexOutputBegin] = data_df.iloc[:,0:indexOutputBegin].values*adjacencyForOutputSensor


    data_prepared,indexOutputBegin = prepareData(data_df.values,
                outputSensorIndex,
                timeOffsets,
                prepareData_function,
                adjacency = adjacencyForOutputSensor,
                sequential = sequential)

    data_prepared.to_csv("/Users/ahartens/Desktop/ty.csv")
    data_final_naDropped = data_prepared.dropna()
    debugInfo(__name__,"From %d total timepoints, %d are being used (%.2f)"%(data_prepared.shape[0],data_final_naDropped.shape[0],(data_final_naDropped.shape[0]/data_prepared.shape[0])))

    #data_final.to_csv("/Users/ahartens/Desktop/Temporary/24_10_16_wideTimeSeriesBelegung.csv")
    if (path_preparedData is not None):
        debugInfo(__name__,"Saving processed file to %s"%(path_preparedData))
        data_final_naDropped.to_csv(path_preparedData,index=False)

    if (splitTrain == True):
        train_df, test_df = dsh.splitDataToTrainAndTest(data_final_naDropped,trainTestFraction)
        debugInfo(__name__,"train_df (%d,%d)\ttest_df (%d,%d)"%(train_df.shape[0],train_df.shape[1],test_df.shape[0],test_df.shape[1]))
        debugInfo(__name__,"Single output sensor at index %d, sensor name : %s"%(outputSensorIndex,data_df.columns.values[outputSensorIndex]))
        
        train_input = train_df.iloc[:,0:indexOutputBegin]
        train_output = train_df.iloc[:,indexOutputBegin:data_final_naDropped.shape[1]]

        test_input = test_df.iloc[:,0:indexOutputBegin]
        test_output = test_df.iloc[:,indexOutputBegin:data_final_naDropped.shape[1]]

        debugInfo(__name__,"Making FullDataSet object containing train/test data")
        # create FullDataSet object with appropriate data
        theData = dsh.FullDataSet(trainInput = train_input.values,
                                    trainOutput = train_output.values,
                                    testInput = test_input.values,
                                    testOutput = test_output.values)
    # Don't split data into train/test (only for testing)
    else:
        test_input = data_final_naDropped.iloc[:,0:indexOutputBegin]
        test_output = data_final_naDropped.iloc[:,indexOutputBegin+outputSensorIndex]        
        debugInfo(__name__,"Making FullDataSet object with only test data")
        # create FullDataSet object with appropriate data
        theData = dsh.FullDataSet(trainInput = np.empty(test_input.shape),
                                    trainOutput = np.empty(test_output.reshape(-1,1).shape),
                                    testInput = test_input.values,
                                    testOutput = test_output.values.reshape(-1,1))
    theData.max_value = max_value
    theData.toString()

    return theData

def prepareData(data_wide,
                indexOutputSensor,
                timeOffsets,
                inputFunction,
                adjacency=None,
                sequential=[0]):
    '''
        Creates a dataframe containing desired input/output within the same table
        Args:
            data_wide : numpy array of all data (eg pivoted and smooth sqlToNumpy output)
            indexOutputSensor : the sensor to be predicted
            timeOffsets : python list of desired output times
            inputFunction : pd_ function (1 of 8) that formats input data in the desired manner
            adjacency : optional : a single numpy vector
            sequential : optional : python list (like timeOffsets) specifying which time points as input
    '''
    # input data is moved vertically down by max of timeOffsets
    max_output = max(timeOffsets)
    max_sequential = max(sequential)
    
    i = inputFunction(data_wide,indexOutputSensor,
                      s = sequential,
                      a = adjacency,
                      max_output=max_output,
                      max_sequential=max_sequential)
    
    # create 'output' data : 
    o = timeOffsetData(data_wide[:,indexOutputSensor],timeOffsets,b=max(sequential))

    print(i.shape)
    print(o.shape)
    # combine input/output in one dataframe
    df = pd.DataFrame(np.hstack((i,o)))
    return df, i.shape[1]

'''
    Preparing Data for 8 types of data sets
    All have the same output : values that are time offset for a single sensor
    1 : Single sensor (self) input
    2 : All sensors
    3 : All neighboring sensors, inclusive self
    4 : All neighboring sensors, exclusive self
    
    The s types are identical to 1-4 except that they are also time offset (similar to the output)
'''

def pd_1_singleInput(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    i = np.zeros((data_wide.shape[0]+max_output,1))
    i[:] = np.NAN
    i[max_output:i.shape[0],0] = data_wide[:,indexOutputSensor]
    return i

def pd_2_allInput(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    i = np.zeros((data_wide.shape[0]+max_output,data_wide.shape[1]))
    i[:] = np.NAN
    i[max_output:i.shape[0],:] = data_wide[:,:]
    return i

def pd_3_adjacency_withSelf(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    indicesToUse = np.where(a!=0)[0]
    i = np.zeros((data_wide.shape[0]+max_output,len(indicesToUse)))
    i[:] = np.NAN
    i[max_output:i.shape[0],:] = data_wide[:,indicesToUse]
    return i

def pd_4_adj_noSelf(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    # remove column corresponding to sensor
    data_wide = np.delete(data_wide,indexOutputSensor,1)
    a = np.delete(a,indexOutputSensor)
    return pd_3_adjacency_withSelf(data_wide,indexOutputSensor,max_output=max_output,a=a,s=s)
    
''' 
    Sequential : input is time offset leading up to t0 
'''

def pd_1s_singleInput(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    return timeOffsetData(data_wide[:,indexOutputSensor],offsets=s,t=max_output,b=0)

def pd_2s_allInput(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    return timeOffsetData(data_wide,offsets=s,t=max_output,b=0)

def pd_3s_adjacency_withSelf(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    indicesToUse = np.where(a!=0)[0]
    return timeOffsetData(data_wide[:,indicesToUse],offsets=s,t=max_output,b=0)

def pd_4s_adj_noSelf(data_wide,indexOutputSensor,a,s,max_output = None,max_sequential=None):
    data_wide = np.delete(data_wide,indexOutputSensor,1)
    a = np.delete(a,indexOutputSensor)
    return pd_3s_adjacency_withSelf(data_wide,indexOutputSensor,max_output=max_output,a=a,s=s)



def timeOffsetData(data,offsets,t=0,b=0):
    '''
        Create output data for a time series. each column contains datapoints 
        for a single sensor. 'slide' the columns along each other, where each
        index represents one minute. result : each row contains t5,t10,t15
    '''
    if len(data.shape)==1:
        data = data.reshape(-1,1)

    # calculate where each column of output should begin. 
    # eg t5 should start at index 55 if maxoffset is 60
    maxOffset = max(offsets)
    offset_comp = [(lambda x,y:y-x)(offset,maxOffset) for offset in offsets]
    df = np.zeros((data.shape[0]+maxOffset+b+t,len(offsets)*data.shape[1]))
    df[:] = np.NAN

    # set 'output' data : offset by increasing time offset
    for i in range(0,len(offset_comp)):
        off = offset_comp[i]
        start = i*data.shape[1]
        end = start+data.shape[1]
        df[off+t:df.shape[0]-offsets[i]-b,start:end]=data
    return df

if __name__ == "__main__":
    args = makeCommandLineArgs()

    main(args)