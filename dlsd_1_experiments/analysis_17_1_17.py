from dlsd.dataset import dataset_helpers as dsh
from dlsd.dataset import dataset_sqlToNumpy as stn

from dlsd.models.SimpleNeuralNetwork import LSTM as lstm
from dlsd import debugInfo
from dlsd import makeCommandLineArgs

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import time

class Configuration:
    ''' Parent class configuration object. change file paths here '''
    n_hidden = 30
    learningRate = 0.01
    max_steps = 10000
    batch_size = 20
    test_step = 5000
    rnn_sequence_length = 5
    sequential = list(range(0,rnn_sequence_length))
    rnn_input_time_sequence = [0]#[0,5,10,15,20,25,30,35,40,45,50,55]
    rnn_target_time_sequence = [30]
    def __init__(self,args):
        self.setPathNames(args)

    def setPathNames(self,args):
        path = args.outputDirectory
        self.path_outputDir = args.outputDirectory
        self.path_TFDir = os.path.join(path,'tensorFlow')
        self.path_TFoutput = os.path.join(self.path_TFDir,time.strftime("%Y_%m_%d__%H_%M"))        
        self.path_sqlFile = args.pathToSQLFile
        self.path_sqlTestFile= args.pathTestData.split(",")
        self.path_preparedData = args.preparedData
        self.path_specifiedSensors = os.path.join(path,'sensorsUsed.csv' if args.specifiedSensors == None else args.specifiedSensors)
        self.path_predictionOutputs = os.path.join(path,'AllPredictions.csv' if args.predictionOutput == None else args.predictionOutput)
        self.path_savedSession = os.path.join(path,"model.ckpt")
        self.path_adjacencyMatrix = None if args.adjacencyMatrixPath is None else args.adjacencyMatrixPath
        self.sql_headers = None if args.sql_headers is None else args.sql_headers.split(',')
        if not os.path.exists(self.path_TFDir):
            os.makedirs(self.path_TFDir)
        if not os.path.exists(self.path_TFoutput):
            os.makedirs(self.path_TFoutput)


def main(args):

    config = Configuration(args)

    methods = [{'func':pd_1s_singleInput,'name':'ffnn_simple','adj':False},]
            #{'func':pd_2s_allInput,'name':'ffnn_all','adj':False},]
            #{'func':pd_3s_adjacency_withSelf,'name':'ffnn_nn+','adj':True},
            #{'func':pd_4s_adj_noSelf,'name':'ffnn_nn','adj':True}]

    # create training data (all of july)
    data_df, max_value, specifiedSensors = formatFromSQL(path_sqlFile = config.path_sqlFile,sql_headers = config.sql_headers)

    # 182-185,281 are missing from adjacency matrix!! remove them! tell max, this needs to be changed!
    remove = [182,183,184,185,281]
    removeInx = []
    for i in remove:
        index = np.where(data_df.columns.values == i)[0]
        if len(index) > 0 : removeInx.append(index[0])
    data_df.drop(data_df.columns[removeInx], axis=1, inplace=True)
    specifiedSensors = pd.DataFrame(data_df.columns.values)

    # create test data (one or more)
    config.test_dicts = []
    for path_test in config.path_sqlTestFile:
        test_df, test_max_value, _ = formatFromSQL(path_sqlFile = path_test,  specifiedSensorsArray = specifiedSensors, sql_headers = config.sql_headers)
        config.test_dicts.append({'df':test_df,'max':test_max_value,'name':os.path.basename(os.path.normpath(path_test)).replace('.csv','')})

    # create a list that contains the (function) index of the minimum MAE (averaged)
    path_idxsMinMae = os.path.join(config.path_outputDir,"indicesMinMaes.csv")
    idxMinMae_list = []

    ## FOR EACH SENSOR ##
    #for indexOutputSensor in range(0,data_df.shape[1]):
    for indexOutputSensor in range(0,1):

        # create folder for current sensor
        debugInfo(__name__,"SENSOR %d"%data_df.columns.values[indexOutputSensor])
        dir_sensor = os.path.join(config.path_outputDir,"s_%d"%data_df.columns.values[indexOutputSensor])
        dir_sensor_tf = os.path.join(dir_sensor,'tf')

        if not os.path.exists(dir_sensor): os.makedirs(dir_sensor)
        if not os.path.exists(dir_sensor_tf): os.makedirs(dir_sensor_tf)

        # set up empty data frame and array for the average MAE of all testing data. first column is names of all the functions
        testSetInfo = []

        # create necessary paths and empty arrays for each training set
        for i in range(0,len(config.test_dicts)):
            avgMaes_df = pd.DataFrame(np.zeros((len(methods),len(config.rnn_target_time_sequence))))
            avgMaes_array = np.zeros((len(methods),len(config.rnn_target_time_sequence)))
            path_avgMaes_df = os.path.join(dir_sensor,"avgMaesForSensor_%d.csv"%data_df.columns.values[indexOutputSensor])

            # create folders for current (test) data
            dir_test = os.path.join(dir_sensor,config.test_dicts[i]['name'])
            dir_test_tf = os.path.join(dir_test,'tf')
            dir_test_results = os.path.join(dir_test,'output')
            if not os.path.exists(dir_test): os.makedirs(dir_test)
            if not os.path.exists(dir_test_tf) : os.makedirs(dir_test_tf)
            if not os.path.exists(dir_test_results) : os.makedirs(dir_test_results)

            testSetInfo.append({'avgMaes_df':avgMaes_df,
                            'avgMaes_array':avgMaes_array,
                            'path_avgMaes_df':path_avgMaes_df,
                            'dir_test':dir_test,
                            'test_dr_tf':dir_test_tf,
                            'dir_test_results':dir_test_results})

        
            
        ## FOR EACH DATA PREPARATION METHOD ## 
        # this affects how the network is formed! changes every time
        for j in range(0,len(methods)):
            
            debugInfo(__name__,"Using %s to prepare data"%methods[j]['name'])
            config.path_savedSession = os.path.join(dir_sensor_tf,"tfsession_%s.ckpt"%methods[j]['name'])
            
            debugInfo(__name__,"Creating Data for Training")
            config.data = makeDataSetObject(data_df = data_df,
                                            max_value = max_value,
                                            outputSensorIndex = indexOutputSensor,
                                            prepareData_function=methods[j]['func'],
                                            path_adjacencyMatrix=None if (methods[j]['adj'] == False) else config.path_adjacencyMatrix,
                                            config=config)
            # the number of data points used as input (per time point). eg all sensors, 1 sensor, only nearest neighbors etc
            config.number_input_neurons = config.data.train.inputData.shape[2]  
            config.number_target_neurons = 1
            # train using training set
            trainNetwork(config)
            

            ## FOR EACH TEST DATA SET ##
            for i in range(0,len(config.test_dicts)):
                
                testData = config.test_dicts[i]
                      
                debugInfo(__name__,"Creating Data for Testing")
                config.path_outputFile = os.path.join(testSetInfo[i]['dir_test_results'],"%s.csv"%methods[j]['name'])
                config.data = makeDataSetObject(data_df = testData['df'],
                                                max_value = testData['max'],
                                                outputSensorIndex = indexOutputSensor,
                                                prepareData_function=methods[j]['func'],
                                                path_adjacencyMatrix=None if (methods[j]['adj'] == False) else config.path_adjacencyMatrix,                                            
                                                config=config)
                maes = testNetwork(config)

                testSetInfo[i]['avgMaes_df'].iloc[j,:] = maes


        # contains average maes for all test data sets (average of averages)
        avgMaeOverTests_df = pd.DataFrame(np.zeros((len(methods),len(config.rnn_target_time_sequence))))
        avgMaeOverTests_array = np.zeros((len(methods),len(config.rnn_target_time_sequence)))
        path_avgMaeOverTests = os.path.join(dir_sensor,"avgMaesForSensor_%d.csv"%data_df.columns.values[indexOutputSensor])

        # iterate over all 'avgMae' tables (for each test data set)
        for i in range(0,len(config.test_dicts)):
            testSetInfo[i]['avgMaes_df'].index = np.array([(lambda x:x['name'])(funcDic) for funcDic in methods])
            testSetInfo[i]['avgMaes_df'].to_csv(testSetInfo[i]['path_avgMaes_df'],
            header = ([(lambda x:"t_%d"%x)(to) for to in config.rnn_target_time_sequence]))
            avgMaeOverTests_array = testSetInfo[i]['avgMaes_df'].values + avgMaeOverTests_array
            
        # get average of maes for all test data
        avgMaeOverTests_array = avgMaeOverTests_array/len(config.test_dicts)
        avgMaeOverTests_df.iloc[:,:] = avgMaeOverTests_array
        avgMaeOverTests_df.index = [(lambda x:x['name'])(funcDic) for funcDic in methods]
        avgMaeOverTests_df.to_csv(path_avgMaeOverTests,header = ([(lambda x:"t_%d"%x)(to) for to in config.rnn_target_time_sequence]))
        
        # get index of function with lowest MAE and save
        idxMinMae_list.append(avgMaeOverTests_array.argmin(axis=0))
        
    idxMinMae_df = pd.DataFrame(np.hstack((specifiedSensors.values[0:2],np.array(idxMinMae_list))))
    idxMinMae_df.to_csv(path_idxsMinMae,header=(['sensor']+[(lambda x:"t_%d"%x)(to) for to in config.rnn_target_time_sequence]))

''' ------------------------------------------------------------------------------------------------
    Create Neural Network and perform training 
    ------------------------------------------------------------------------------------------------ '''

def setupNet(config):
    '''
        Creates the operation graph in tensorflow
        returns all components necessary for training/testing
    '''
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):

        # batch size, number of input sequences, size of input sequence
        pl_input = tf.placeholder(tf.float32,shape=[None,len(config.rnn_input_time_sequence),config.number_input_neurons],name="input_placeholder")

        # batch size, number of outputs (which is equal to number of input sequences), size of output (predicts 5 timeputs every )
        pl_output = tf.placeholder(tf.float32,shape=[None,len(config.rnn_target_time_sequence),config.number_target_neurons],name="target_placeholder")
        
        # create neural network and define in graph
        debugInfo(__name__,"Creating neural network")
        nn = lstm(data = pl_input,
                target = pl_output,
                number_hidden_nodes = config.n_hidden,
                learning_rate = config.learningRate,
                rnn_number_steps = len(config.rnn_input_time_sequence))

        saver = tf.train.Saver()
        summary_op = tf.merge_all_summaries()

        return pl_input, pl_output, nn, saver, graph, summary_op


def trainNetwork(config):

    pl_input, pl_output, nn, saver, graph , summary_op = setupNet(config)

    with tf.Session(graph = graph) as sess:

        summary_writer = tf.train.SummaryWriter(config.path_TFoutput, sess.graph)
    
        sess.run(tf.initialize_all_variables())

        for step in range(config.max_steps):
            myFeedDict = config.data.train.fill_feed_dict(
                                       pl_input,
                                       pl_output,
                                       Configuration.batch_size)

            sess.run([nn.optimize],feed_dict = myFeedDict)
            if(step%Configuration.test_step == 0):
                if (args.trackPredictions != None): test_allDataAppendToDf(nn,sess,pl_input,pl_output,config_track,int(step/config.test_step)+1)
                #debugInfo(__name__,dsh.denormalizeData(predicted,config.data.max_value))
                #summary_writer.add_summary(summary_str)
                #summary_writer.flush()
                
                mean,error,pred = sess.run([nn.evaluation,nn.error,nn.prediction],feed_dict = myFeedDict)
                print(dsh.denormalizeData(mean,config.data.max_value))
                debugInfo(__name__,"Training step : %d of %d"%(step,config.max_steps))
                debugInfo(__name__,"Mean test error is %f"%dsh.denormalizeData(mean,config.data.max_value))
        print(config.path_outputDir)
        print(config.path_savedSession)
        path_savedSession = saver.save(sess,config.path_savedSession)    
        print(path_savedSession)

''' ------------------------------------------------------------------------------------------------
    Create Data 
    ------------------------------------------------------------------------------------------------ '''

def formatFromSQL(path_sqlFile=None, path_preparedData = None,specifiedSensorsArray = None,sql_headers = None):
    # remake data from SQL output and min/max normalize it
    if (path_sqlFile is not None):
        debugInfo(__name__,"Processing data from an SQL file %s"%path_sqlFile)
        data_df,_,specifiedSensors = stn.pivotAndSmooth(inputFile = path_sqlFile,specifiedSensors = specifiedSensorsArray,sql_headers = sql_headers)
        data_df, max_value = dsh.normalizeData(data_df)
    # If no SQL data then open file and min/max normalize data
    else:
        debugInfo(__name__,"Opening preprocessed data file %s"%path_preparedData)
        data_df, max_value = dsh.normalizeData(pd.read_csv(path_preparedData))
    return data_df, max_value, specifiedSensors


def makeDataSetObject(data_df, 
                max_value,
                prepareData_function,
                path_adjacencyMatrix,
                outputSensorIndex,
                config=None):
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
    adjacencyForOutputSensor = None
    # add in the adjacency matrix
    
    if (path_adjacencyMatrix is not None):
        debugInfo(__name__,"Found an adjacency matrix : multiplying it in!")
        # 182-185,281 are missing from adjacency matrix!! remove them! tell max, this needs to be changed!
        #data_df=pd.DataFrame(data_df.iloc[:,5:data_df.shape[1]].values,columns=data_df.columns.values[5:data_df.shape[1]])

        # list of sensors columns that we are using
        desired = data_df.columns.values

        # read adjacency matrix
        adjMatrix_orig = pd.read_csv(path_adjacencyMatrix)

        # adjacency matrix csv has headers as type string, with columns 0,1 actual strings : rename all columns as ints!
        sensorsList = list(adjMatrix_orig.columns.values[2:adjMatrix_orig.shape[1]].astype(np.int64))
        columns = [0,1]+sensorsList
        adjMatrix_orig.columns = columns

        # remove all columns (sensors) that we don't want, leaving only sensors that are desired
        # this uses header names to reference the columns that i want
        removed = adjMatrix_orig[desired]

        # get row index of single sensor being used for output (as a string) : this row is the adjacency!
        indexForSensorInMatrix = np.where(adjMatrix_orig.iloc[:,1]==data_df.columns.values[outputSensorIndex])[0]
        adjacencyForOutputSensor = removed.iloc[indexForSensorInMatrix,:].values
        print(data_df.columns.values[np.where(adjacencyForOutputSensor[0]==1)[0]])
    
    # create input and output vectors
    input_,output_, indexOutputBegin = prepareData(data_df.values,
                indexOutputSensor = outputSensorIndex,
                inputFunction = prepareData_function,
                config=config,
                adjacency = adjacencyForOutputSensor)

    
    debugInfo(__name__,"Making FullDataSet object containing train/test data")
    # create FullDataSet object with appropriate data
    theData = dsh.FullDataSet(trainInput = input_,
                                trainOutput = output_)
    theData.max_value = max_value
    theData.train.rowNames = data_df.index[:-(max(config.rnn_target_time_sequence)-1)]
    return theData

def prepareData(data_wide,
                indexOutputSensor,
                inputFunction,
                config=None,
                adjacency = None):
    '''
        Creates a dataframe containing desired input/output within the same table
        Args:
            data_wide : numpy array of all data (eg pivoted and smooth sqlToNumpy output)
            indexOutputSensor : the sensor to be predicted
            timeOffsets : python list of desired output times
            inputFunction : pd_ function (1 of 8) that formats input data in the desired manner
            adjacency : optional : a single numpy vector
            sequential : optional : python list (like timeOffsets) specifying which time points as input

            target :                 input : 



            t_5                     t_0
            t_6                     t_1
            .                       .
            .                       .
            t15                     t_10
    '''
    # input data is moved vertically down by max of timeOffsets
    #max_output = max(timeOffsets)
    max_output = 0

    index_output_begin = max(config.rnn_input_time_sequence) + min(config.rnn_target_time_sequence)

    i = inputFunction(data_wide,indexOutputSensor,
                      s = config.rnn_input_time_sequence,
                      a = adjacency,
                      max_output=0)[max(config.rnn_input_time_sequence):-index_output_begin,:]
    #df = pd.DataFrame(i)
    #df.to_csv("/Users/ahartens/Desktop/input.csv")

    i = i.reshape([i.shape[0],len(config.rnn_input_time_sequence),-1])
    print(i.shape)
    debugInfo(__name__,"Preparing data : %d inputs %d"%(i.shape[1],i.shape[0]))
    # create 'output' data : 
    #o = timeOffsetData(data_wide[:,indexOutputSensor],timeOffsets,b=max(sequential))
    o = data_wide[index_output_begin:,indexOutputSensor]
    #df = pd.DataFrame(o)
    #df.to_csv("/Users/ahartens/Desktop/output.csv")

    o = o.reshape([o.shape[0],len(config.rnn_target_time_sequence),1])
    debugInfo(__name__,"Preparing data : %d outputs %d"%(o.shape[1],o.shape[0]))
    

   
    # combine input/output in one dataframe
    return i,o,i.shape[1]



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
    
'''  Sequential : input is time offset leading up to t0 '''
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
    
    # create empty data frame
    df = np.zeros((data.shape[0]+maxOffset+b+t,len(offsets)*data.shape[1]))
    df[:] = np.NAN

    # set 'output' data : offset by increasing time offset
    for i in range(0,len(offset_comp)):
        off = offset_comp[i]
        start = i*data.shape[1]
        end = start+data.shape[1]
        df[off+t:df.shape[0]-offsets[i]-b,start:end]=data
    return df

# ------------------------------------------------------------------------------------------------
#    Testing : Methods to Restore from session and Track Progress 
# ------------------------------------------------------------------------------------------------

def testNetwork(config):
    ''' Train all data using a session saved at config.path_savedSession '''   
    pl_input, pl_output, nn, saver, graph, _ = setupNet(config)
    with tf.Session(graph = graph) as sess:
        saver.restore(sess,config.path_savedSession)
        debugInfo(__name__,"Restored session")
        #return test_DataPrintOutput(nn,sess,pl_input,pl_output,config,fileName = config.path_outputFile)
        prediction = test_nonRandomizedPrediction(nn,sess,pl_input,pl_output,config)
    
    output = pd.DataFrame(np.empty((config.data.getNumberTrainingPoints(),2*config.number_target_neurons)))
    y = dsh.denormalizeData(config.data.train.outputData.reshape([-1,config.number_target_neurons]),config.data.max_value)
    y_  = dsh.denormalizeData(prediction.reshape([-1,config.number_target_neurons]),config.data.max_value)
    output.iloc[:,0:config.number_target_neurons]= y
    output.iloc[:,config.number_target_neurons:2*config.number_target_neurons]= y_


    #output.index = config.data.train.rowNames
    output.to_csv(config.path_outputFile)
    debugInfo(__name__,"Printing prediction output to %s"%config.path_outputFile)

    mae = np.mean(np.abs(y - y_),0)
    print(mae)
    return mae


def test_allDataAppendToDf(nn,sess,pl_input,pl_output,config,i,fileName="AllPredictionsOverTime.csv"):
    ''' config_tracking contains a dataframe containing output against time '''
    prediction = test_nonRandomizedPrediction(nn,sess,pl_input,pl_output,config)
    config.trackedPredictions.iloc[:,i]=dsh.denormalizeData(prediction,config.data.max_value)

def test_nonRandomizedPrediction(nn,sess,pl_input,pl_output,config):
    ''' Use to do prediction with the model using *non randomized* test data (meaning indices of datapoints
        is unchanged) '''
    myFeedDict = {
                pl_input : config.data.train.inputData,
                pl_output : config.data.train.outputData,
            }
    prediction = sess.run(nn.prediction,feed_dict=myFeedDict)
    return prediction


if __name__ == "__main__":
    args = makeCommandLineArgs()

    main(args)