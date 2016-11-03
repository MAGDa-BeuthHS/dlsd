from . import dataset_helpers as dsh
from . import dataset_sqlToNumpy as stn
import pandas as pd
import numpy as np
from dlsd import Common as c
'''
    dataset_generators

    This module contains functions to reproducibly create FullDataSet objects from
    source files. 

    Also contains fill_feed_dict() method to get random batches of data for training.

    Alex Hartenstein 14/10/2016

'''


def fill_feed_dict(data_set,input_pl, output_pl, batch_size):
    '''
    Args : 
        data_set :      a FullDataSet object from dataset_helpers 
                                containing two DataSet objects (for next_batch method)
        input_pl :      tensorflow Placeholder for input data
        output_pl :     tensorflow Placeholder for correct data
        batch_size :    int value determining how many rows of dataset to feed into dictionary
    Return :
        feed_dict      dict with placeholder:numpy array for giving to session.run() method
    '''
    inputData,correctOutputData = data_set.next_batch(batch_size)
    feed_dict = {
        input_pl : inputData,
        output_pl : correctOutputData,
    }
    return feed_dict

def makeData_allSensorsWithAllTimeOffsetAsInput(inputFilePath,
                                            remakeData = False,
                                            outputFilePath = "",
                                            saveOutputFile = False, 
                                            timeOffset = 15,
                                            splitTrain = True,
                                            trainTestFraction =.8):
    '''
        16_11_2 : Analysis 3
        Args : 
            inputFilePath :         Path to csv file 26_8_16_PZS_Belgugn_All_Wide_NanOmitec.csv or similar
            remakeData :            Boolean : if True then inputFilePath refers to an SQL output file and the data is remade
            outputFilePath :        Path to outputfile if saveOutputFile is True
            timeOffset :            Int : number of minutes that 
        
        Return :
            theData :       FullDataSet object from dataset_helpers containing two DataSet 
                            objects containing two numpy arrays(input/target), contains next_batch() function!
    '''
    if (remakeData == True):
        c.debugInfo(__name__,"Processing data from an SQL file")
        data_df, max_value = dsh.normalizeData(stn.sqlToNumpy_allSensorsWithAllTimeOffsetAsInput(inputFilePath,saveOutputFile = saveOutputFile,outputFilePath = outputFilePath,timeOffset=timeOffset))
    else:
        c.debugInfo(__name__,"Opening preprocessed data file %s"%inputFilePath)
        data_df, max_value = dsh.normalizeData(pd.read_csv(inputFilePath))
    
    

    # first half of data is input

    indexOutputBegin = int((data_df.shape[1])/timeOffset)

    # define index of single output sensor (the output is at some time in the future)
    outputSensorIndex = 0

    if (splitTrain == True):
        train_df, test_df = dsh.splitDataToTrainAndTest(data_df,trainTestFraction)
        c.debugInfo(__name__,"train_df (%d,%d)\ttest_df (%d,%d)"%(train_df.shape[0],train_df.shape[1],test_df.shape[0],test_df.shape[1]))
        c.debugInfo(__name__,"Single output sensor at index %d, sensor name : %s"%(outputSensorIndex,data_df.columns.values[outputSensorIndex]))
        
        train_input = train_df.iloc[:,indexOutputBegin:data_df.shape[1]]
        train_output = train_df.iloc[:,outputSensorIndex]

        test_input = test_df.iloc[:,indexOutputBegin:data_df.shape[1]]
        test_output = test_df.iloc[:,outputSensorIndex]

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
        theData = dsh.FullDataSet(trainInput = np.empty((0,0)),
                                    trainOutput = np.empty((0,0)),
                                    testInput = test_input.values,
                                    testOutput = test_output.values.reshape(-1,1))
    theData.max_value = max_value
    theData.toString()

    return theData


def makeData_allSensorsInOneOutWithTimeOffset(inputFilePath,
                                            remakeData = False,
                                            outputFilePath = "",
                                            saveOutputFile = False, 
                                            timeOffset = 15,
                                            splitTrain = True,
                                            trainTestFraction =.8):
    '''
        16_10_20 : Analysis 2

        Args : 
            inputFilePath :         Path to csv file 26_8_16_PZS_Belgugn_All_Wide_NanOmitec.csv or similar
            remakeData :            Boolean : if True then inputFilePath refers to an SQL output file and the data is remade
            outputFilePath :        Path to outputfile if saveOutputFile is True
            timeOffset :            Int : number of minutes that 
        
        Return :
            theData :       FullDataSet object from dataset_helpers containing two DataSet 
                            objects containing two numpy arrays(input/target), contains next_batch() function!
    '''
    if (remakeData == True):
        c.debugInfo(__name__,"Processing data from an SQL file")
        data_df, max_value = dsh.normalizeData(stn.sqlToNumpy_allSensorsInAllOutWithTimeOffset(inputFilePath,saveOutputFile = saveOutputFile,outputFilePath = outputFilePath,timeOffset=timeOffset))
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
        theData = dsh.FullDataSet(trainInput = np.empty((0,0)),
                                    trainOutput = np.empty((0,0)),
                                    testInput = test_input.values,
                                    testOutput = test_output.values.reshape(-1,1))
    theData.max_value = max_value
    theData.toString()

    return theData

def makeData_allSensorsInAllOutWithTimeOffset(inputFilePath,
                                            remakeData = False,
                                            outputFilePath = "",
                                            saveOutputFile = False, 
                                            timeOffset = 15,
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
    if (remakeData == True):
        c.debugInfo(__name__,"Processing data from an SQL file")
        data_df,max_value = dsh.normalizeData(stn.sqlToNumpy_allSensorsInAllOutWithTimeOffset(inputFilePath,saveOutputFile = saveOutputFile,outputFilePath = outputFilePath,timeOffset=timeOffset))
    else:
        c.debugInfo(__name__,"Opening preprocessed data file %s"%inputFilePath)
        data_df,max_value = dsh.normalizeData(pd.read_csv(inputFilePath))
    
    train_df, test_df = dsh.splitDataToTrainAndTest(data_df,trainTestFraction)
    c.debugInfo(__name__,"train_df (%d,%d)\ttest_df (%d,%d)"%(train_df.shape[0],train_df.shape[1],test_df.shape[0],test_df.shape[1]))
    
    # first half of data is input, second half is output
    indexOutputBegin = int((data_df.shape[1])/2)
    train_input = train_df.iloc[:,0:indexOutputBegin]
    train_output = train_df.iloc[:,indexOutputBegin:data_df.shape[1]]

    test_input = test_df.iloc[:,0:indexOutputBegin]
    test_output = test_df.iloc[:,indexOutputBegin:data_df.shape[1]]

    c.debugInfo(__name__,"Making FullDataSet object containing train/test data")

    # create FullDataSet object with appropriate data
    theData = dsh.FullDataSet(trainInput = train_input.values,
                                trainOutput = train_output.values,
                                testInput = test_input.values,
                                testOutput = test_output.values)
    theData.toString()
    return theData


def makeData_julyOneWeek2015(inputFilePath):

    '''
        Args :
            inputFilePath :  Path to csv file 26_8_16_PZS_Belgugn_All_Wide_NanOmitec.csv or similar
        
        Return :
            theData :   FullDataSet object from dataset_helpers containing two DataSet 
                                objects containing two numpy arrays(input/target)
    '''
    all_data = pd.read_csv(inputFilePath,sep=",")
    print(all_data.shape)
    data_df = all_data.iloc[:,2:all_data.shape[1]]
    max_value = np.amax(data_df.values)
    data_df = ((data_df/max_value)*.99) + 0.01
    train_df, test_df = dsh.splitDataToTrainAndTest(data_df,0.8)

    index_of_Output = 0
    train_df_toUse = train_df.drop(train_df.columns[[index_of_Output]],axis=1)
    test_df_toUse = test_df.drop(test_df.columns[[index_of_Output]],axis=1)
    test_output_df = test_df.iloc[:,index_of_Output]
    train_output_df = train_df.iloc[:,index_of_Output]
    
    theData = dsh.FullDataSet(trainInput = train_df_toUse.values,
                            trainOutput = train_output_df.reshape(-1,1),
                            testInput = test_df_toUse.values,
                            testOutput = test_output_df.reshape(-1,1))
    theData.toString()

    return theData