from . import dataset_helpers as dsh
import pandas as pd
import numpy as np

'''
    dataset_generators

    This module contains functions to reproducibly create FullDataSet objects from
    source files. 

    Also contains fill_feed_dict() method to get random batches of data for training.

    Alex Hartenstein 14/10/2016

'''




'''
    @ param filePath        Path to csv file 26_8_16_PZS_Belgugn_All_Wide_NanOmited.csv or similar
    @ return theData        FullDataSet object from dataset_helpers containing two DataSet 
                            objects containing two numpy arrays(input/target)
'''
def makeData_julyOneWeek2015(filePath):
    all_data = pd.read_csv(filePath,sep=",")
    data_df = all_data.iloc[:,2:all_data.shape[1]]
    max_value = np.amax(data_df.values)
    data_df = ((data_df/max_value)*.99) + 0.01
    train_df, test_df = dsh.splitDataToTrainAndTest(data_df,0.8)

    index_of_Output = 0
    train_df_toUse = train_df.drop(train_df.columns[[index_of_Output]],axis=1)
    test_df_toUse = test_df.drop(test_df.columns[[index_of_Output]],axis=1)
    test_output_df = test_df.iloc[:,index_of_Output]
    train_output_df = train_df.iloc[:,index_of_Output]
    
    theData = dsh.FullDataSet()
    theData.setTrain_input(train_df_toUse.values)
    theData.setTest_input(test_df_toUse.values)
    theData.setTrain_output(train_output_df.values.reshape(-1,1))
    theData.setTest_output(test_output_df.values.reshape(-1,1))
    return theData

'''
    @ param data_set        a FullDataSet object from dataset_helpers 
                            containing two DataSet objects (for next_batch method)
    @ param input_pl        tensorflow Placeholder for input data
    @ param output_pl       tensorflow Placeholder for correct data
    @ param batch_size      int value determining how many rows of dataset to feed into dictionary

    @ return feed_dict      dict with placeholder:numpy array for giving to session.run() method
'''
def fill_feed_dict(data_set,input_pl, output_pl, batch_size):
    inputData,correctOutputData = data_set.next_batch(batch_size)
    feed_dict = {
        input_pl : inputData,
        output_pl : correctOutputData,
    }
    return feed_dict