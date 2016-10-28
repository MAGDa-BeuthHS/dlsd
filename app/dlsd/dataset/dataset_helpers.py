import numpy as np
from dlsd import Common as c

'''
    dataset_helpers

    TensorFlow requires that data be fed into placeholders during a session.run() call
    This is done using feed dictionaries, a dictionary mapping from a numpy array to 
    a tf.Placeholder 

    This module has classes to wrap data in a way that easily allows for the filling of 
    a feed dict.

    Also contains function to split data into a training/test set 

    Alex Hartenstein 14/10/2016

'''


def splitDataToTrainAndTest(data_df,train_frac):
    '''
        @   param data_df       Pandas dataframe object of all data, each row is data point
        @   param train_frac    Float determining how much reserved for training
        
        @   return  train,test  Two pandas dataframes                 
    '''
    c.debugInfo(__name__,"Splitting data to train and test fraction %.2f"%(train_frac))
    train = data_df.sample(frac=train_frac,random_state=1)
    test = data_df.loc[~data_df.index.isin(train.index)]
    return train,test

def normalizeData(data_df):
    max_value = np.amax(data_df.values)
    c.debugInfo(__name__,"Max value in maxMinNormalization is %d"%max_value)
    return ((data_df/max_value)*.99999999) + 0.00000001, max_value

def denormalizeData(data_df,max_value):
    return ((data_df - 0.00000001)/.99999999)*max_value


class FullDataSet:
    '''
        Wrapper for a training and test dataset
        Contains two 'DataSet' objects, one for training test respectively
        Dataset objects then each contain input/output data
    '''
    def __init__(self, trainInput, trainOutput, testInput, testOutput):
        self.test = DataSet()
        self.train = DataSet()
        self.train.inputData = trainInput
        self.train.outputData = trainOutput
        self.test.inputData = testInput
        self.test.outputData = testOutput
        self.max_value = 0

    def getNumberInputs(self):
        return self.test.inputData.shape[1]
    def getNumberOutputs(self):
        return self.test.outputData.shape[1]
    def getNumberTrainingPoints(self):
        return self.train.inputData.shape[0]
    def getNumberTestPoints(self):
        return self.test.inputData.shape[0]
    def toString(self):
        c.debugInfo(__name__,"FullDataSet Object : [ Train : input (%d, %d)  output (%d, %d) ]\t [ Test : input (%d, %d)  output (%d, %d) ]"%(
            self.train.inputData.shape[0],
            self.train.inputData.shape[1],
            self.train.outputData.shape[0],
            self.train.outputData.shape[1],
            self.test.inputData.shape[0],
            self.test.inputData.shape[1],
            self.test.outputData.shape[0],
            self.test.outputData.shape[1]))

class DataSet:
    '''
        Wrapper for a single input/output numpy array of values
        Call 'next_batch' to get a batch of values
    '''
    def __init__(self):
        self.inputData = []
        self.outputData = []
    def next_batch(self,batch_size):
        b_in = self.inputData[np.random.choice(self.inputData.shape[0],batch_size,replace=False),:]
        b_out = self.outputData[np.random.choice(self.outputData.shape[0],batch_size, replace=False),:]
        return b_in,b_out
    def num_examples(self):
        return self.inputData.shape[0]
