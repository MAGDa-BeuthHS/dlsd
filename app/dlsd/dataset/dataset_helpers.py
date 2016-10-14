import numpy as np


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



'''
    @   param data_df       Pandas dataframe object of all data, each row is data point
    @   param train_frac    Float determining how much reserved for training
    
    @   return  train,test  Two pandas dataframes                 
'''
def splitDataToTrainAndTest(data_df,train_frac):
    train = data_df.sample(frac=train_frac,random_state=1)
    test = data_df.loc[~data_df.index.isin(train.index)]
    return train,test

'''
    Wrapper for a training and test dataset
    Contains two 'DataSet' objects, one for training test respectively
    Dataset objects then each contain input/output data
'''
class FullDataSet:
    def __init__(self):
        self.test = DataSet()
        self.training = DataSet()
    def setTrain_input(self,data):
        self.training.inputData = data
    def setTrain_output(self,data):
        self.training.outputData = data
    def setTest_input(self,data):
        self.test.inputData = data
    def setTest_output(self,data):
        self.test.outputData = data
    def getNumberInputs(self):
        return self.test.inputData.shape[1]
    def getNumberOutputs(self):
        return self.test.outputData.shape[1]

'''
    Wrapper for a single input/output numpy array of values
    Call 'next_batch' to get a batch of values
'''
class DataSet:
    def __init__(self):
        self.inputData = []
        self.outputData = []
    def next_batch(self,batch_size):
        b_in = self.inputData[np.random.choice(self.inputData.shape[0],batch_size,replace=False),:]
        b_out = self.outputData[np.random.choice(self.outputData.shape[0],batch_size, replace=False),:]
        return b_in,b_out
    def num_examples(self):
        return self.inputData.shape[0]