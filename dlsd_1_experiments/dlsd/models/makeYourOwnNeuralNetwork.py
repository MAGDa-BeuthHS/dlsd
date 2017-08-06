import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        self.activationfunction = lambda x:scipy.special.expit(x)

        pass
    
    def train(self,inputs_list,targets_list):
        inputs=np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
     
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activationfunction(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activationfunction(final_inputs)
        

        output_errors = targets-final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)

        self.who += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))
        
    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activationfunction(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activationfunction(final_inputs)
        return final_outputs
   


