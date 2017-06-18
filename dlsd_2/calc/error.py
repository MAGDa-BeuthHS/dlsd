import numpy as np

def mae(prediction, target):
	''' 
		mean average error 
	'''
	return np.mean(np.absolute(prediction - target),axis=0)

def mape(prediction,target):
	''' 
		mean average percentage error 
	'''
	n = target.shape[0]
	return (100/float(n))*(np.sum(np.abs((target-prediction)/target),axis=0))