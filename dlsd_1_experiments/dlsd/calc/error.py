import numpy as np

def mape(data):
	'''
		Calculates mean average percentage error
		Args : 
			data : numpy array containing actual in first half of columns and target in second half
		Returns :
			mape : numpy array with mape

	'''
	n = data.shape[0]
	numOutputs = int(data.shape[1]/2)
	actuals = data[:,0:numOutputs]
	predicts = data[:,numOutputs:data.shape[1]]
	mape = (100/float(n))*(np.sum(np.abs((actuals-predicts)/actuals),axis=0))	
	return mape

def mae(prediction, target):
	mae = np.mean(np.absolute(prediction - target),axis=0)
	return mae