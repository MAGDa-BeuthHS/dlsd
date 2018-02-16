import analysis2 as a
from dlsd import Common as c
import os
import pandas as pd
import numpy as np
SQL_TRAIN = '/Users/ahartens/Desktop/Work/24_10_16_PZS_Belegung_oneMonth.csv'
SQL_TEST = '/Users/ahartens/Desktop/Work/16_11_11_PZS_Belegung_augustLimited.csv'
O___PATH_OUTPUTDIR ='/Users/ahartens/Desktop/Experiment'
SS_PATH_SENSORLIST='sensorsUsedForTraining.csv'

'''
	Used with analysis2.py : 
	- Input : All (efficient) sensors (69 sensors)
	- Output : Single sensor, in this case sensor 182

	Create a table of mean average error (MAE of sensor 182 abs(predicted - actual))
	varying :
	1. hidden layer size
	2. timeOffset
'''
def getMAE(fileName):
    predictions = pd.read_csv(fileName)
    return np.mean(np.absolute(predictions.iloc[:,0]-predictions.iloc[:,1]))

def main():
	args = c.makeCommandLineArgs()

	args.verbose = True

	# during training write sensors used to this path, during testing read from it
	args.specifiedSensors = SS_PATH_SENSORLIST

	# never save prepared SQL to Numpy data to file (constantly remade)
	args.preparedData = None

	# desired values
	hiddenSize = [10,50,70,100,150]
	timepoints = [5,10,20,30,40,60,90,120]

	# output table
	results = pd.DataFrame(np.zeros((len(hiddenSize),len(timepoints))),columns=timepoints,index=hiddenSize)

	for i in range(0,len(hiddenSize)):
		size = hiddenSize[i]
		c.debugInfo(__name__,"Hidden size : %d"%size)
		a.Configuration.n_hidden = size
		for j in range(0,len(timepoints)):
			offset = timepoints[j]
			c.debugInfo(__name__,"TimeOffset : %d"%offset)
			
			'''
				Train the network : dataset created from scratch from sql output file
			'''
			args.restoreSess = None
			args.pathToSQLFile = SQL_TRAIN
			a.Configuration.timeOffset = offset

			a.main(args)
			'''
				Test the previously trained network on new data
				in this case test network trained on july on data from august
			'''
			path_predictionsOutput = os.path.join(O___PATH_OUTPUTDIR,"prediction_h%d_t%d.csv"%(size,offset))
			args.restoreSess = True
			args.pathToSQLFile = SQL_TEST
			args.predictionOutput = path_predictionsOutput

			a.main(args)

			'''
				Calculate mean average error of Test output and save in table
			'''
			mae = getMAE(path_predictionsOutput)
			results.iloc[i,j] = mae
			c.debugInfo(__name__,"The MAE is %f"%mae)

	# print out results table
	results.to_csv(os.path.join(args.outputDirectory,"allResults.csv"))


if __name__ == "__main__":
	main()

