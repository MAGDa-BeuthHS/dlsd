import sys, getopt
import argparse
import os
import pandas as pd
from dlsd.calc import error

'''
	Calculates Mean Average Percent Error for each sensor in following directory structure : 

	root 
	|__	s_1
	|	|__	** F2 : Avg of avg mapes for all test months
	|	|__	test_month_1
	|	|	|__ ** F1 : Avg mape for month 1
	|	|	|__ output
	|	|	
	|	|__ test_month_2
	|		|__ ** F1 : Avg mape for month 2
	|		|__ output	
	|
	|__	s_2 ..

	Where output contains the output of a machine learning algorithm, 
		- actual values in first 1/2 of columns, 
		- predicted value in second half
	For each sensor, 
		1. calculates Mape for each test month
		2. then calculates average of averages (avg mapes over months)

'''

def getArguments():
	parser = argparse.ArgumentParser(description='Calculate Mape')
	parser.add_argument('-i','--inputDir',help='Path to directory containg predictions_ files',required=True)
	parser.add_argument('-o','--outputFile',help='Path to output file containg calculatesd Mapes',required=True)
	args = parser.parse_args()
	return args

def main():
	args = getArguments()

	sensorsInfo = {}
	
	# create F1 , saving relevant data for creating F2
	for dirname,dirlist,filelist in os.walk(args.inputDir):
	    
	    if ("output" in dirname):
	    	parentDir = os.path.split(dirname)[:-1][0]
	    	sensorIdDir = os.path.split(parentDir)[:-1][0]
	    	sensorIdDirName = os.path.split(sensorIdDir)[-1]
	    	sensorId = sensorIdDirName[sensorIdDirName.index("_")+1:]
	    	
	    	if sensorsInfo.get(sensorId) is None : 
	    		sensorsInfo[sensorId] = {"path":sensorIdDir,"mapes":[]}
	    	mapesForDir = []
	    	headers = None
	    	indices = []

	    	for filename in os.listdir(dirname):
	    		data = pd.read_csv(os.path.join(dirname,filename),index_col=0,header=0)
	    		if (headers is None): headers = data.columns.values[0:int(data.shape[1]/2)]
	    		indices.append(filename.replace(".csv","").replace("predictions_",""))
	    		mapesForDir.append(error.mape(data.values))

	    	mapesForDir = pd.DataFrame(mapesForDir)
	    	mapesForDir.columns = headers
	    	mapesForDir.index = indices
	    	mapesForDir.to_csv(os.path.join(parentDir,"avgMapesForSensor_"+sensorId+".csv"),index=True,header=True)
	    	sensorsInfo[sensorId]['mapes'].append(mapesForDir)
    

	# create F2
	for key in sensorsInfo:
		avgMapeOfMonths = sensorsInfo[key]['mapes'][0]
		for i in range(1,len(sensorsInfo[key]['mapes'])):
			avgMapeOfMonths = avgMapeOfMonths + sensorsInfo[key]['mapes'][i]
		avgMapeOfMonths = avgMapeOfMonths/len(sensorsInfo[key]['mapes'])
		avgMapeOfMonths.to_csv(os.path.join(sensorsInfo[key]['path'],"allMapesForSensor_"+key+".csv"),header=True,index=True)

if __name__ == "__main__":
	main()

