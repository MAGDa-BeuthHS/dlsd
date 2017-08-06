import os
import pandas as pd
import numpy as np
import argparse

'''
	eg docker exec -i hartenstein-tensorflow python /data/local/home/hartenstein/synchedCode/dlsd/getTopMethod.py -i /data/local/home/hartenstein/analysis_16_12_12/ -o /data/local/home/hartenstein/analysis_16_12_12/topMinMaxDataPreps_byMape.csv -f allMapesForSensor
'''
def getArgs():
	parser = argparse.ArgumentParser(description='Get the best and worse prediction method')
	parser.add_argument('-i','--inputDir',help='Path to directory containg predictions_ files',required=True)
	parser.add_argument('-o','--outputFile',help='Path to output file containg calculatesd Mapes',required=True)
	parser.add_argument('-f','--fileHandle',help='File name handle to look for, eg avgMaesForSensor',required=True)
	args = parser.parse_args()
	return args

def main():
	args = getArgs()
	mins = []
	maxs = []
	sens = []
	for dirname,dirlist,filelist in os.walk(args.inputDir):
	    for file in filelist:
	        if args.fileHandle in file:
	            data = pd.read_csv(os.path.join(dirname, file),header=0,index_col=0)
	            mins.append(data.idxmin(0))
	            maxs.append(data.idxmax(0))
	            sens.append(file.replace(args.fileHandle,"").replace(".csv",""))
	df_min = pd.DataFrame(np.vstack(mins))
	df_min.index = sens
	df_min.columns = data.columns

	df_max = pd.DataFrame(np.vstack(maxs))
	df_max.index = sens
	df_max.columns = data.columns


	topMin = np.array([(lambda x:df_min.iloc[:,x].astype('category').describe()[2])(col) for col in range(0,df_min.shape[1])])
	topMax = np.array([(lambda x:df_max.iloc[:,x].astype('category').describe()[2])(col) for col in range(0,df_max.shape[1])])
	freq_min = np.array([(lambda x:df_min.iloc[:,x].astype('category').describe()[3])(col) for col in range(0,df_min.shape[1])])
	freq_max = np.array([(lambda x:df_max.iloc[:,x].astype('category').describe()[3])(col) for col in range(0,df_max.shape[1])])


	df = pd.DataFrame([topMin,freq_min,topMax,freq_max])
	df.index = ['top_min','freq_min','top_max','freq_max']

	df.columns = data.columns
	df.to_csv(args.outputFile,header=True,index=True)

if __name__=="__main__":
	main()