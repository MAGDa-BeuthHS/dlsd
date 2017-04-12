from dlsd.dataset import dataset_sqlToNumpy as stn
import numpy as np
import pandas as pd
import argparse

'''
	Use to turn a sql wide table to a formatted input table for machine learning with one 
	row per time point and one column per sensor.

	Example Usage : 
		docker exec -i hartenstein-tensorflow python /data/local/home/hartenstein/synchedCode_all/dlsd/convertSQLtoDLSDinput.py -i=/data/local/home/hartenstein/data_sql/pzs_2015_occupancy.csv -o=/data/local/home/hartenstein/data_formatted/pzs_2015_occupancy_formatted_twoSensors.csv -s=285,286 -he=ZEIT,S_IDX,WERT

'''
def getArguments():
	parser = argparse.ArgumentParser(description="Convert sql output to a wide table")
	parser.add_argument('-i','--inputFile',help='Path to sql output file',required=True)
	parser.add_argument('-o','--outputFile',help='Path to write output to',required=True)
	parser.add_argument('-s','--specifiedSensors',help='list of sensors to be extracted separated by commas',required=False)
	parser.add_argument('-he','--headers',help='headers of sql output file column headers separated by commas in order time,id,value')
	args = parser.parse_args()
	return args

def main():
	args = getArguments()

	# 
	wide_data = stn.pivotAndSmooth(args.inputFile,
		None if args.specifiedSensors is None else pd.DataFrame(args.specifiedSensors.split(","),dtype=int),
		headers = args.headers.split(","))[0]

	# write data to file
	wide_data.to_csv(args.outputFile,index=False,header=True)
	

if __name__=="__main__":
	main()