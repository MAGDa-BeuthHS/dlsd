from dlsd.dataset import dataset_sqlToNumpy as stn
import numpy as np
import pandas as pd
import argparse

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



	wide_data = stn.pivotAndSmooth(args.inputFile,
		None if args.specifiedSensors is None else pd.DataFrame(args.specifiedSensors.split(","),dtype=int),
		headers = args.headers.split[","])[0]
	wide_data.to_csv(args.outputFile,index=False,header=True)
	

if __name__=="__main__":
	main()