# Alex Hartenstein 2017

import os
import pandas as pd
import numpy as np

def main():
	path_dir = '/hartensa/Repair/fixed'
	path_output = '/hartensa/Repair/all_fixed.csv'
	files = next(os.walk(path_dir))[2]
	all_data = make_empty_array(files, path_dir)
	all_headers = []
	for i in range(len(files)):
		file = files[i]
		all_data.iloc[:,i] = read_data_file_in_dir(file,path_dir)
		all_headers.append(file.replace(".csv",""))

	all_data.columns = all_headers
	print(all_headers)
	all_data.to_csv(path_output)

def make_empty_array(files, path_dir):
	first_file_contents = pd.read_csv(os.path.join(path_dir, files[0]), header=None)
	number_rows = first_file_contents.shape[0]
	df = pd.DataFrame(np.zeros([number_rows,len(files)]))
	df.index = first_file_contents.iloc[:,1] # time is in column idx 1
	return df

def read_data_file_in_dir(file, path_dir):
	path_file = os.path.join(path_dir, file)
	data = pd.read_csv(path_file, header=None)
	return data.iloc[:,3].values


if __name__ == '__main__':
	main()
