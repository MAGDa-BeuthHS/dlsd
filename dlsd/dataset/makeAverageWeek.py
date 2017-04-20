import pandas as pd
import numpy as np
from .dataset_sqlToNumpy import pivot_simple
from .fillTimeGaps import fill_time_gaps
from dlsd import debugInfo
import datetime


def make_average_week(filename, sql_headers = ['S_IDX','ZEIT','WERT'], time_string = '%Y-%m-%d %H:%M:%S.%f'):
	'''
		filename :		File path to csv file (sql output with 3 columns)
	'''

	debugInfo(__name__,"making average week : preparing data")
	
	df_pd, new_row_names = fill_time_gaps(pivot_simple(filename,None,sql_headers = sql_headers),time_string)

	df = df_pd.values
	
	length_week = 7*1440
	num_sensors = df.shape[1]
	num_weeks = df.shape[0]/length_week

	df_avg = np.zeros([length_week,num_sensors])

	debugInfo(__name__,"Data successfully prepared, finding average of %d weeks"%num_weeks)

	for time_in_week in range(0,length_week):
	    # get indices of all rows corresponding to a certain time of the day/week 
	    idxs = [(length_week*week_idx)+time_in_week for week_idx in range(0,int(num_weeks))]
	    # (eg monday 00:02) is equal to the average of every monday at 00:02
	    df_avg[time_in_week] = np.nanmean(df[idxs],0)

	df_avg_pd = pd.DataFrame(df_avg)

	avg_row_names = [datetime.datetime.strftime(i, '%H:%M:%S') for i in new_row_names[0:df_avg.shape[0]]]
	
	df_avg_pd.index = avg_row_names
	df_avg_pd.columns = df_pd.columns.values
	
	return df_avg_pd





