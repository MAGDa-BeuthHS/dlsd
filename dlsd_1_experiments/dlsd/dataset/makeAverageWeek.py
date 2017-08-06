import pandas as pd
import numpy as np
from .dataset_sqlToNumpy import pivot_simple
from .fillTimeGaps import TimeGapFiller
from dlsd import debugInfo
import datetime


class AverageWeekMaker:
	
	def __init__(self):
		pass


	def make_average_week(self, filename, sql_headers = ['S_IDX','ZEIT','wert'], time_format = '%Y-%m-%d %H:%M:%S'):
		
		self.time_format = time_format
		self.sql_headers = sql_headers

		self.open_sql_file_and_pivot(filename)
		self.fill_time_gaps()
		self.calculate_average_week()
		self.format_week_to_start_on_monday()


	def open_sql_file_and_pivot(filename):
		
		self.df = pivot_simple(filename,None,sql_headers = self.sql_headers)
		

	def fill_time_gaps():
		
		tgf = TimeGapFiller()
		self.df = tgf.fill_time_gaps(self.df,self.time_format)


	def calculate_average_week():
		
		df = self.df.values
		length_week = 7*1440
		num_sensors = df.shape[1]
		num_weeks = df.shape[0]/length_week

		df_avg = np.zeros([length_week,num_sensors])

		debugInfo(__name__,"Data successfully prepared, finding average of %d weeks"%num_weeks)

		for time_in_week in range(0,length_week):
		    # get indices of all rows corresponding to a certain time of the day/week 
		    idxs_for_time_n = [(length_week*week_idx)+time_in_week for week_idx in range(0,int(num_weeks))]
		    # (eg monday 00:02) is equal to the average of every monday at 00:02
		    df_avg[time_in_week] = np.nanmean(df[idxs_for_time_n],0)

		self.df = pd.DataFrame(df_avg)


	def format_week_to_start_on_monday():

	def make_average_week_start_on_day():
		n=3
		
		n_begin = n*self.length_day

		week_starting_on_n = pd.DataFrame(np.zeros([self.length_week,awm.df.shape[1]]))
		week_starting_on_n.iloc[0:self.length_week-n_begin,]=standardized_avg_week.iloc[n_begin:self.length_week,].values
		week_starting_on_n.iloc[self.length_week-n_begin:self.length_week,]=standardized_avg_week.iloc[0:n_begin,].values






#def make_average_week(filename, sql_headers = ['S_IDX','ZEIT','sert'], time_string = '%Y-%m-%d %H:%M:%S.%f'):

def make_average_week(filename, sql_headers = ['S_IDX','ZEIT','wert'], time_format = '%Y-%m-%d %H:%M:%S'):
	'''
		filename :		File path to csv file (sql output with 3 columns)
	'''

	self.time_format = time_format
	self.sql_headers = sql_headers
	debugInfo(__name__,"making average week : preparing data")
	

	df_pd = 

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

	# day of week as integer with sunday being 1
	avg_row_names = [datetime.datetime.strftime(i, '%w_%H:%M:%S') for i in new_row_names[0:df_avg.shape[0]]]
	
	df_avg_pd.index = avg_row_names
	df_avg_pd.columns = df_pd.columns.values
	
	return df_avg_pd


def get_first_day_int():
	first_day_time_stamp = df.index.values[0]
	print(first_day_time_stamp)


def get
