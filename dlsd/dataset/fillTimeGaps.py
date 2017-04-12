import datetime
import numpy as np
import pandas as pd
from dlsd import debugInfo

'''
    fillTimeGaps

    Alex Hartenstein 12/04/2017

'''

def fill_time_gaps(df_pd, time_string = ):

	## Functions ##
	
	def increment_time_get_delta(_current_date_next,_date_end):
		_date_next = _current_date_next + datetime.timedelta(0,60)
		return _date_next, _date_end - _date_next

	def no_gap_found(_new_i):
	    new_row_names.append(row_names[i])
	    new_df.iloc[new_i,] = df_pd.iloc[i,].values
	    return _new_i + 1

	def gap_found(_new_i,_i):
	    print("Found a gap, filling")
	    date_start = row_names[_i-1]
	    date_end = row_names[_i]

	    date_next, delta = increment_time_get_delta(date_start,date_end)

	    while(delta.seconds > 0):
	        new_row_names.append(date_next)
	        date_next, delta = increment_time_get_delta(date_next,date_end)
	        _new_i = _new_i + 1
	    new_row_names.append(date_next)
	    new_df.iloc[_new_i,] = df_pd.iloc[_i,].values
	    return _new_i + 1

	def count_gaps(_row_names):
		_count = 0
		for i in range(1,len(_row_names)):
		    if(_row_names[i].minute is not _row_names[i-1].minute+1):
		        if(_row_names[i].minute ==0 and _row_names[i-1].minute==59):
		            pass
		        else:
		            _count = _count+1
		return _count
	

	## Code ##

	# Get all row names as a python date time object
	row_names = df_pd.index.values

	# convert every row name time stamp to a datetime object
	for i in range(0,row_names.shape[0]):
	    row_names[i] = datetime.datetime.strptime(row_names[i], time_string)

	# if there are no gaps, return original df
	if count_gaps(row_names) is 0:
		debugInfo(__name__,"No time gaps found")
		return df_pd

	# get indices where gaps begin
	idxs_gaps = []
	for i in range(1,row_names.shape[0]):
	    if(row_names[i].minute is not row_names[i-1].minute+1):
	        if not (row_names[i].minute ==0 and row_names[i-1].minute==59):
	            idxs_gaps.append(i-1)

	# Need to know how large new matrix is : count size of all gaps
	total_gap_size = 0
	for i in range(len(idxs_gaps)):
	    #for gap_start in idxs_gaps:
	    date_start = row_names[idxs_gaps[i]]
	    date_end = row_names[idxs_gaps[i]+1]

	    # add a single next minute
	    date_next = date_start
	    date_next = date_start + datetime.timedelta(0,60)
	    delta = date_end - date_next

	    while(delta.seconds is not 60): 
	        total_gap_size = total_gap_size + 1
	        date_next = date_next + datetime.timedelta(0,60)
	        delta = date_end - date_next
	    total_gap_size = total_gap_size + 1


	new_df = pd.DataFrame(np.empty([df_pd.shape[0]+total_gap_size,df_pd.shape[1]]))
	new_df.iloc[:]=np.NAN

	debugInfo(__name__,"Time gaps found, beginning to fill")
	# Create empty list that will contain all row names without gaps
	new_row_names = []

	# Add first row name (because in loop require previous time step)
	new_row_names.append(row_names[0])
	new_df.iloc[0] = df_pd.iloc[0].values

	new_i = 1

	# Iterate over every row name again
	    
	for i in range(1,row_names.shape[0]):
	    if(row_names[i].minute is not row_names[i-1].minute+1):
	        if(row_names[i].minute ==0 and row_names[i-1].minute==59):
	            new_i = no_gap_found(new_i)
	        else:
	            new_i = gap_found(new_i,i)
	    else:
	        new_i = no_gap_found(new_i)

	# assign row names, headers
	new_df.index = new_row_names
	new_df.columns = df_pd.columns.values

	if count_gaps(new_row_names) is 0:
		debugInfo(__name__,"Time gaps successfully filled")  
	else:
		raise Exception("%d Gaps found!!!"%count)

	return new_df, new_row_names