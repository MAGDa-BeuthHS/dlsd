import datetime
import numpy as np
import pandas as pd
from dlsd import debugInfo

'''
    Alex Hartenstein 12/04/2017
'''

def fill_time_gaps(orig_df, time_string = '%Y-%m-%d %H:%M:%S.%f'):
	'''
		Public method for filling time gaps
		Params
			orig_df : panda data frame containing wide data (one row per time stamp, one column per sensor)
			time_string : format that time stamp is to be parsed with. some differences exist (no millisecond etc)
		Returns
			[new_df, new_time_stamps]
	'''
	
	orig_time_stamps = get_orig_time_stamps_as_datetime_objects(orig_df, time_string)

	if count_gaps(orig_time_stamps) is 0:
		debugInfo(__name__,"No time gaps found")
		return orig_df

	debugInfo(__name__,"Time gaps found, beginning to fill")

	return fill_gaps(orig_df, orig_time_stamps)


#	------------------------------------------------------------


def get_orig_time_stamps_as_datetime_objects(orig_df, time_string):

	orig_time_stamps = orig_df.index.values
	for i in range(0,orig_time_stamps.shape[0]):
	    orig_time_stamps[i] = datetime.datetime.strptime(orig_time_stamps[i], time_string)
	
	return orig_time_stamps	


def fill_gaps(orig_df, orig_time_stamps):

	new_df, new_time_stamps = create_empty_variables(orig_df, orig_time_stamps)
	new_df, new_time_stamps, new_i = fill_first_row(orig_df, new_df, new_time_stamps, orig_time_stamps)
	new_df, new_time_stamps = fill_table(orig_df, new_df, new_time_stamps, orig_time_stamps, new_i)
	new_df = label_table(orig_df, new_df, new_time_stamps)
	check(new_time_stamps)

	return new_df, new_time_stamps


#	------------------------------------------------------------	


def create_empty_variables(orig_df, orig_time_stamps):
	
	total_gap_size = get_total_length_of_all_gaps(orig_time_stamps)
	new_df = pd.DataFrame(np.empty([orig_df.shape[0]+total_gap_size,orig_df.shape[1]]))
	new_df.iloc[:]=np.NAN
	new_time_stamps = []
	
	return new_df, new_time_stamps


def fill_first_row(orig_df, new_df, new_time_stamps, orig_time_stamps):
	
	new_time_stamps.append(orig_time_stamps[0])
	new_df.iloc[0] = orig_df.iloc[0].values
	new_i = 1

	return new_df, new_time_stamps, new_i


def fill_table(orig_df, new_df, new_time_stamps, orig_time_stamps, new_i):
	
	for i in range(1,orig_time_stamps.shape[0]):
	    if(orig_time_stamps[i].minute is not orig_time_stamps[i-1].minute+1):
	        if(orig_time_stamps[i].minute ==0 and orig_time_stamps[i-1].minute==59):
	            new_i = no_gap_found(orig_df, orig_time_stamps, new_time_stamps, new_df, new_i, i)
	        else:
	            new_i = gap_found(orig_df, orig_time_stamps, new_time_stamps, new_df, new_i,i)
	    else:
	        new_i = no_gap_found(orig_df, orig_time_stamps, new_time_stamps, new_df, new_i, i)

	return new_df, new_time_stamps


def label_table(orig_df, new_df, new_time_stamps):
	
	new_df.index = new_time_stamps
	new_df.columns = orig_df.columns.values
	
	return new_df


def check(new_time_stamps):

	if count_gaps(new_time_stamps) is 0:
		debugInfo(__name__,"Time gaps successfully filled")  
	else:
		raise Exception("%d Gaps found!!!"%count)

#	------------------------------------------------------------	


def no_gap_found(orig_df, orig_time_stamps, new_time_stamps, new_df, new_i, i):

    new_time_stamps.append(orig_time_stamps[i])
    new_df.iloc[new_i,] = orig_df.iloc[i,].values

    return new_i + 1


def gap_found(orig_df, orig_time_stamps, new_time_stamps, new_df, _new_i,_i):

    date_start = orig_time_stamps[_i-1]
    date_end = orig_time_stamps[_i]

    date_next, delta = increment_time_get_delta(date_start,date_end)

    while(delta.seconds > 0):
        new_time_stamps.append(date_next)
        date_next, delta = increment_time_get_delta(date_next,date_end)
        _new_i = _new_i + 1
    new_time_stamps.append(date_next)
    new_df.iloc[_new_i,] = orig_df.iloc[_i,].values

    return _new_i + 1


#	------------------------------------------------------------	


def increment_time_get_delta(_current_date_next,_date_end):
	_date_next = _current_date_next + datetime.timedelta(0,60)
	return _date_next, _date_end - _date_next


def count_gaps(_orig_time_stamps):
	_count = 0
	for i in range(1,len(_orig_time_stamps)):
	    if(_orig_time_stamps[i].minute is not _orig_time_stamps[i-1].minute+1):
	        if(_orig_time_stamps[i].minute ==0 and _orig_time_stamps[i-1].minute==59):
	            pass
	        else:
	            _count = _count+1
	return _count


#	------------------------------------------------------------	


def get_total_length_of_all_gaps(orig_time_stamps):

	idxs_gaps = get_indices_where_gaps_begin(orig_time_stamps)

	# Need to know how large new matrix is : count size of all gaps
	total_gap_size = 0
	for i in range(len(idxs_gaps)):
	    #for gap_start in idxs_gaps:
	    date_start = orig_time_stamps[idxs_gaps[i]]
	    date_end = orig_time_stamps[idxs_gaps[i]+1]

	    # add a single next minute
	    date_next = date_start
	    date_next = date_start + datetime.timedelta(0,60)
	    delta = date_end - date_next

	    while(delta.seconds is not 60): 
	        total_gap_size = total_gap_size + 1
	        date_next = date_next + datetime.timedelta(0,60)
	        delta = date_end - date_next
	    total_gap_size = total_gap_size + 1

	return total_gap_size

def get_indices_where_gaps_begin(orig_time_stamps):

	idxs_gaps = []
	for i in range(1,orig_time_stamps.shape[0]):
	    if(orig_time_stamps[i].minute is not orig_time_stamps[i-1].minute+1):
	        if not (orig_time_stamps[i].minute ==0 and orig_time_stamps[i-1].minute==59):
	            idxs_gaps.append(i-1)

	return idxs_gaps




