import datetime
import numpy as np
import pandas as pd
from dlsd import debugInfo

'''
    Alex Hartenstein 12/04/2017
'''

class TimeGapFiller:
	def __init__(self):
		pass

	
	def fill_time_gaps(self, orig_df, time_format = '%Y-%m-%d %H:%M:%S.%f'):
		'''
			Public method for filling time gaps
			Params
				self.orig_df : panda data frame containing wide data (one row per time stamp, one column per sensor)
				self.time_format : format that time stamp is to be parsed with. some differences exist (no millisecond etc)
			Returns
				[self.new_df, self.new_time_stamps]
		'''
		self.time_format = time_format
		self.orig_df = orig_df

		self.convert_orig_time_stamps_as_datetime_objects()

		if self.count_gaps(self.orig_time_stamps) is 0:
			debugInfo(__name__,"No time gaps found")
			self.convert_datetime_objects_to_orig_time()
			return self.orig_df

		debugInfo(__name__,"Time gaps found, beginning to fill")

		return self.fill_gaps()


	#	------------------------------------------------------------


	def convert_orig_time_stamps_as_datetime_objects(self):

		self.orig_time_stamps = self.orig_df.index.values
		for i in range(0,self.orig_time_stamps.shape[0]):
		    self.orig_time_stamps[i] = datetime.datetime.strptime(self.orig_time_stamps[i],self.time_format)
		
		self.orig_df.index = self.orig_time_stamps


	def convert_datetime_objects_to_orig_time(self):

		time_stamps = self.orig_df.index.values
		for i in range(0,time_stamps.shape[0]):
		    time_stamps[i] = datetime.datetime.stfptime(time_stamps[i],self.time_format)
		
		self.orig_df.index = time_stamps


	def fill_gaps(self):

		self.create_empty_variables()
		self.fill_first_row()
		self.fill_table()
		self.label_table()
		self.check_if_has_gaps()

		return self.new_df


	#	------------------------------------------------------------	


	def create_empty_variables(self):
		
		total_gap_size = self.get_total_length_of_all_gaps()
		self.new_df = pd.DataFrame(np.empty([self.orig_df.shape[0]+total_gap_size,self.orig_df.shape[1]]))
		self.new_df.iloc[:]=np.NAN
		self.new_time_stamps = []
		self.new_i = 0


	def fill_first_row(self):
		
		self.new_time_stamps.append(self.orig_time_stamps[0])
		self.new_df.iloc[0] = self.orig_df.iloc[0].values
		self.new_i = 1


	def fill_table(self):
		
		for i in range(1,self.orig_time_stamps.shape[0]):
		    if(self.orig_time_stamps[i].minute is not self.orig_time_stamps[i-1].minute+1):
		        if(self.orig_time_stamps[i].minute ==0 and self.orig_time_stamps[i-1].minute==59):
		            new_i = self.no_gap_found(i)
		        else:
		            new_i = self.gap_found(i)
		    else:
		        new_i = self.no_gap_found(i)


	def label_table(self):
		
		self.new_df.index = self.new_time_stamps
		#self.new_df.index = [datetime.datetime.strftime(i,self.time_format) for i in self.new_time_stamps]
		self.new_df.columns = self.orig_df.columns.values
		

	def check_if_has_gaps(self):

		if self.count_gaps(self.new_time_stamps) is 0:
			debugInfo(__name__,"Time gaps successfully filled")  
		else:
			raise Exception("%d Gaps found!!!"%count)

	#	------------------------------------------------------------	


	def no_gap_found(self, i):

	    self.new_time_stamps.append(self.orig_time_stamps[i])
	    self.new_df.iloc[self.new_i,] = self.orig_df.iloc[i,].values
	    self.new_i = self.new_i + 1


	def gap_found(self, i):

	    date_start = self.orig_time_stamps[i-1]
	    date_end = self.orig_time_stamps[i]

	    date_next, delta = self.increment_time_get_delta(date_start,date_end)

	    while(delta.seconds > 0):
	        self.new_time_stamps.append(date_next)
	        date_next, delta = self.increment_time_get_delta(date_next,date_end)
	        self.new_i = self.new_i + 1
	    self.new_time_stamps.append(date_next)
	    self.new_df.iloc[self.new_i,] = self.orig_df.iloc[i,].values

	    self.new_i = self.new_i + 1


	#	------------------------------------------------------------	


	def increment_time_get_delta(self, _current_date_next, _date_end):
		_date_next = _current_date_next + datetime.timedelta(0,60)
		return _date_next, _date_end - _date_next


	def count_gaps(self, np_array):
		_count = 0
		for i in range(1,len(np_array)):
		    if(np_array[i].minute is not np_array[i-1].minute+1):
		        if(np_array[i].minute ==0 and np_array[i-1].minute==59):
		            pass
		        else:
		            _count = _count+1
		return _count


	#	------------------------------------------------------------	


	def get_total_length_of_all_gaps(self):

		idxs_gaps = self.get_indices_where_gaps_begin(self.orig_time_stamps)

		# Need to know how large new matrix is : count size of all gaps
		total_gap_size = 0
		for i in range(len(idxs_gaps)):
		    #for gap_start in idxs_gaps:
		    date_start = self.orig_time_stamps[idxs_gaps[i]]
		    date_end = self.orig_time_stamps[idxs_gaps[i]+1]

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

	def get_indices_where_gaps_begin(self, vector_of_time_stamps):

		idxs_gaps = []
		for i in range(1,vector_of_time_stamps.shape[0]):
		    if(vector_of_time_stamps[i].minute is not vector_of_time_stamps[i-1].minute+1):
		        if not (vector_of_time_stamps[i].minute ==0 and vector_of_time_stamps[i-1].minute==59):
		            idxs_gaps.append(i-1)

		return idxs_gaps




