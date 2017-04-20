from dlsd.dataset.makeAverageWeek import make_average_week
import pandas as pd

filename = '/Users/ahartens/Desktop/Work/pzs_oneYear_belegung.csv'

# get the average week for year 2015
df_avg_pd = make_average_week(filename)

# print it out
df_avg_pd.to_csv('/Users/ahartens/Desktop/Work/pzs_oneYear_belegung_averageWeek2.csv',index=True,header=True)