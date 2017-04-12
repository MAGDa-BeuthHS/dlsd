from dlsd.dataset.makeAverageWeek import make_average_week
import pandas as pd
filename = '/Users/ahartens/Desktop/Work/pzs_oneYear_belegung.csv'

df_avg_pd = make_average_week(filename)

df_avg_pd.to_csv('/Users/ahartens/Desktop/Work/pzs_oneYear_belegung_averageWeek2.csv',index=True,header=True)