import logging

import numpy as np
import pandas as pd


class Inefficient_Sensors_Remover:
    def remove_inefficient_sensors(self, df, efficiency_threshold=1.0):
        efficient_sensors_idxs = self._get_efficient_sensors_idxs(df, efficiency_threshold)
        new_df = df.iloc[:, efficient_sensors_idxs[0]]
        new_df.columns = df.columns.values[efficient_sensors_idxs]
        new_df.index = df.index.values
        logging.info("Removed %d inefficient sensors" % (df.shape[1] - new_df.shape[1]))
        return new_df

    def get_efficiency_df(self):
        return self.efficiency_df

    def _get_efficient_sensors_idxs(self, df, efficiency_threshold):
        self.efficiency_df = self._calc_sensor_efficiency(df)
        idxs = np.where(self.efficiency_df.iloc[1, :].values >= efficiency_threshold)
        return idxs

    def _calc_sensor_efficiency(self, df):
        '''
            Creates a dataframe with sensor efficiency. 
            One sensor per column, NaN count in top row, efficiency in row below it
        '''
        counts = self._count_NaNs_per_sensor(df)
        sensors_efficiency_table = pd.DataFrame(np.zeros((2, counts.shape[0])))
        sensors_efficiency_table.columns = df.columns.values
        sensors_efficiency_table.iloc[1, :] = 1 - counts.reshape(1, -1) / (df.shape[0])
        sensors_efficiency_table.iloc[0, :] = counts.reshape(1, -1)
        sensors_efficiency_table.index = ["NaN count", "efficiency"]
        return sensors_efficiency_table

    def _count_NaNs_per_sensor(self, df):
        counts = np.zeros((df.shape[1], 1))
        for i in range(0, df.shape[1]):
            counts[i] = len(np.where(np.isnan(df.iloc[:, i]))[0])
        return counts
