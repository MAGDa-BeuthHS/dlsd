import pandas as pd
import numpy as np

from dlsd import debugInfo

def pivotAndSmooth(inputFile,specifiedSensors,sensorsOutputPath=None,sensorEfficiency=.999,window = 15,sql_headers=None):
    '''
        First step of all further analyses :
        Long/narrow dataset from SQL is made wide (one column per sensor, time stamps are rows)
        Format of input file must be [S_IDX,ZEIT,wert]
        Args : 
            inputFile :         Path to csv file generated by sql
            specifiedSensors :  numpy array of sensor Ids that should be used. If present then inefficients sensors not removed
            sensorEfficiency :  cutoff for which sensors are removed. If specifiedSensors not None, this is irrelevant
        Return :
            data_wide :         Pandas dataframe containing desired data
    '''
    all_data = pd.read_csv(inputFile,sep=",")
    debugInfo(__name__,"Read input SQL file with shape : (%d, %d)"%(all_data.shape[0],all_data.shape[1]))


    if (specifiedSensors is not None):
        debugInfo(__name__,"%d Sensors specified, getting indices from"%specifiedSensors.shape[0])
        sensorIndices = np.where(all_data.iloc[:,0].values==specifiedSensors.values)[1]
        all_data = all_data.iloc[sensorIndices,:]

    if sql_headers is None : sql_headers = ['S_IDX','ZEIT','wert']
    
    # make into a wide table
    data_wide_all = all_data.pivot(index=sql_headers[1], columns=sql_headers[0], values=sql_headers[2])
    debugInfo(__name__,"Pivoted input shape : (%d, %d)"%(data_wide_all.shape[0],data_wide_all.shape[1]))

    # do the rolling mean
    data_wide_all = data_wide_all.rolling(window,min_periods =1).mean()
    debugInfo(__name__,"Calculated the rolling average using window %d : (%d, %d)"%(window,data_wide_all.shape[0],data_wide_all.shape[1]))

    # make table containing only efficient sensors (only columns with efficiency > sensorEfficiency nan are used)
    if (specifiedSensors is None):
        data_wide = removeInefficientSensors(data_wide_all,sensorEfficiency)
        specifiedSensors = pd.DataFrame(data_wide.columns.values.reshape((data_wide.columns.values.shape[0],-1)))

        # print out sensors used for this analysis to file as long list. Used later for specifying sensors
        if (sensorsOutputPath is not None):
            debugInfo(__name__,"Saving Sensors list to %s"%(sensorsOutputPath))
            specifiedSensors.to_csv(sensorsOutputPath,index=False,header=False)
    else:
        debugInfo(__name__,"Sensors list is specified : not resaving sensors")
        data_wide = data_wide_all

    return data_wide, data_wide_all.shape[0], specifiedSensors

def removeInefficientSensors(data_wide_all,sensorEfficiency):
    #count the number of times each column has an 'na' value
    counts = np.zeros((data_wide_all.shape[1],1))
    print(data_wide_all.shape)
    for i in range(0,data_wide_all.shape[1]):
        counts[i] = len(np.where(np.isnan(data_wide_all.iloc[:,i]))[0])

    # calculate the efficiency of the sensor
    sensorsToEfficiency = pd.DataFrame(np.zeros((3,counts.shape[0])))
    sensorsToEfficiency.iloc[0,:]=data_wide_all.columns.values.reshape(1,-1)
    sensorsToEfficiency.iloc[2,:]=1-counts.reshape(1,-1)/(data_wide_all.shape[0])
    sensorsToEfficiency.iloc[1,:]=counts.reshape(1,-1)
    
    # make table containing only efficient sensors (only columns with <10 nan are used)
    efficientSensorIndices = np.where(sensorsToEfficiency.iloc[2,:].values>sensorEfficiency)
    data_wide = data_wide_all.iloc[:,efficientSensorIndices[0]]
    data_wide.shape
    debugInfo(__name__,"Data where sensors have efficiency > %.2f : (%d, %d)"%(sensorEfficiency,data_wide.shape[0],data_wide.shape[1]))
    debugInfo(__name__,"There are %d sensors in total, but only %d have efficiency > %.2f"%(data_wide_all.shape[1], data_wide.shape[1],sensorEfficiency))
    return data_wide


def getAverageWeek(inputFile,sql_headers = None):
    '''
        Creates an average week given
        Params
            df :    path to an SQL file
        Return
    '''
    # create wide data set (one column per sensor)
    df = pivotInefficiently(inputFile,None,sql_headers = sql_headers)[0].values
    
    # define values
    length_week = 7*1440
    num_sensors = df.shape[1]
    num_weeks = df.shape[0]/length_week

    # empty matrix for values
    df_avg = np.zeros([length_week,num_sensors])

    # iterate over time point within a single week and get average value
    for time_in_week in range(0,length_week):
        # get indices of all rows corresponding to a certain time of the day/week 
        idxs = [(length_week*week_idx)+time_in_week for week_idx in range(0,int(num_weeks))]

        # (eg monday 00:02) is equal to the average of every monday at 00:02
        df_avg[time_in_week] = np.mean(df[idxs],0)
    return df_avg

def pivot_simple(inputFile,specifiedSensors,sql_headers=None,window = None):
    '''
        First step of all further analyses :
        Long/narrow dataset from SQL is made wide (one column per sensor, time stamps are rows)
        Format of input file must be [S_IDX,ZEIT,wert]
        Args : 
            inputFile :         Path to csv file generated by sql
            specifiedSensors :  numpy array of sensor Ids that should be used. If present then inefficients sensors not removed
            window (opt)     :  If included, performs smoothing operation. If not provided no smoothing is done!
        Return :
            data_wide_all :     Pandas dataframe containing desired data
    '''
    debugInfo(__name__,"Beginning to read file")
    all_data = pd.read_csv(inputFile,sep=",")
    debugInfo(__name__,"Read input SQL file with shape : (%d, %d)"%(all_data.shape[0],all_data.shape[1]))


    if (specifiedSensors is not None):
        debugInfo(__name__,"%d Sensors specified, getting indices from"%specifiedSensors.shape[0])
        sensorIndices = np.where(all_data.iloc[:,0].values==specifiedSensors.values)[1]
        all_data = all_data.iloc[sensorIndices,:]

    if sql_headers is None : sql_headers = ['S_IDX','ZEIT','wert']
    
    # make into a wide table
    data_wide_all = all_data.pivot(index=sql_headers[1], columns=sql_headers[0], values=sql_headers[2])
    debugInfo(__name__,"Pivoted input shape : (%d, %d)"%(data_wide_all.shape[0],data_wide_all.shape[1]))

    # do the rolling mean
    if window is not None:
        data_wide_all = data_wide_all.rolling(window,min_periods =1).mean()
        debugInfo(__name__,"Calculated the rolling average using window %d : (%d, %d)"%(window,data_wide_all.shape[0],data_wide_all.shape[1]))

    return data_wide_all