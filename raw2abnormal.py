#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
raw2abnormal: a tool that change spatio-temporal solar data into abnormal events 
"""
import numpy as np
import pandas as pd
import scipy.io as sio

def API_extraction(output_name,scale):
    gap = 0.038*2
    df = pd.DataFrame({})
    for i in range(scale):  
        for j in range(scale):
            print(i,j)
            # Declare all variables as strings. Spaces must be replaced with '+', i.e., change 'John Smith' to 'John+Smith'.
            # Define the lat, long of the location and the year
            lat,lon,year = 33.7490+i*gap, -84.3880+j*gap, 2018
            # You must request an NSRDB api key from the link above
            api_key = 'w63qaGKhPZNl9PHiNjZSi0JikYPAyUcY7V27fTbv'
            # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
            attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
            # Choose year of data
            year = '2018'
            # Set leap year to true or false. True will return leap day data if present, false will not.
            leap_year = 'false'
            # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
            interval = '30'
            # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
            # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
            # local time zone.
            utc = 'false'
            # Your full name, use '+' instead of spaces.
            your_name = 'Ruyi+Ding'
            # Your reason for using the NSRDB.
            reason_for_use = 'beta+testing'
            # Your affiliation
            your_affiliation = 'Georgia+Tech'
            # Your email address
            your_email = 'thuzmh@gmail.com'
            # Please join our mailing list so we can keep you up-to-date on new developments.
            mailing_list = 'true'

            # Declare url string
            url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
            # Return just the first 2 lines to get metadata:
            info = pd.read_csv(url, nrows=1)
            # See metadata for specified properties, e.g., timezone and elevation
            timezone, elevation = info['Local Time Zone'], info['Elevation']
            # Return all but first 2 lines of csv to get data:
            df_temp = pd.read_csv('https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes), skiprows=2)

            # Set the time index in the pandas dataframe:
            df_temp = df_temp.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
            df_temp['x'] = i
            df_temp['y'] = j
            df = pd.concat([df, df_temp])
    df.to_csv(f"../{output_name}.csv")


def _toabnormal_single(df,winsize, delta):
    """
    For each location, find the abnormal events according to history data and threshold 
    winsize: preserved history data
    delta: threshold
    """
    result = pd.DataFrame({})
    dfDay = df.groupby('everyDay')['GHI','DHI','DNI'].mean().reset_index()
    for i in range(winsize+1,len(dfDay)):
        dfTemp = dfDay[i-1-winsize:i-1]
        for kpi in ['GHI','DHI','DNI']:
            if ((dfDay[kpi][i] > dfTemp[kpi].quantile(1-delta)) or (dfDay[kpi][i] < dfTemp[kpi].quantile(delta))):
                result = pd.concat([result, dfDay[i:i+1]])
                break                
    return result

def _toabnormal_multiple(df,winsize, delta):
    """
    For each location, find the positive and negative events separately according to history data and threshold 
    winsize: preserved history data
    delta: threshold
    """
    result_pos = pd.DataFrame({})
    result_neg = pd.DataFrame({})
    dfDay = df.groupby('everyDay')['GHI','DHI','DNI'].mean().reset_index()
    for i in range(winsize+1,len(dfDay)):
        dfTemp = dfDay[i-1-winsize:i-1]
        for kpi in ['GHI','DHI','DNI']:
            if (dfDay[kpi][i] > dfTemp[kpi].quantile(1-delta)):
                result_pos = pd.concat([result_pos, dfDay[i:i+1]])
                break                

            if (dfDay[kpi][i] < dfTemp[kpi].quantile(delta)):
                result_neg = pd.concat([result_neg, dfDay[i:i+1]])
                break  

    return result_pos,result_neg

def ToAbnormalByDay(file,winsize=30,delta=0.005, single=True):
    """ 
    turn from a pandas solar data to numpy data where abnormal events are daily observed
    table strcture:
    Unnamed: 0	Year	Month	Day	Hour	Minute	GHI	DHI	DNI	Wind Speed	Temperature	Solar Zenith Angle	x	y
    """
    griddf = pd.read_csv(file)
    griddf['everyDay'] = [griddf['Unnamed: 0'][i].split(' ')[0][0:] for i in range(len(griddf))]

    griddf['x-y'] = [str(griddf['x'][i])+'-'+str(griddf['y'][i]) for i in range(len(griddf))]
    abdf = pd.DataFrame({})
    if single:
        for i in range(len(griddf['x-y'].unique())):
            df = griddf[griddf['x-y'] == griddf['x-y'].unique()[i]]
            result = _toabnormal_single(df,winsize,delta)
            result['x'] = (griddf['x-y'].unique()[i][0])
            result['y'] = (griddf['x-y'].unique()[i][2])
            abdf = pd.concat([abdf,result])

        abdf['start'] = ('2018-01-01')
        abdf['start'] = abdf['start'].apply(pd.to_datetime)
        abdf['everyDay'] = abdf['everyDay'].apply(pd.to_datetime)
        abdf['t'] = (abdf['everyDay']-abdf['start']).dt.days
        abdf = abdf.drop(["everyDay", "start"], axis=1)

        df_np = abdf.to_numpy()

        return df_np

    # return abnormal events with marks
    else:
        abdf_neg = pd.DataFrame({})
        for i in range(len(griddf['x-y'].unique())):
            df = griddf[griddf['x-y'] == griddf['x-y'].unique()[i]]
            res_pos,res_neg = _toabnormal_multiple(df,winsize,delta)
            res_pos['x'] = (griddf['x-y'].unique()[i][0])
            res_pos['y'] = (griddf['x-y'].unique()[i][2])
            res_neg['x'] = (griddf['x-y'].unique()[i][0])
            res_neg['y'] = (griddf['x-y'].unique()[i][2])
            abdf = pd.concat([abdf,res_pos])
            abdf_neg = pd.concat([abdf_neg,res_neg])

        abdf['start'] = ('2018-01-01')
        abdf['start'] = abdf['start'].apply(pd.to_datetime)
        abdf['everyDay'] = abdf['everyDay'].apply(pd.to_datetime)
        abdf['t'] = (abdf['everyDay']-abdf['start']).dt.days
        abdf = abdf.drop(["everyDay", "start"], axis=1)

        df_np = abdf.to_numpy()

        abdf_neg['start'] = ('2018-01-01')
        abdf_neg['start'] = abdf_neg['start'].apply(pd.to_datetime)
        abdf_neg['everyDay'] = abdf_neg['everyDay'].apply(pd.to_datetime)
        abdf_neg['t'] = (abdf_neg['everyDay']-abdf_neg['start']).dt.days
        abdf_neg = abdf_neg.drop(["everyDay", "start"], axis=1)

        df_np_neg = abdf_neg.to_numpy()
        return df_np,df_np_neg
        
def ToAbnormalByHour(file,winsize=30,delta=0.005):
    """
    turn from a pandas solar data to numpy data where abnormal events are hourly observed
    table strcture:
    Unnamed: 0	Year	Month	Day	Hour	Minute	GHI	DHI	DNI	Wind Speed	Temperature	Solar Zenith Angle	x	y
    """
    data = pd.read_csv(file)
    data['time'] = [i for i in range(len(data))]
    len_per_grid = 24*2*365   # recording every half hour
    data['time'] = data['time']%len_per_grid
    data = data.sort_values(by=['time'])
    test = data[data['x']==0].reset_index(drop = True)
    test = test[test['y']==1].reset_index(drop = True)

    # only consider Feb data here for simplicity
    Febdata = data[(data['Month']==2)|(data['Month']==1)]
    Febdata = Febdata.sort_values(by=['time'])

    dayLength = winsize
    Label = []
    for xi in range(3):
        for yi in range(3):
            print(f"Now considering the grid ({xi},{yi});")
            subdata = Febdata[Febdata['x']==xi].reset_index(drop = True)
            subdata = subdata[subdata['y']==yi].reset_index(drop = True)
            subdata = subdata.sort_values(by=['time'])
            # print(len(subdata))
            for i in range(len(subdata)):
                if i < dayLength *48:
                    continue
                else:
                    hour = subdata['Hour'][i]
                    minute = subdata['Minute'][i]
                    df = subdata[(subdata['Hour'] == hour)&(subdata['Minute'] == minute)].reset_index(drop = True)
                    df = df[(i//48-dayLength):(i//48)]
                    # print(df)
                    # df = df[:dayLength]
                    df1 = df.groupby(['Hour','Minute'])['GHI','DHI','DNI'].quantile(1-delta).reset_index()
                    df2 = df.groupby(['Hour','Minute'])['GHI','DHI','DNI'].quantile(delta).reset_index()
                    if (subdata['GHI'][i]>df1['GHI'][0])|(subdata['GHI'][i]<df2['GHI'][0])|\
                    (subdata['DHI'][i]>df1['DHI'][0])|(subdata['DHI'][i]<df2['DHI'][0])|\
                    (subdata['DNI'][i]>df1['DNI'][0])|(subdata['DNI'][i]<df2['DNI'][0]):
                        Label.append([subdata['time'][i]-dayLength *48,xi,yi])

    abnormal = np.asarray(Label)
    print(abnormal)
    return abnormal




def ToMatrix_daily(numpy_data,grid):
    # data = np.load(numpy_data,allow_pickle=True)
    data = numpy_data[:,[3,4,5]] #[x,y,t]
    data = data.astype(int)
    datashape = data.shape
    print(datashape)
    print(f"density is {datashape[0]/365/grid/grid}")
    print(data)
    mat = np.zeros((365,grid,grid))
    for i in range(datashape[0]):
        mat[data[i,2],data[i,0],data[i,1]] = 1
    print(mat.shape)
    vecmat = mat.reshape(365*grid*grid,1)
    # print(f"daily_mat is :\n {mat}")

    print(vecmat.shape)
    sio.savemat('../../Hawkes_discrete_code/saved_solar.mat', {'obs': vecmat})
    return mat

def ToMatrix_daily_multi(numpy_data1, numpy_data2, grid):
    """
    Generate multistate vectors to feed into matlab codes 
    """
    data1 = numpy_data1[:,[3,4,5]] #[x,y,t]
    data1 = data1.astype(int)
    datashape = data1.shape
    print(f"density1 is {datashape[0]/365/grid/grid}")
    mat = np.zeros((365,grid,grid))
    for i in range(datashape[0]):
        mat[data1[i,2],data1[i,0],data1[i,1]] = 1



    data2 = numpy_data2[:,[3,4,5]] #[x,y,t]
    data2 = data2.astype(int)
    datashape = data2.shape
    print(f"density2 is {datashape[0]/365/grid/grid}")
    for i in range(datashape[0]):
        mat[data2[i,2],data2[i,0],data2[i,1]] = 2


    vecmat = mat.reshape(365*grid*grid,1)
    # print(f"daily_mat is :\n {mat}")

    sio.savemat('../../Hawkes_discrete_code/saved_solar_multi.mat', {'obs': vecmat})
    return mat

def ToMatrix_hourly(numpy_data,grid):
    data = numpy_data.astype(int)
    datashape = data.shape          #[t,x,y]
    # Feb data
    mat = np.zeros((28*48,grid,grid))
    print(f"density is {datashape[0]/28/48/grid/grid}")
    for i in range(datashape[0]):
        mat[data[i,0],data[i,1],data[i,2]] = 1
    # print(f"hourly_mat is :\n {mat}")
    vecmat = mat.reshape(28*48*grid*grid,1)
    sio.savemat('../../Hawkes_discrete_code/saved_solar_hour.mat', {'obs': vecmat})


