import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib as mpl
from .values import *
from .freq_drift import *


def create_dataframe_from_csv(path,remove_not_used=True):
    df = pd.read_csv(path)


    date_peak_th = []
    date_onset_xray = []
    date_peak_hxr = []
    date_onset_it3 = []
    date_change = []
    delay_onset = []
    delay_hxr_peak = []
    delay_th_peak=[]
    delay_change_to_onset=[]



    for index,row in df.iterrows():

        # print row status
        print('[',index,']  ',row['date'],row['peak_time_4_10'],'   ',row["peak_rate_4_10"],'cts/sec',
            '(faint)' if row["faint"]=='yes' else '','[NOT USED]' if row["used"]=='no' else '')
        



        # biuld datetime objects 
        date_peak_th.append(dt.datetime.strptime(row['date']+' '+row['peak_time_4_10'],std_date_fmt))
        date_onset_xray.append(dt.datetime.strptime(row['date']+' '+row['onset_time_xray'],std_date_fmt))
        date_peak_hxr.append(dt.datetime.strptime(row['date']+' '+row['peak_time_close_hxr'],std_date_fmt))
        date_onset_it3.append(dt.datetime.strptime(row['date']+' '+row['onset_time_it3'],std_date_fmt))
        date_change.append(dt.datetime.strptime(row['date']+' '+row['change_time_before_hxr_peak'],std_date_fmt))

        # estimate delays 
        delay_onset.append((date_onset_it3[-1]-date_onset_xray[-1]).total_seconds())
        delay_hxr_peak.append((date_onset_it3[-1]-date_peak_hxr[-1]).total_seconds())
        delay_th_peak.append((date_onset_it3[-1]-date_peak_th[-1]).total_seconds())
        delay_change_to_onset.append((date_onset_it3[-1]-date_change[-1]).total_seconds())
        

        
    # as arrays 
    date_peak_th = np.array(date_peak_th)
    date_onset_xray = np.array(date_onset_xray)
    date_peak_hxr = np.array(date_peak_hxr)
    date_onset_it3 = np.array(date_onset_it3)
    delay_onset = np.array(delay_onset)
    delay_hxr_peak = np.array(delay_hxr_peak)
    delay_th_peak = np.array(delay_th_peak)
    delay_change_to_onset = np.array(delay_change_to_onset)
    delay_onset_fraction = delay_onset/df["duration"]
        
    #append to df
    df["date_peak_4_10"] = date_peak_th
    df["date_peak_close_hxr"] = date_peak_hxr
    df["date_onset_xray"] = date_onset_xray
    df["date_onset_it3"] = date_onset_it3
    df["delay_onset"]=delay_onset
    df["delay_peak_th"]=delay_th_peak
    df["delay_peak_hxr"]=delay_hxr_peak
    df["delay_change_to_onset"]=delay_change_to_onset
    df["delay_onset_fraction"]=delay_onset_fraction

    df['streamer'] = ['no' if x is np.nan else 'yes' for x in df['streamer']]


    if(remove_not_used):
        print("[Note] Rows with 'used = No' were removed form final table")
        return df[df['used']=='yes']


    return df





# df['onset_distance_leblanc_rs'] = np.array(sololab.r_from_freq(df['Onset frequency'],sololab.ne_from_r_leblanc))
# df['exp_delay_005'] = np.array(sololab.r_from_freq(df['Onset frequency'],sololab.ne_from_r_leblanc))*sololab.km_per_Rs/(0.05*sololab.speed_c_kms)
# df['exp_delay_01'] = np.array(sololab.r_from_freq(df['Onset frequency'],sololab.ne_from_r_leblanc))*sololab.km_per_Rs/(0.1*sololab.speed_c_kms)
# df['exp_delay_05'] = np.array(sololab.r_from_freq(df['Onset frequency'],sololab.ne_from_r_leblanc))*sololab.km_per_Rs/(0.5*sololab.speed_c_kms)


def estimate_expected_heights_delays(dataframe,density_model=ne_from_r_leblanc,name='leblanc',velocities=[0.05,0.1,0.5]):

    dist_key = f'onset_distance_{name}_rs'
    dataframe[dist_key] = np.array(r_from_freq(dataframe['onset_frequency'],density_model))

    for vel in velocities:
        vel_string = "{:.2f}".format(vel).replace(".","")
        vel_estim_key = 'expected_delay_{}_{}c'.format(name,vel_string)
        dataframe[vel_estim_key]=dataframe[dist_key]*km_per_Rs/(vel*speed_c_kms)
    return dataframe




#def create_numerical_dataframe(dataframe):


