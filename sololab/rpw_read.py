from .values import *
import os
import numpy as np
import math
from datetime import datetime,time,timedelta
## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table, vstack, hstack
import astropy.units as u


#os.environ["CDF_LIB"] = "~/Documents/cdfpy38/src/lib/"
#os.environ["CDF_LIB"] = "~/Documents/cdf38/src/lib/"
os.environ["CDF_LIB"] = "/home/dpaipa/Documents/cdf38/src/lib/"
from spacepy import pycdf

def rpw_read_tnr_cdf(filepath, sensor=4, start_index=0, end_index=-99, data_index=0):

    with pycdf.CDF(filepath) as data_L2:

        freq_tnr1 = np.append(
            data_L2['TNR_BAND_FREQ'][0, :], data_L2['TNR_BAND_FREQ'][1, :]
        )
        freq_tnr2 = np.append(
            data_L2['TNR_BAND_FREQ'][2, :], data_L2['TNR_BAND_FREQ'][3, :]
        )
        freq_tnr = np.append(freq_tnr1, freq_tnr2)
        freq_tnr = freq_tnr / 1000.0  # frequency in kHz
        nn = np.size(data_L2['Epoch'][:])
        if end_index == -99:
            end_index = nn
        epochdata = data_L2['Epoch'][start_index:end_index]
        sensor_config = np.transpose(
            data_L2['SENSOR_CONFIG'][start_index:end_index, :]
        )
        auto1_data = np.transpose(data_L2['AUTO1'][start_index:end_index, :])
        auto2_data = np.transpose(data_L2['AUTO2'][start_index:end_index, :])
        sweep_num = data_L2['SWEEP_NUM'][start_index:end_index]
        bande = data_L2['TNR_BAND'][start_index:end_index]
        if sensor == 7:
            auto1_data = np.transpose(
                data_L2['MAGNETIC_SPECTRAL_POWER1'][start_index:end_index, :]
            )
            auto2_data = np.transpose(
                data_L2['MAGNETIC_SPECTRAL_POWER2'][start_index:end_index, :]
            )
        puntical = (data_L2['FRONT_END'][start_index:end_index] == 1).nonzero()
    epochdata = epochdata[puntical[0]]
    sensor_config = sensor_config[:, puntical[0]]
    auto1_data = auto1_data[:, puntical[0]]
    auto2_data = auto2_data[:, puntical[0]]
    sweep_numo = sweep_num[puntical[0]]
    bande = bande[puntical[0]]
    sweep_num = sweep_numo
    timet=epochdata
    #deltasw = sweep_numo[ 1:: ] - sweep_numo[ 0:np.size ( sweep_numo ) - 1 ]
    deltasw = abs (np.double(sweep_numo[ 1::]) - np.double(sweep_numo[ 0:np.size(sweep_numo)-1 ]))
    xdeltasw = np.where ( deltasw > 100 )
    xdsw = np.size ( xdeltasw )
    if xdsw > 0:
        xdeltasw = np.append ( xdeltasw, np.size ( sweep_numo ) - 1 )
        nxdeltasw = np.size ( xdeltasw )
        for inswn in range ( 0, nxdeltasw - 1 ):
            #sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] = sweep_num[
            #                                                           xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] + \
            #                                                           sweep_num[ xdeltasw[ inswn ] ]
            sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] + 1 ] = sweep_num[
                                                                           xdeltasw[ inswn ] + 1:xdeltasw[
                                                                                                     inswn + 1 ] + 1 ] + \
                                                                           sweep_numo[ xdeltasw[ inswn ] ]
    sens0 = (sensor_config[0, :] == sensor).nonzero()[0]
    sens1 = (sensor_config[1, :] == sensor).nonzero()[0]
    psens0 = np.size(sens0)
    psens1 = np.size(sens1)

    if (np.size(sens0) > 0 and np.size(sens1) >0):
        auto_calib = np.hstack ((auto1_data[ :, sens0 ], auto2_data[:, sens1 ]))
        sens = np.append(sens0, sens1)
        timet_ici = np.append(timet[ sens0 ], timet[ sens1 ] )
    else:
        if (np.size(sens0) > 0):
            auto_calib = auto1_data[ :, sens0 ]
            sens = sens0
            timet_ici = timet[ sens0 ]
        if  (np.size(sens1) > 0):
            auto_calib = auto2_data[ :, sens1 ]
            sens = sens1
            timet_ici = timet[ sens1 ]
        if (np.size(sens0) == 0 and np.size(sens1) == 0):
            print('no data at all ?!?')
            V = (128, 128)
            V = np.zeros ( V ) + 1.0
            time = np.zeros ( 128 )
            sweepn_TNR = 0.0
            return {
                'voltage': V,
                'time': time,
                'frequency': freq_tnr,
                'sweep': sweepn_TNR,
                'sensor': sensor,
            }
    ord_time = np.argsort ( timet_ici )
    timerr = timet_ici[ ord_time ]
    sens = sens[ ord_time ]
    bandee = bande[ sens ]
    auto_calib=auto_calib[:,ord_time]
    maxsweep = max ( sweep_num[ sens ] )
    minsweep = min ( sweep_num[ sens ] )
    sweep_num = sweep_num[ sens ]
    V1 = np.zeros(128)
    V = np.zeros(128)
    time = 0.0
    sweepn_TNR = 0.0
    for ind_sweep in range ( minsweep, maxsweep + 1 ):
        ppunt = (sweep_num == ind_sweep).nonzero ()[ 0 ]
        xm = np.size ( ppunt )
        if xm > 0:
            for indband in range ( 0, xm ):
                V1[
                32
                * bandee[ ppunt[ indband ] ]: 32
                                              * bandee[ ppunt[ indband ] ]
                                              + 32
                ] = np.squeeze ( auto_calib[ :, [ ppunt[ indband ] ] ] )

        if np.sum ( V1 ) > 0.0:
            V = np.vstack ( (V, V1) )
            sweepn_TNR = np.append ( sweepn_TNR, sweep_num[  ppunt[ 0 ] ]  )
        V1 = np.zeros ( 128 )
        if xm > 0:
            time = np.append ( time, timerr[ min ( ppunt ) ] )
    V = np.transpose ( V[ 1::, : ] )
    time = time[ 1:: ]
    sweepn_TNR = sweepn_TNR[ 1:: ]
    return {
    'voltage': V,
    'time': time,
    'frequency': freq_tnr,
    'sweep': sweepn_TNR,
    'sensor': sensor,}

def rpw_read_hfr_cdf(filepath, sensor=9, start_index=0, end_index=-99):


    #import datetime

    with pycdf.CDF ( filepath ) as l2_cdf_file:

        frequency = l2_cdf_file[ 'FREQUENCY' ][ : ]  # / 1000.0  # frequency in MHz
        nn = np.size ( l2_cdf_file[ 'Epoch' ][ : ] )
        if end_index == -99:
            end_index = nn
        frequency = frequency[ start_index:end_index ]
        epochdata = l2_cdf_file[ 'Epoch' ][ start_index:end_index ]
        sensor_config = np.transpose (
            l2_cdf_file[ 'SENSOR_CONFIG' ][ start_index:end_index, : ]
        )
        agc1_data = np.transpose ( l2_cdf_file[ 'AGC1' ][ start_index:end_index ] )
        agc2_data = np.transpose ( l2_cdf_file[ 'AGC2' ][ start_index:end_index ] )
        sweep_num = l2_cdf_file[ 'SWEEP_NUM' ][ start_index:end_index ]
        cal_points = (
            l2_cdf_file[ 'FRONT_END' ][ start_index:end_index ] == 1
        ).nonzero ()
    frequency = frequency[ cal_points[ 0 ] ]
    epochdata = epochdata[ cal_points[ 0 ] ]
    sensor_config = sensor_config[ :, cal_points[ 0 ] ]
    agc1_data = agc1_data[ cal_points[ 0 ] ]
    agc2_data = agc2_data[ cal_points[ 0 ] ]
    sweep_numo = sweep_num[ cal_points[ 0 ] ]
    ssweep_num = sweep_numo
    timet = epochdata

    # deltasw = sweep_numo[ 1:: ] - sweep_numo[ 0:np.size ( sweep_numo ) - 1 ]
    deltasw = abs ( np.double ( sweep_numo[ 1:: ] ) - np.double ( sweep_numo[ 0:np.size ( sweep_numo ) - 1 ] ) )
    xdeltasw = np.where ( deltasw > 100 )
    xdsw = np.size ( xdeltasw )
    if xdsw > 0:
        xdeltasw = np.append ( xdeltasw, np.size ( sweep_numo ) - 1 )
        nxdeltasw = np.size ( xdeltasw )
        for inswn in range ( 0, nxdeltasw - 1 ):
            # sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] = sweep_num[
            # xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] + \
            # sweep_numo[ xdeltasw[ inswn ] ]
            sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] + 1 ] = sweep_num[
                                                                           xdeltasw[ inswn ] + 1:xdeltasw[
                                                                                                     inswn + 1 ] + 1 ] + \
                                                                           sweep_numo[ xdeltasw[ inswn ] ]
    sens0 = (sensor_config[ 0, : ] == sensor).nonzero ()[ 0 ]
    sens1 = (sensor_config[ 1, : ] == sensor).nonzero ()[ 0 ]
    print("  sensors: ",np.shape(sens0),np.shape(sens1))
    psens0 = np.size ( sens0 )
    psens1 = np.size ( sens1 )
    timet_ici=[]

    if (np.size ( sens0 ) > 0 and np.size ( sens1 ) > 0):
        agc = np.append ( np.squeeze ( agc1_data[ sens0 ] ), np.squeeze ( agc2_data[ sens1 ] ) )
        frequency = np.append ( np.squeeze ( frequency[ sens0 ] ), np.squeeze ( frequency[ sens1 ] ) )
        sens = np.append ( sens0, sens1 )
        timet_ici = np.append ( timet[ sens0 ], timet[ sens1 ] )
    else:
        if (np.size ( sens0 ) > 0):
            agc = np.squeeze ( agc1_data[ sens0 ] )
            frequency = frequency[ sens0 ]
            sens = sens0
            timet_ici = timet[ sens0 ]
        if (np.size ( sens1 ) > 0):
            agc = np.squeeze ( agc2_data[ sens1 ] )
            frequency = frequency[ sens1 ]
            sens = sens1
            timet_ici = timet[ sens1 ]
        if (np.size ( sens0 ) == 0 and np.size ( sens1 ) == 0):
            print('  no data at all ?!?')
            V = (321)
            V = np.zeros ( V ) + 1.0
            time = np.zeros ( 128 )
            sweepn_HFR = 0.0
    #           return {
    #               'voltage': V,
    #               'time': time,
    #               'frequency': frequency,
    #               'sweep': sweepn_HFR,
    #               'sensor': sensor,
    #           }
    ord_time = np.argsort ( timet_ici )
    timerr = timet_ici[ ord_time ]
    sens = sens[ ord_time ]
    agc = agc[ ord_time ]
    frequency = frequency[ ord_time ]
    maxsweep = max ( sweep_num[ sens ] )
    minsweep = min ( sweep_num[ sens ] )
    sweep_num = sweep_num[ sens ]

    V1 = np.zeros ( 321 ) - 99.
    V = np.zeros ( 321 )
    freq_hfr1 = np.zeros ( 321 ) - 99.
    freq_hfr = np.zeros ( 321 )
    time = 0.0
    sweepn_HFR = 0.0
    # ind_freq = [(frequency - 0.375) / 0.05]
    ind_freq = [ (frequency - 375.) / 50. ]
    ind_freq = np.squeeze ( ind_freq )
    ind_freq = ind_freq.astype ( int )
    for ind_sweep in range ( minsweep, maxsweep + 1 ):
        ppunt = (sweep_num == ind_sweep).nonzero ()[ 0 ]
        xm = np.size ( ppunt )
        if xm > 0:
            V1[ ind_freq[ ppunt ] ] = agc[ ppunt ]
            freq_hfr1[ ind_freq[ ppunt ] ] = frequency[ ppunt ]
            # print(frequency[ppunt])
        if np.max ( V1 ) > 0.0:
            V = np.vstack ( (V, V1) )
            freq_hfr = np.vstack ( (freq_hfr, freq_hfr1) )
            sweepn_HFR = np.append ( sweepn_HFR, sweep_num[ ppunt[ 0 ] ] )
        V1 = np.zeros ( 321 ) - 99
        freq_hfr1 = np.zeros ( 321 )  # - 99
        if xm > 0:
            time = np.append ( time, timerr[ min ( ppunt ) ] )
    # sys.exit ( "sono qui" )
    V = np.transpose ( V[ 1::, : ] )
    time = time[ 1:: ]
    sweepn_HFR = sweepn_HFR[ 1:: ]
    freq_hfr = np.transpose ( freq_hfr[ 1::, : ] )
    return {
        'voltage': V,
        'time': time,
        'frequency': freq_hfr,
        'sweep': sweepn_HFR,
        'sensor': sensor,
    }



# -*- coding: utf-8 -*-

def fft_filter(x1, tlow, tup):
    npp = np.size(x1)
    yfo = np.fft.ifft(x1)
    yyf0 = yfo[0]
    freq = np.arange(float((npp - 1) / 2) + 1.0)
    freq = freq / (npp)
    if (np.mod(npp, 2)) == 0:
        yyf = yfo[1 : int(npp / 2)]  # yyf = yfo[ 1:(np / 2) - 1 ]
        yyf0 = yfo[0]
        yyfn = yfo[int(npp / 2)]
        freq = np.arange(float(npp / 2) + 1.0)
        freq[0 : int(npp / 2)] = freq[0 : int(npp / 2)] / (npp)
        freq[int(npp / 2)] = 1.0 / (2.0)
    period = [1.0 / freq[1::], 0]
    fup = tup / npp
    flow = tlow / (npp)
    nnf = np.size(freq)
    if flow > freq[nnf - 1]:
        flow = freq[nnf - 1]
    pp = np.where((freq > flow) & (freq < fup))
    xn = np.size(pp)
    if xn > 0:
        yf = np.zeros(npp, dtype=complex)
        yf[pp] = yfo[pp]
        yf[npp - np.squeeze(pp)] = yfo[npp - np.squeeze(pp)]
        yf[0] = 0.0
        yfnull = yf * 0.0
        yfnull[0] = yyf0

    xf = np.fft.fft(yf)
    xf = np.real(xf)
    xfnull = np.fft.fft(yfnull)
    xfnull = np.real(xfnull)
    return xf, xfnull


def tnr_pre_process(V, tlow=0.01, tup=280.0):
    """
    Mask and filter data to remove artifacts

    Note: output data are in log scale
    """

    ndim = np.shape(V)
    Vf = (ndim[0], ndim[1])
    Vf = np.zeros(Vf)
    Vf0 = (ndim[0], ndim[1])
    Vf0 = np.zeros(Vf0)

    V2o = 10.0 * np.log10(V)
    V2 = V2o
    nap = np.isinf(V2o)

    for ii in range(128):
        napp = nap[ii, :]
        V2[ii, napp] = np.mean(V2o[np.isfinite(V2o)])

    for ii in range(128):
        x = np.squeeze(V2[ii, :])
        outf = fft_filter(x, tlow, tup)
        xf = outf[0]
        xfnull = outf[1]
        Vf[ii, :] = xf
        Vf0[ii, :] = xfnull

    V[75:80, :] = 1e-25
    V[85:86, :] = 1e-25
    V[103:104, :] = 1e-25
    V[110:111, :] = 1e-25
    V[115:116, :] = 1e-25
    V[118:119, :] = 1e-25

    #VV = 10.0 * np.log10(V)

    Vfil = Vf + np.median(Vf0)
    return V, Vfil

def tnr_del_unwanted_values(array, freq,tnr_remove_idx=tnr_remove_idx):
    array_o=array.copy()
    freq_o=freq.copy()
    cmpt=0
    for ix in tnr_remove_idx:
            array_o=np.delete(array_o, (ix-cmpt), axis=0)
            freq_o=np.delete(freq_o, (ix-cmpt))
            cmpt=cmpt+1

    return array_o, freq_o




# RPW get data object
def rpw_get_data(file,sensor=None,filter = tnr_remove_idx):
    data_types = ["hfr","tnr"]
    for dtp in data_types:
        if dtp in os.path.basename(file):
            data_type = dtp

    if(sensor==None):
        if(data_type=="hfr"):
            sensor=9
        elif(data_type=="tnr"):
            sensor=4
    # data read
    print("Extracting info:")
    infos = os.path.basename(file).split("-")[0].split("_")
    #dts = [datetime.strptime(x,"%Y%m%dT%H%M%S") for x in os.path.basename(pathfile).split("_")[3].split("-")]
    #dts = [datetime.strftime(x,dt_fmt)for x in dts]
    txt_type = "{}-{} {}".format(infos[2],data_type.upper() ,infos[1]).upper()

    #output  dict
    print("  File: ",os.path.basename(file))
    print("  Type: ",txt_type)
    data = None
    if(data_type=="hfr"):
        data= rpw_read_hfr_cdf(file,sensor=sensor)
    if(data_type=="tnr"):

        data = rpw_read_tnr_cdf(file,sensor=sensor)
        data = rpw_filter_freq_tnr(data,filter=filter)

    data["type"]=data_type
    return data

def rpw_filter_freq_tnr(data,filter=tnr_remove_idx):

    V,vfil  = tnr_pre_process(data["voltage"])
    new_v,new_freq = tnr_del_unwanted_values(V,data["frequency"],tnr_remove_idx=filter)

    data["frequency"] = new_freq
    data["voltage"] = new_v

    print("  RPW-TNR frequencies filtered.")

    return data


# RPW filter frequencies
def rpw_select_freq_indexes(frequency,data_type='hfr',**kwargs):#,freq_col=0,proposed_indexes=None):
    #indexes of frequencies different from 0 or -99 (column 0 in frequency matrix)




    selected_freqs=None
    if(data_type=='hfr'):
        fcol = kwargs["freq_col"]
        freq_nozero = np.where(frequency.T[fcol]>0)[0]
        selected_freqs = freq_nozero
        dfreq = np.array(frequency[freq_nozero,fcol])
        dfreq = dfreq[1:]-dfreq[:-1]
        if kwargs["which_freqs"]=="both":
            selected_freqs = [ j  for j in kwargs["proposed_indexes"] if j in freq_nozero]

        if(not kwargs["freq_range"]==None):
            #print(frequency[selected_freqs,fcol])
            selected_freqs = [selected_freqs[j] for j in range(len(selected_freqs)) if np.logical_and(frequency[selected_freqs[j],fcol] <= kwargs["freq_range"][1], frequency[selected_freqs[j], fcol]>=kwargs["freq_range"][0])]
        return selected_freqs,frequency[selected_freqs,fcol],dfreq
    elif(data_type=='tnr'):
        freq_nozero = np.where(frequency.T>0)[0]
        selected_freqs = freq_nozero
        dfreq = np.array(frequency[freq_nozero])
        dfreq = dfreq[1:]-dfreq[:-1]
        #print("nz",dfreq)
        if kwargs["which_freqs"]=="both":
            selected_freqs = [ j  for j in kwargs["proposed_indexes"] if j in freq_nozero]

        if(not kwargs["freq_range"]==None):
            #print(frequency[selected_freqs,fcol])
            selected_freqs = [selected_freqs[j] for j in range(len(selected_freqs)) if np.logical_and(frequency[selected_freqs[j]] <= kwargs["freq_range"][1], frequency[selected_freqs[j]]>=kwargs["freq_range"][0])]
        return selected_freqs,frequency[selected_freqs],dfreq

def rpw_create_PSD(data,freq_range=None,date_range=None,freq_col=0,proposed_indexes=rpw_suggested_indexes,which_freqs="both",rpw_bkg_interval=None,sort=True):
    # return,x,y,z

    time_data = data["time"]
    z = data["voltage"]
    data_type = data["type"]




    date_idx = np.arange(len(time_data))
    start_date,end_date = time_data[0],time_data[-1]
    # when date range provided,select time indexes
    if(date_range):
        start_date = datetime.strptime(date_range[0], std_date_fmt)
        end_date = datetime.strptime(date_range[1], std_date_fmt)

    date_idx = np.array( np.where( np.logical_and(time_data<=end_date,time_data>=start_date))[0] ,dtype=int)

    if(len(date_idx)==0):
        print("  RPW Error! no data in between provided date range")
        return
    print("  data cropped from ",start_date," to ",end_date)
    # define time axis
    date_idx = np.array(date_idx)

    t_axis = time_data[date_idx]

    #define energy axis
    freq_ = data['frequency']

    if data_type  == "hfr":
        freq_idx,freq_axis,dfreq = rpw_select_freq_indexes(freq_,data_type=data_type,freq_col=freq_col,freq_range=freq_range,
                                             proposed_indexes=proposed_indexes,which_freqs=which_freqs)
    elif data_type == "tnr":
        freq_idx,freq_axis,dfreq = rpw_select_freq_indexes(freq_,data_type=data_type,freq_col=freq_col,freq_range=freq_range,
                                         proposed_indexes=proposed_indexes,which_freqs="non_zero")
    freq_idx = np.array(freq_idx)
    print(f"  {len(freq_axis)} Selected frequencies [kHz]: ",*np.round(np.sort(freq_axis),2))

    # selecting Z axis (cropping)

    z_axis= z[:,date_idx]

    z_axis = z_axis[freq_idx,:]

    mn_bkg=None

# BKG subtraction (approx) if needed
    if rpw_bkg_interval :
        rpw_bkg_interval=[datetime.strptime(x,std_date_fmt) for x in rpw_bkg_interval ]
        if(rpw_bkg_interval[0]>t_axis[-1] or rpw_bkg_interval[-1]<t_axis[0]):
            print("  [!] Your bkg interval is outside of the PSD time range")

        print("  Creating mean bkg from ",rpw_bkg_interval[0]," to ",rpw_bkg_interval[1],"...")

        idx_in = [j for j in range(len(data["time"])) if np.logical_and(data["time"][j]>=rpw_bkg_interval[0],data["time"][j]<=rpw_bkg_interval[-1])]

        mn_bkg = np.mean(z[np.ix_(freq_idx,idx_in)],axis=1)
        mn_bkg = np.array([mn_bkg for i in range(np.shape(z_axis)[1])]).T
        mn_bkg = mn_bkg.clip(0,np.inf)

        z_axis=np.clip(z_axis-mn_bkg,1e-16,np.inf)

        print("  bkg done.")

    verbose_dts(np.array(t_axis))
    return_dict = {
        "t_idx":date_idx,
        "freq_idx":freq_idx,
        "time":t_axis,
        "frequency":freq_axis,
        "v":z_axis,
        "df":dfreq,
        "bkg":mn_bkg,
        "type":data_type
    }
    print("  RPW PSD Done.")

    if(sort):
        return_dict = rpw_sort_psd(return_dict)


    return return_dict


def rpw_sort_psd(rpw_psd):
    print("  Sorting PSD time axis...")
    ksort = sorted(zip(rpw_psd["time"],range(len(rpw_psd["v"].T))))
    z_axis = np.array([rpw_psd["v"][:,y] for x,y in ksort]).T
    t_axis = np.array(sorted(rpw_psd["time"]))

    rpw_psd["v"] = z_axis
    rpw_psd["time"] = t_axis

    print("  Sorting PSD frequency axis...")
    ff=rpw_psd["frequency"]
    vv=rpw_psd["v"]
    sorted_v=np.array([x for _, x in sorted(zip(ff, vv))])
    sorted_f=np.array([_ for _, x in sorted(zip(ff, vv))])
    rpw_psd["v"]=sorted_v
    rpw_psd["frequency"]=sorted_f


    print("  PSD Sorted.")
    return rpw_psd
