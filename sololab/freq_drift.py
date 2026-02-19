

#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
from .values import *
from .quicklooks import *
from .fit_functions import *
## PLOT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
## MATH
from scipy.optimize import curve_fit as cfit
import numpy as np


## ASTROPY
from astropy.time.core import Time, TimeDelta
from astropy.table import Table, vstack, hstack
import astropy.units as u



def vc_to_k(v,units="kev",relativistic=False):
    m = m_e_kg
    c = 1000*speed_c_kms

    k_joule = 0.5*m*(v*c)**2 #kg m2/s2 Joule
    if(relativistic):
        beta = v
        gamma = 1/np.sqrt(1-beta**2)
        k_joule = (gamma-1)*m*c**2



    if(units == "j"):
        return k_joule
    elif(units =="ev"):
        return ev_per_joule*k_joule #ev
    elif(units == "kev"):
        return ev_per_joule*k_joule/1000. #kev






def fit_func_gaussian(x,a,b,c,d):
    return (10**a) * np.exp(-(x-b)**2/( c**2)) + (10**d)

def dt_to_sec_t0(time_data,t0=None):
    def _to_seconds(delta):
        if hasattr(delta, "seconds"):
            return delta.seconds + delta.microseconds / 1e6
        if isinstance(delta, np.timedelta64):
            return delta / np.timedelta64(1, "s")
        return delta
    
    if(t0==None):
        t0=time_data[0]
    time_dts = [time_data[i]-t0 for i in range(len(time_data))]
    t_0 = np.array([_to_seconds(t) for t in time_dts])

    return [t_0,time_data[0]]

def sec_t0_to_dt(secs, t0):
    tmes = [t0 + dt.timedelta(seconds=secs[j]) for j in range(len(secs))]
    return tmes

def create_suggested_times(time,freq,end_estimate=0.5):

    mid_time = time[0] + end_estimate*(time[-1]-time[0])
    idx_min=np.argmin(np.abs(np.array(time)-mid_time))
    

    lin_m = (np.log10(freq[0])-np.log10(freq[-1]))/(time[idx_min]-time[0])
    lin_b = np.log10(freq[0])-lin_m*time[idx_min]

    sugg= [(np.log10(f)-lin_b)/lin_m for f in freq]
   
    return sugg




# TO REVIEW!
def rpw_fit_time_profiles(rpw_psd,func,time_range,frequency_range,metrics=["rmse"],
                    excluded_freqs=[],tol = 0.2,smooth_pts=None,secondary_loop=False,
                    name_tag=""):
    time_data = rpw_psd["time"]
    v_data = rpw_psd["v"]
    freq_data = rpw_psd["frequency"]

    #date fmt
    dt_fmt=std_date_fmt

    #estimate error
    # time delta
    t_err = np.mean(time_data[1:]-time_data[:-1]).seconds/2.
    #F delta
    f_err = (freq_data[1:]-freq_data[:-1])/2.


    #print header
    print("\n")
    print(35*"=","FIT TIME PROFILES",35*"=")
    print(" Data uncertainty:")
    print("   Time: {} s".format(round(t_err,2)))
    print("   Freq: between {}-{} kHz".format(np.min(f_err),np.max(f_err)))
    print(" Defining time reference...")

    # selecting fit intervals
    date_range_dt = [datetime.strptime(x,dt_fmt) for x  in time_range]
    idx_sel_time = np.logical_and(time_data<=date_range_dt[1],time_data>=date_range_dt[0])

    #crop time axis and psd
    time_data = time_data[idx_sel_time]
    v_data = v_data[:,idx_sel_time]

    # if freq range provided, crop
    if(frequency_range):
        idx_sel_freq = np.logical_and(freq_data<=frequency_range[1],freq_data>=frequency_range[0])
        freq_data = freq_data[idx_sel_freq]
        v_data = v_data[idx_sel_freq,:]

    # convering time to seconds
    time_sec,t0 = dt_to_sec_t0(time_data) # time axis in seconds wrt to t0 (first time bin)
    time_span = time_sec[-1]-time_sec[0] # total time axis length in seconds
    print("  t0 = {} [{} seconds interval]".format(datetime.strftime(t0,dt_fmt),round(time_span,2)))
    print(" Fitting peaks for {} frequencies:\n Range: {} kHz to {} kHz".format(len(freq_data),int(freq_data[0]),int(freq_data[-1])))
    print("  Intensity model to fit: {}".format(func.name().upper()))

    # START MAIN LOOP
    print(35*"-","Main Loop (all freqs)",35*"-")
    #asume that peak of max freq. is close to the beggining of the timespan
    prev_center = None

    # Type III onset
    starting_point =[]

    # save fit results
    curve_fits = {}

    unsolved_freqs = [] # main loop
    failed_freqs = [] # after secondary loop

    #save metadata of the fit process
    curve_fits_meta={
                     "t":time_sec,
                     "v":v_data,
                     "f":freq_data,
                     "t0":t0,
                     "dt":t_err,
                     "time_interval":time_range,
                     "freq_range":frequency_range,
                     "excluded_f":excluded_freqs,
                     "fit_function":func,
                     "metrics":metrics,
                     "name":func.name()+"_"+name_tag if name_tag!="" else func.name()
    }


    #suggested initial points(linear)
    suggested_times = create_suggested_times(time_sec,freq_data,end_estimate = 0.5)



    # main loop - iterate over the in-range frequencies
    # from higher to lower
    for i in range(len(freq_data)-1,-1,-1):

        # if frequency excluded then skip
        if(freq_data[i] in excluded_freqs):
            print("[{}] {:.0f} kHz   : Excluded frequency!   omitted.".format(i,freq_data[i]))
            continue
        #if not defined, use approx position in timespan (lineal)
        if(not prev_center):
            prev_center= time_sec[0]


        # define xy
        x_ = time_sec
        y_ = v_data[i,:]
        # logscale scale Y variable (voltage psd)
        scale_y = np.log10(y_)
        if(smooth_pts):
            scale_y=smooth(scale_y,smooth_pts)

        try:
            # target elements for params and param covariances
            popt=[]
            pcov=[]



            # different fuction cases

            if (func.name() == "gaussian"):
                #params ->  [mean,dev,scale,bkg]
                bkg_aprox = np.mean(np.sort(scale_y)[-int(0.2*len(scale_y)):])
                bnds = ([np.min(x_),0.1,0,np.min(scale_y)],
                        [np.max(x_),np.max(x_),20,bkg_aprox])
                #po = (prev_center,(np.max(x_)-np.min(x_))/10,3,np.min(scale_y))
                po = (suggested_times[i],(np.max(x_)-np.min(x_))/10,3,np.min(scale_y))
                #print(bnds,po)
                popt,pcov = cfit(func.f,x_,scale_y,p0=po,bounds=bnds)

            elif (func.name() == "fluorescence"):
                #params ->  [t_on,t_off,scale,bkg,phase]

                tau_sugg = lambda f: (np.max(x_)-np.min(x_))*f
                bkg_aprox = np.mean(np.sort(scale_y)[-int(0.2*len(scale_y)):])


                bnds = ([0,0,0,np.min(scale_y),np.min(x_)],
                        [1.5*np.max(x_),1.5*np.max(x_),20,bkg_aprox,np.max(x_)])
                po = (tau_sugg(0.1),tau_sugg(0.5),2,np.min(scale_y),suggested_times[i])
                #print(bnds,po)
                popt,pcov = cfit(func.f,x_,scale_y,p0=po,bounds=bnds)
            elif (func.name() == "gauss_gauss" or func.name() == "gauss_exp"):
                #params ->  [t_on,t_off,tpeak,scale,bkg]

                tau_sugg = lambda f: (np.max(x_)-np.min(x_))*f
                bkg_aprox = np.mean(np.sort(scale_y)[-int(0.2*len(scale_y)):])


                bnds = ([0,0,np.min(x_),0,np.min(scale_y)],
                        [1.5*np.max(x_),1.5*np.max(x_),np.max(x_),20,bkg_aprox])
                po = (tau_sugg(0.1),tau_sugg(0.5),suggested_times[i],2,np.min(scale_y))
                #print(bnds,po)
                popt,pcov = cfit(func.f,x_,scale_y,p0=po,bounds=bnds)
            


            # for cases where parameters were found
            if (len(popt)>0) :
                found_center = func.center(popt)
                
                # diference with previous found point
                dif = found_center-prev_center
                if(i==len(freq_data)-1):
                    dif = 0
                # discard if center out of bounds
                if(found_center<(x_[0]*(1-tol)) or found_center>(x_[-1]*(1+tol))):
                    #print(i,freq_data[i])
                    print("[{}] {:.0f} kHz   : Not in bounds! omitted.".format(i,freq_data[i]))
                    unsolved_freqs.append(freq_data[i])
                else:
                    # if no starting point, this frequency is starting point
                    if(len(starting_point)==0):
                        starting_point = [freq_data[i],sec_t0_to_dt([found_center],t0)[0]]
                        sp_t=datetime.strftime(starting_point[1],dt_fmt)
                        #print(freq_data[i],f_err[i],sp_t)
                        print("Starting point ---------- frequency: {:.0f}+-{:.0f} kHz   time: {}".format(freq_data[i],f_err[i],sp_t))


                    # FITTING ERROR

                    pk_hgh = func.peak_height(popt)
                    if (isinstance(pk_hgh, (list, tuple, np.ndarray))):
                        pk_hgh = pk_hgh[0]
                    snr = 10**(pk_hgh)
                    

                    ## ADD CURVE FIT TO SOLUTIONS
                    f_key = int(freq_data[i])
                    curve_fits[f_key] = {}
                    curve_fits[f_key]["params"]=popt
                    curve_fits[f_key]["covar"]=pcov
                    curve_fits[f_key]["snr"]=snr
                    curve_fits[f_key]["df"]=f_err[i]
                    curve_fits[f_key]["center"]=func.center(popt)
                    curve_fits[f_key]["dev"]=func.dev(popt)
                    curve_fits[f_key]["fit_function"]=func.name()
                    curve_fits[f_key]["peak_height"]=pk_hgh

                    if(func.f_onset):
                        curve_fits[f_key]["onset"]= func.onset(popt)

                    for mtrc in metrics:
                         metric_function = get_metric(mtrc)
                         val = metric_function(x_,scale_y,func.f,popt)
                         curve_fits[f_key][mtrc] = val

                    # print metrics text
                    txt_metrics = ""
                    for mtrc in metrics:
                        txt_metrics+="{} = {}  ".format(mtrc.upper(),round(curve_fits[f_key][mtrc],2))
                    # prints
                    #print(i,freq_data[i],found_center,dif,snr)
                    print("[{}] {:.0f} kHz: Fit found!   t-t0: {:.2f} s   Dif.: {:.2f} s  Log10(S/N): {:.2f}".format(i,freq_data[i],found_center,dif,pk_hgh))
                    print("                             "+txt_metrics)
                    # restart center position if center found
                    prev_center = found_center
            else:
                print("[{}] {:.0f} kHz   : Fit not convergent.".format(i,freq_data[i]))
                unsolved_freqs.append(freq_data[i])

        except Exception as e:
            print("[{}] {:.0f} kHz   : [ERROR] Not processed.".format(i,freq_data[i]))
            print("    ",e)
            unsolved_freqs.append(freq_data[i])
    # I know you might feel compelled to delete this secondary loop, David, but pls dont, trust me
    if(secondary_loop and len(unsolved_freqs)>0):

        print(35*"-","Secondary Loop (unsolved freqs)",35*"-")
        #iterate over all freqs
        for i in range(1,len(freq_data)-1):
            # if freq was unsolved by main loop
            # search for closest defined freqs in each side
            if(freq_data[i] in unsolved_freqs):
                # taget freqs
                f_before = None
                f_after = None
                for i_before in range(i):
                    if(freq_data[i_before] in list(curve_fits.keys())):
                        f_before = freq_data[i_before]
                for i_after in range(len(freq_data)-1,i,-1):
                    if(freq_data[i_after] in list(curve_fits.keys())):
                        f_after = freq_data[i_after]

                if(np.logical_and( f_before!=None , f_after!= None )):
                    popt_bef = curve_fits[f_before]["params"]
                    popt_aft = curve_fits[f_after]["params"]

                    proposed_po = [(popt_bef[p]+popt_aft[p])/2 for p in range(len(popt_aft)) ]
                    prev_center = func.center(popt_aft)


                    # define xy
                    x_ = time_sec
                    y_ = v_data[i,:]
                    # logscale scale Y variable (voltage psd)
                    scale_y = np.log10(y_)
                    if(smooth_pts):
                        scale_y=smooth(scale_y,smooth_pts)

                    try:
                        # target elements for params and param covariances
                        popt=[]
                        pcov=[]

                        # different fuction cases

                        if (func.name() == "gaussian"):
                            #params ->  [mean,dev,scale,bkg]
                            bkg_aprox = np.mean(np.sort(scale_y)[-int(0.3*len(scale_y)):])

                            bnds = ([func.center(popt_aft)-10,0.1,0,np.min(scale_y)],
                            [func.center(popt_bef)+10,np.max(x_),20,bkg_aprox])
                            popt,pcov = cfit(func.f,x_,scale_y,p0=proposed_po,bounds=bnds)

                        elif (func.name() == "fluorescence"):
                            #params ->  [t_on,t_off,scale,bkg,phase]
                            bkg_aprox = np.mean(np.sort(scale_y)[-int(0.3*len(scale_y)):])

                            bnds = ([0,0,0,np.min(scale_y),np.min(x_)],
                            [np.max(x_),np.max(x_),20,bkg_aprox,np.max(x_)])
                            popt,pcov = cfit(func.f,x_,scale_y,p0=proposed_po,bounds=bnds)

                        elif (func.name() == "gauss_gauss" or func.name() == "gauss_exp"):
                            #params ->  [t_on,t_off,tpeak,scale,bkg]

                            
                            bkg_aprox = np.mean(np.sort(scale_y)[-int(0.3*len(scale_y)):])


                            bnds = ([0,0,np.min(x_),0,np.min(scale_y)],
                                    [np.max(x_),np.max(x_),np.max(x_),20,bkg_aprox])
                            
                            popt,pcov = cfit(func.f,x_,scale_y,p0=proposed_po,bounds=bnds)


                        # for cases where parameters were found
                        if (len(popt)>0) :
                            found_center = func.center(popt)
                            # diference with previous found point
                            dif = found_center-prev_center
                            if(i==len(freq_data)-1):
                                dif = 0
                            # discard if center out of bounds
                            if(found_center<(x_[0]*(1-tol)) or found_center>(x_[-1]*(1+tol))):
                                print("[{}] {:.0f} kHz   : Not in bounds! omitted.".format(i,freq_data[i]))
                                failed_freqs.append(freq_data[i])
                            else:

                                # FITTING ERROR

                                pk_hgh = func.peak_height(popt)
                                if (isinstance(pk_hgh, (list, tuple, np.ndarray))):
                                    pk_hgh = pk_hgh[0]
                                
                                snr = 10**(pk_hgh)

                                ## ADD CURVE FIT TO SOLUTIONS
                                f_key = int(freq_data[i])
                                curve_fits[f_key] = {}
                                curve_fits[f_key]["params"]=popt
                                curve_fits[f_key]["covar"]=pcov
                                curve_fits[f_key]["snr"]=snr
                                curve_fits[f_key]["df"]=f_err[i]
                                curve_fits[f_key]["center"]=func.center(popt)
                                curve_fits[f_key]["dev"]=func.dev(popt)
                                curve_fits[f_key]["peak_height"]=pk_hgh
                                curve_fits[f_key]["fit_function"]=func.name()

                                if(func.f_onset):
                                    curve_fits[f_key]["onset"]= func.onset(popt)
                                for mtrc in metrics:
                                     metric_function = get_metric(mtrc)
                                     val = metric_function(x_,scale_y,func.f,popt)
                                     curve_fits[f_key][mtrc] = val

                                # print metrics text
                                txt_metrics = ""
                                for mtrc in metrics:
                                    txt_metrics+="{} = {}  ".format(mtrc.upper(),round(curve_fits[f_key][mtrc],2))
                                # prints
                                print("[{}] {:.0f} kHz: Fit found!   t-t0: {:.2f} s   Dif.: {:.2f} s  Log10(S/N): {:.2f}".format(i,freq_data[i],found_center,dif,pk_hgh))
                                print("                             "+txt_metrics)
                        else:
                            print("[{}] {:.0f} kHz   : Fit not convergent.".format(i,freq_data[i]))
                            failed_freqs.append(freq_data[i])
                    except Exception as e:
                        print("[{}] {:.0f} kHz   : [ERROR] Not processed.".format(i,freq_data[i]))
                        print("    ",e)
                        failed_freqs.append(freq_data[i])

    #print failed freqs
    if(len(unsolved_freqs)+len(failed_freqs)>0):
        print(35*"-","Unsolved frequencies",35*"-")
        if(secondary_loop):
            _ = [print(x,"kHz") for x in failed_freqs]
        else:
            _ = [print(x,"kHz") for x in unsolved_freqs]

    # return fir results + metadata
    fit_results ={
        "freq_fits":curve_fits,
        "metadata":curve_fits_meta
    }

    return fit_results






# def rpw_fit_freq_peaks(rpw_psd,peak_model,date_range,frequency_range=None,initial_pos_guess=None,excluded_freqs=[],dt_fmt="%d-%b-%Y %H:%M:%S"):
#     time_data = rpw_psd["time"]
#     v_data = rpw_psd["v"]
#     freq_data = rpw_psd["frequency"]
#
#
#     #estimate error
#     t_err = np.mean(time_data[1:]-time_data[:-1]).seconds/2.
#     f_err = (freq_data[1:]-freq_data[:-1])/2.
#     print("Estimated uncertainty:")
#     print("  Time: {} s".format(t_err))
#     print("  Freq: between {}-{} kHz".format(np.min(f_err),np.max(f_err)))
#     print("Defining time reference...")
#     # selecting fit intervals
#     date_range_dt = [datetime.strptime(x,dt_fmt) for x  in date_range]
#
#     idx_sel_time = np.logical_and(time_data<=date_range_dt[1],time_data>=date_range_dt[0])
#
#     time_data = time_data[idx_sel_time]
#     v_data = v_data[:,idx_sel_time]
#
#     if(frequency_range):
#         idx_sel_freq = np.logical_and(freq_data<=frequency_range[1],freq_data>=frequency_range[0])
#         freq_data = freq_data[idx_sel_freq]
#         v_data = v_data[idx_sel_freq,:]
#
#     # convering time to seconds
#     time_sec,t0 = dt_to_sec_t0(time_data)
#     time_span = time_sec[-1]-time_sec[0]
#     print(" t0 = ",datetime.strftime(t0,dt_fmt))
#     print("Fitting peaks for {} frequencies between {} kHz and {} kHz".format(len(freq_data),freq_data[0],freq_data[-1]))
#     curve_fits = {}
#     curve_fits_meta = {}
#     #asume that peak of max freq. is close to the beggining of the timespan
#     prev_center = None
#     if(initial_pos_guess):
#         prev_center = dt_to_sec_t0([datetime.strptime(pos_guess,dt_fmt)],t0)[0][0]
#     # starting point (time,freq)
#     starting_point =[]
#     for i in range(len(freq_data)-1,-1,-1):
#
#         if(freq_data[i] in excluded_freqs):
#             print("[{}] {:.0f} kHz   : Excluded!   omitted.".format(i,freq_data[i]))
#             continue
#
#
#         #if not defined, use approx position in timespan (lineal)
#         if(not prev_center):
#             prev_center= time_sec[0]
#
#
#         x_ = time_sec
#         y_ = v_data[i,:]  #V[freqn,date_idx]
#         curve_fits_meta={
#                          "t":x_,
#                          "y":y_,
#                          "t0":t0,
#                          "dt":t_err,
#                          "time_interval":date_range,
#                          "freq_range":frequency_range,
#                          "excluded_f":excluded_freqs
#                         }
#         #fit_bounds = [(1e-18,1e-11),(0,np.max(t_sec0)),(1,1000),(1e-18,1)]
#         #fit_bounds = ((-18.,0.,1.,-18.),(-14.,np.max(x_),time_span,-14.))
#         try:
#             #if(p0):
#
#             init_guess = [np.log10(np.max(y_)),prev_center,60.,-16.]
#             popt,pcov = cfit(peak_model,x_,y_,p0=init_guess, method="lm")#,bounds=fit_bounds)
#
#             # for cases where aprameters were found
#             if (len(popt)>0) :
#                 # diference with previous found point
#                 dif = popt[1]-prev_center
#                 if(i==len(freq_data)-1):
#                     dif = 0
#                 # discard if center out of bounds
#                 if(popt[1]<x_[0]*0.8 or popt[1]>x_[-1]*1.2):
#                     print("[{}] {:.0f} kHz   : Not in bounds! omitted.".format(i,freq_data[i]))
#                     #popt= []
#                     #pcov = []
#                 else:
#                     if(len(starting_point)==0):
#                         starting_point = [freq_data[i],sec_t0_to_dt([popt[1]],t0)[0]]
#                         sp_t=datetime.strftime(starting_point[1],dt_fmt)
#                         #curve_fits_meta={
#                          #"t":x_,
#                          #"y":y_,
#                          #"t0":t0,
#                          #"dt":t_err,
#                          #"df":f_err[i],
#                          #"time_interval":date_range,
#                          #"excluded_f":excluded_freqs
#                         #}
#                         print("Starting point ---------- frequency: {:.0f}+-{:.0f} kHz   time: {}".format(freq_data[i],f_err[i],sp_t))
#
#
#                     # FITTING ERROR
#                     rmse = estimate_rmse(x_,y_,peak_model,popt)
#                     snr = 10**(popt[0]-popt[3])
#
#                     ## ADD CURVE FIT TO SOLUTIONS
#                     curve_fits[freq_data[i]] = {
#                      "params":popt,
#                      "covar":pcov,
#                      "rmse":rmse,
#                      "snr":snr,
#                      "df":f_err[i]}
#
#
#                     print("[{}] {:.0f} kHz: Fit found!   t-t0: {:.2f} s   Dif.: {:.2f} s  Log10(RMSE): {:.2f}  Log10(S/N): {:.2f}".format(i,freq_data[i],popt[1],dif,np.log10(rmse),np.log10(snr)))
#
#                     prev_center = popt[1]#np.mean([time_span * (1-i/len(freq_data))**2,popt[1]])
#
#         except:
#             print("[{}] {:.0f} kHz   : Not found".format(i,freq_data[i]))
#             #popt= []
#             #pcov=[]
#
#     #print(curve_fits.keys())
#     fit_results ={
#         "freq_fits":curve_fits,
#         "metadata":curve_fits_meta
#     }
#
#     return fit_results
def can_combine_fits(fits):

    # time reference and used metrics
    t = fits[0]["metadata"]["time_interval"]
    metrics = fits[0]["metadata"]["metrics"].sort()
    freq =  fits[0]["metadata"]["freq_range"]


    # exit if fits do not have the same ref time or metrics
    for ft in fits:
        t_fit=ft["metadata"]["time_interval"]
        freq_fit=ft["metadata"]["freq_range"]
        metrics_ft = ft["metadata"]["metrics"].sort()
        # if(freq != freq_fit):
        #     print("[Error] frequency range is not the same in all fits")
        #     return False
        if(t != t_fit):
            print("[Error] Reference time is not the same in all fits")
            return False
        if(metrics!=metrics_ft):
            print("[Error] All fits must have the same metrics")
            return False
    return True



def best_fit_values(fits,metric="rmse"):
    # check if the fits correspond in metrics and time
    if(not can_combine_fits(fits)):
            return
    # targat array to save best fits
    best_values={}

    # for all fits, check which has the lower error metric for each frequency
    for ft in fits:
        fit_name=ft["metadata"]["name"]

        # exit if metric does not apply to this fit
        if(metric not in ft["metadata"]["metrics"]):
            print("[Error] The metric '{}' is not in the metrics used for '{}'".format(metric,fit_name))
            return

        # for all freqs (with an optimal fit found by the fit algorithm)
        for f in ft["freq_fits"]:


            params=ft["freq_fits"][f]["params"]
            func = ft["metadata"]["fit_function"]
            metric_val = ft["freq_fits"][f][metric]

            #if freq already listed by other fit, compare and keep lower
            # else, save freq fit
            if(f in best_values):
                if(metric_val<best_values[f][metric]):
                    best_values[f] = ft["freq_fits"][f]
                    best_values[f]["fit_function"] = func.name()
            else:

                best_values[f] = ft["freq_fits"][f]
                best_values[f]["fit_function"] = func.name()
    # sort dictionary entries (frequencies)
    best_values_sort= sorted(best_values)
    best_values = {key:best_values[key] for key in best_values_sort}


    meta = fits[0]['metadata'].copy()
    meta['combined'] = metric
    meta['name'] = '+'.join([x['metadata']['name'] for x in fits])
    return_obj = {'freq_fits':best_values,
                  'metadata':meta}
    return return_obj
def plot_compare_fits(fits,metrics=['rmse'],subplot_size = (9,3)):

    if(not can_combine_fits(fits)):
        return

    n_fits = len(fits)
    n_metrics = len(metrics)

    # Start with a square Figure.
    fig = plt.figure(figsize=(subplot_size[0], subplot_size[1]*n_metrics))
    # Add a gridspec with n_metrics rows and two columns and a ratio of 1 to 4 between

    gs = fig.add_gridspec(n_metrics, 2,  width_ratios=(4, 1),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    axes_sc = []
    axes_kde = []

    for m in range(n_metrics):
        mtrc = metrics[m]

        if(m!=0):
            axes_sc.append(fig.add_subplot(gs[m, 0],sharex=axes_sc[-1]))
            axes_kde.append(fig.add_subplot(gs[m, 1],sharey=axes_sc[-1],sharex=axes_kde[-1]))
        else:
            axes_sc.append(fig.add_subplot(gs[m, 0]))
            axes_kde.append(fig.add_subplot(gs[m, 1],sharey=axes_sc[-1]))

        axes_kde[-1].tick_params(axis="y", labelleft=False)

        for ft in range(n_fits):
            sel_fit = fits[ft]
            metric_val = [sel_fit["freq_fits"][f][mtrc] for f in sel_fit["freq_fits"] ]
            snr_val = [sel_fit["freq_fits"][f]["snr"] for f in sel_fit["freq_fits"] ]
            f = [f for f in sel_fit["freq_fits"] ]

            fit_name = sel_fit["metadata"]["name"]

            axes_sc[-1].scatter(f,metric_val,marker="o",s=snr_val,alpha=0.8,label=fit_name)
            #


            ax_ylims = axes_sc[-1].get_ylim()
            xx = np.linspace(ax_ylims[0],ax_ylims[1],1000)
            kde_val = get_kde(metric_val,xx)
            axes_kde[-1].plot(kde_val,xx)

            axes_sc[-1].set_ylabel(mtrc.upper())

            if(m!= n_metrics-1):
                axes_sc[-1].tick_params(axis="x", labelbottom=False)
                axes_kde[-1].tick_params(axis="x", labelbottom=False,labeltop=True)
            else:
                axes_kde[-1].set_xlabel("KDE")
                axes_sc[-1].set_xlabel("Frequency [kHz]")
            if(m==0):
                axes_sc[-1].text(.01, 1.1, "marker size = S/N ratio", ha='left', va='top', transform=axes_sc[-1].transAxes,fontsize=9)
                axes_sc[-1].legend()
        axes_sc[-1].grid()

        axes_kde[-1].grid()



def rpw_plot_all_fits(fits,rpw_psd,n_cols=2,subplot_size=(5,2),savename=None,
short_tag=None,metric="rmse",observe_freqs=[]):

    color_palette = ["red","blue","limegreen","orange","magenta"]

    freqs = [list(fp["freq_fits"].keys()) for fp in fits]
    freqs = np.sort([item for sublist in freqs for item in sublist])

    n_freqs =len(freqs)
    n_rows = int(n_freqs/n_cols)


    f_ax = fits[0]["metadata"]["f"]
    x_ax = fits[0]["metadata"]["t"]


    plt.figure(figsize=(int(subplot_size[0]*n_cols),int(subplot_size[1]*n_rows)),dpi=150)

    for f in range(len(f_ax)):
        sel_f = int(f_ax[f])
        y_ax = fits[0]["metadata"]["v"][f,:]

        plt.subplot(n_rows,n_cols,f+1)



        color_ix = 0
        for ft in fits:
            fitfun = ft["metadata"]["fit_function"]
            if(int(sel_f) in ft["freq_fits"].keys()):
                pars = ft["freq_fits"][int(sel_f)]["params"]
                mse = ft["freq_fits"][int(sel_f)][metric]
                y_mod = fitfun.f(x_ax,*pars)
                cent = fitfun.center(pars)
                ons = fitfun.onset(pars)

                fname = fitfun.name() if not short_tag else fitfun.name()[:short_tag].upper()

                col=color_palette[color_ix]
                color_ix+=1
                plt.plot(x_ax,y_mod,label="{}  ({})".format(fname,round(mse,2)),color=col)
                plt.axvline(cent,color=col,ls="--")
                plt.axvline(ons,color=col,ls=":")
        if(f==len(f_ax)-1):
            plt.xlabel("Time (s)")
        if(f==0):
            plt.ylabel("Log$_{10}$ PSD(V)")

        if(sel_f in observe_freqs):
            print(f'[{sel_f} kHz] Peak height : ',ft['freq_fits'][sel_f]['peak_height'], '  |  Max. val. point: ',round(np.max(np.log10(y_ax)),2))
            

        plt.plot(x_ax,np.log10(y_ax),color="k",ls="None",marker="x",label="{}kHz (R/Rs = {})".format(sel_f,round(r_from_freq([sel_f],ne_from_r_leblanc)[0],2)))
        leg=plt.legend(labelcolor="linecolor",fontsize=10,alignment="left",borderpad=0.3)
        leg.get_frame().set_alpha(0.3)
        for item in leg.legendHandles:
            item.set_visible(False)
        plt.grid()



    if savename:
        plt.savefig(savename,bbox_inches="tight")



def rpw_plot_fits(fits,hfr_psd,tnr_psd=None,ax=None,ax_tnr=None
,cmap="inferno",fit_limits=True,t_format="%H:%M"):


    # define ax and choose colormap
    cmap = mpl.cm.get_cmap(cmap)


    ax1 = None
    ax2 = None
    ""
    ax = ax if ax else plt.gca()
    if(tnr_psd):
        if(ax and ax_tnr):
            ax1 = ax
            ax2 = ax_tnr
        elif(ax):
            print("Provide the 2 axes for HFR and TNR, otherwise do not provide any.")
            return
        else:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
    else:
        if(ax):
            ax1 = ax
        else:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(211)




    # define spectrogram limits
    lim_date = [np.min(hfr_psd["time"]),np.max(hfr_psd["time"])]
    if(tnr_psd):
        lim_date[0] = np.max(lim_date[0],np.min(tnr_psd["time"]))
        lim_date[1] = np.min(lim_date[1],np.max(tnr_psd["time"]))



    if(not tnr_psd):


        if(fit_limits):
            dt_range=sec_t0_to_dt(meta["t"],meta["t0"])
            dt_range = [dt_range[0]-dt.timedelta(seconds=20),dt_range[-1]+dt.timedelta(seconds=20)]
            dt_range = [datetime.strftime(x,dt_fmt)for x in dt_range]

            rpw_plot_psd(rpw_psd,cmap="binary")#,frequency_range=meta['freq_range'])
            plt.gca().invert_yaxis()
            solar_event(event_type="interval",times={'start':dt_range[0],'end':dt_range[1]},color="blue",linewidth=0.5,hl_alpha=0.2).paint()
            #plt.xlim(dt_range)
        else:
            rpw_plot_psd(rpw_psd,cmap="binary",frequency_range=meta['freq_range'])

    else:
        print("")











    if(fit_limits):
        frequency_range=[int(flist[0]),int(flist[-1])]
        dt_range=sec_t0_to_dt(meta["t"],meta["t0"])
        dt_range = [dt_range[0]-dt.timedelta(seconds=20),dt_range[-1]+dt.timedelta(seconds=20)]
        dt_range = [datetime.strftime(x,dt_fmt)for x in dt_range]

        rpw_plot_psd(rpw_psd,cmap="binary")#,frequency_range=meta['freq_range'])
        plt.gca().invert_yaxis()
        solar_event(event_type="interval",times={'start':dt_range[0],'end':dt_range[1]},color="blue",linewidth=0.5,hl_alpha=0.2).paint()
        #plt.xlim(dt_range)
    else:
        rpw_plot_psd(rpw_psd,cmap="binary",frequency_range=meta['freq_range'])

def rpw_plot_fit_results(fit_results,rpw_psd,cmap="jet",fit_limits=False):

    #dt_fmt="%d-%b-%Y %H:%M:%S"

    # choose colormap
    cmap = mpl.cm.get_cmap(cmap)

    # decopress fit results
    c_fits=fit_results["freq_fits"]
    meta = fit_results["metadata"]

    # order frequency list
    flist = list(c_fits.keys())
    flist = list(np.sort(flist))

    # if freq limits defined, crop
    if(fit_limits):
        frequency_range=[int(flist[0]),int(flist[-1])]
        dt_range=sec_t0_to_dt(meta["t"],meta["t0"])
        dt_range = [dt_range[0]-dt.timedelta(seconds=20),dt_range[-1]+dt.timedelta(seconds=20)]
        dt_range = [datetime.strftime(x,dt_fmt)for x in dt_range]

        rpw_plot_psd(rpw_psd,cmap="binary")#,frequency_range=meta['freq_range'])
        plt.gca().invert_yaxis()
        solar_event(event_type="interval",times={'start':dt_range[0],'end':dt_range[1]},color="blue",linewidth=0.5,hl_alpha=0.2).paint()
        #plt.xlim(dt_range)
    else:
        rpw_plot_psd(rpw_psd,cmap="binary",frequency_range=meta['freq_range'])
    #print("frnge",frequency_range)

    for i in range(len(flist)):

        params = c_fits[flist[i]]["params"]
        covars = c_fits[flist[i]]["covar"]
        if(len(params)>0):

            t0 = meta["t0"]
            ctime = sec_t0_to_dt([params[1]],t0=t0)[0]
            times_sigma1= sec_t0_to_dt([params[1]-params[2],params[1]+params[2]],t0=t0)
            times_sigma1 = times_sigma1[1]-times_sigma1[0]
            ydat = meta["y"]
            f_sigma = c_fits[flist[i]]["df"]

            rgba = cmap(i/len(flist))
            #delays.append(c_fits[flist[i]]["params"][1])
            #freqs.append(flist[i])
            lbl = "{} MHz".format(round(flist[i]/1000.,2))
            if(len(flist)>20 and i%3!=0 and i!=len(flist)-1):
                lbl=None
            plt.errorbar(ctime,int(flist[i]),xerr=np.abs(times_sigma1),yerr=np.abs(f_sigma),color=rgba,
                         label=lbl,marker="o",markersize=3)
    plt.legend(ncol=3,fontsize=9)
    plt.xlabel("Date")
    plt.ylabel("Frequency [MHz]")
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

#
#
# def rpw_freq_drifts(fit_results,excluded_freqs=[],):
#     peak_fits =fit_results["freq_fits"]
#     meta = fit_results["metadata"]
#     flist = np.array(list(peak_fits.keys()))
#     # peak times
#     delays = []
#     # frequencies
#     freqs = []
#     # peak time unceertainty
#     devs = []
#     # freq uncertainty
#     dfs = []
#     #time uncertainty
#     dts = []
#
#
#     for i in range(len(flist)):
#         params_ =peak_fits[flist[i]]["params"]
#         covs_ =peak_fits[flist[i]]["covar"]
#         if(len(params_)>0 and not int(flist[i]) in excluded_freqs ):
#             delays.append(params_[1])
#             freqs.append(flist[i])
#             #error
#             #devs.append(np.sqrt(np.diag(covs_)[1]))
#             dts.append(np.mean(meta["dt"])/2.)
#             devs.append(np.sqrt(np.abs(params_[2])))#(2.*np.mean(peak_fits[flist[i]]["dt"])))
#
#             dfs.append(peak_fits[flist[i]]["df"])
#             #plt.scatter(int(flist[i]),cf[flist[i]]["params"][1], label=i)
#
#     delays = np.array(delays)
#     devs = np.abs(np.array(devs))
#     dfs = np.abs(np.array(dfs))
#     freqs = np.array(freqs)
#
#     #f drift estimation
#     dif_freqs = freqs[1:]-freqs[:-1]
#     dif_delays = delays[1:]-delays[:-1]
#     f_drifts = dif_freqs/dif_delays
#
#     err_delays = devs[:-1]#np.sqrt((devs[1:]**2) + (devs[:-1]**2))
#     err_freqs =  dfs[:-1]#np.sqrt((dfs[1:]**2) + (dfs[:-1]**2))
#
#     #print(err_delays[:],dif_delays[:],err_freqs[:],dif_freqs[:])
#     err_fdrift = np.abs( f_drifts[:]*np.sqrt((err_delays[:]/dif_delays[:])**2 + (err_freqs[:]/dif_freqs[:])**2) )
#
#     return_dict = {"frequencies":flist,
#                    "conv_frequencies": freqs,
#                    "delays":delays,
#                    "freq_drifts":f_drifts,
#                    "sigma_dfdt":err_fdrift,
#                    "sigma_f" :err_freqs,
#                    "sigma_tpeak":err_delays,
#                    "sigma_t":dts}
#     #print([(len(return_dict[x]),x) for x in list(return_dict.keys())])
#
#     #print(f_drifts/1000)
#     return  return_dict



# receivesfrequenciesin kHz
# returns plasma number density in cm-3
def ne_from_freq(freqs,coef=9.):
    return [(f/coef)**2 for f in freqs]

def freq_from_ne(ne,coef=9.):
    return [coef*np.sqrt(n) for n in ne]


# Electron density  models

# input : distance in Rs 
# output: density in cm-3 

# Leblanc 1998
def ne_from_r_leblanc(radius):
    return [ 3.3e5*(r**(-2))+4.1e6*(r**(-4))+8.0e7*(r**(-6)) for r in radius]

# Sittler Guhatakhurta
def ne_from_r_sittler_guhathakurta(radius):
    f_exp = lambda x: 3.26e11*np.exp(3.67/x)
    f_1 = lambda x: x**(-2) + 4.89*x**(-3) + 7.6*x**(-4) + 6*x**(-5)
    return np.array([f_exp(x)*f_1(x)*1e-6 for x in radius])



def ne_from_r_kontar(radius):
    return [ 4.8e9 * (r**-14)+ 3e8 * (r**-6) + 1.4e6 * (r**-2.3) for r in radius]


def dfdn_from_ne(ne,coef=9.):
    return[(coef/2)*(1./np.sqrt(n)) for n in ne]
    



def r_from_ne(nes,ne_model,r_interv=[1,400],n_iter=5,c=0.1,npoints=1000,error=False):

    r_mins = []
    r_err= []
    for  n in nes:
        bounds = r_interv.copy()


        for i in range(n_iter):

            r_span = np.linspace(bounds[0],bounds[1],npoints)
            r_span_l = bounds[1]-bounds[0]
            ne_span = ne_model(r_span)

            r_min = r_span[np.argmin(np.abs(np.array(ne_span)-n))]
            bounds =[max(r_min-c*r_span_l,r_interv[0]),min(r_min+c*r_span_l,r_interv[1])]
        r_err.append(bounds[1]-bounds[0])
        r_mins.append(r_min)
    if(error):
        return r_mins,r_err
    return r_mins
def r_from_freq(freqs,ne_model):
    return r_from_ne(ne_from_freq(freqs),ne_model)

def freq_from_r(r,ne_model):
    return freq_from_ne(ne_model(r))
def dndr_from_r(radius):
    return [ -6.6e5*(r**(-3))-16.4e6*(r**(-5))-48.0e7*(r**(-7)) for r in radius]




def rpw_freq_drifts(fit_res,density_model=ne_from_r_leblanc,excluded_freqs=[],only_negative_drifts=False,allow_vel_over_c=False,use_onsets=False):

    fit_results = dict(fit_res)


    meta = fit_results["metadata"]

    using = 'onset times' if use_onsets else 'peak times'
    print(f'[!] Using {using} to estimate frequency drift rates...')
    peak_fits =fit_results["freq_fits"]

    fit_function = meta['fit_function']

    combined = False
    if('combined' in meta.keys()):
        print("[Warning] you are trying to obtain the frequency drift rates from previously combined/compared fits.")
        print("          This is possible but consider that some exceptions in the script are not well handled.")
    
        combined=True

    flist = np.array(list(peak_fits.keys()))
    # peak times
    delays = []
    # frequencies
    freqs = []
    # peak time unceertainty
    devs = []
    # freq uncertainty
    dfs = []
    #time uncertainty
    dts = []

    # for each frequency, check if the fit is convergent and
    # the frequency is not excluded, then adds it to the list
    for i in range(len(flist)):
        f_selec = flist[i]
        f_fit = peak_fits[f_selec]
        covs_ =f_fit["covar"]
        if(len(f_fit['params'])>0 and not int(flist[i]) in excluded_freqs ):
            # add freq to the list

            freqs.append(flist[i])


            #error
            d_center = 0

            if(use_onsets):
                delays.append(f_fit['onset'])
                d_center=get_fit_function(f_fit['fit_function']).err_onset(covs_)
            else:
                delays.append(f_fit['center'])
                d_center=get_fit_function(f_fit['fit_function']).err_center(covs_)


            #add errors to the list
            #devs.append(np.sqrt(np.diag(covs_)[1]))
            dts.append(np.median(meta["dt"]))
            #devs.append(np.sqrt(np.abs(f_fit['dev']))/2)
            devs.append(d_center)
            #print(f_selec,devs[-1],)
            dfs.append(f_fit["df"])
    
    
    # array all lists
    delays = np.array(delays)
    devs = np.abs(np.array(devs))
    dfs = np.abs(np.array(dfs))
    freqs = np.array(freqs)


    # sort based on frequency
    freqs,delays,devs,dfs=np.array(sorted(zip(freqs,delays,devs,dfs))).T


    # ESTIMATE FREQ DRIFT
    # for two contiguous freqs, estimate df/dt with peaks
    # for the negative cases, continue by estimating velocity parameters
    dif_freqs = []
    dif_delays = []
    f_drifts = []

    df_dne_arr = []
    dne_dr_arr = []
    vel_arr = []
    freq_couples = []
    delay_couples = []
    r_couples = []
    ne_couples = []
    
    
    
    err_delays = []
    err_freqs = []



    for i in range(len(freqs)-1):
        #df
        f_dif = freqs[i+1] - freqs[i]
        #dt
        t_dif = delays[i+1] - delays[i]

        #frequency drift rate
        drift = f_dif/t_dif

        # if only negative drift taken into account
        # UNDDER DEVELOPMENT - not using it is recommended
        if(only_negative_drifts and drift>0):
            continue
        else:

            #append errors - check!

            dev = np.sqrt( (devs[i+1])**2 + (devs[i])**2 )
            df = np.sqrt( (dfs[i+1])**2 + (dfs[i])**2 )
            #dt = (t_dif + dts[i]+dts[i+1])/2

            #velocity params

            #estimate n_e from F
            ne_vals = ne_from_freq([freqs[i],freqs[i+1]])
            # estimate R form n_e
            r_vals = r_from_ne(ne_vals,ne_model=density_model)

            r_couples.append(r_vals)
            ne_couples.append(ne_vals)


            # estimate point-to-point differentials 
            df_dne = f_dif / (ne_vals[1]-ne_vals[0])
            dne_dr = (ne_vals[1]-ne_vals[0])/(r_vals[1]-r_vals[0])

            # derive exciter velocity
            vel_estimate = (1/df_dne)*(1/dne_dr)*drift
            vel_estimate_c = convert_RoSec_to_c([vel_estimate])[0]

            if(vel_estimate<0):
                print("Warning! negative velocity estimate")
                print("  V ",convert_RoSec_to_c([vel_estimate]))
                print("  dfdt ",drift)
                print("  dne_dr ",dne_dr)
                print("  df_dne ",df_dne)
                print("  f ",[freqs[i],freqs[i+1]])
                print("  ne ",ne_vals)
                print("  R ",r_vals)
            if(vel_estimate>0 or only_negative_drifts == False):

                if(vel_estimate_c > 1):
                    print(f"Warning! Non-physical velocity found between {freqs[i]} kHz and {freqs[i+1]} kHz ({round(vel_estimate_c,2)}c)")
                    if(allow_vel_over_c==False):
                        print("... Omitted!")
                        continue
               
                #append Fdrift
                f_drifts.append(drift)
                #append errors
                err_delays.append(dev)
                err_freqs.append(df)
                

                # append differentials
                df_dne_arr.append(df_dne)
                dne_dr_arr.append(dne_dr)

                #append vel
                vel_arr.append(vel_estimate)

                #append pairs
                freq_couples.append([freqs[i],freqs[i+1]])
                delay_couples.append([delays[i],delays[i+1]])
                #append freq and time diffference
                dif_freqs.append(f_dif)
                dif_delays.append(t_dif)


    # save everything as arrays
    dif_freqs = np.array(dif_freqs)
    dif_delays = np.array(dif_delays)
    f_drifts = np.array(f_drifts)
    err_delays = np.array(err_delays)
    err_freqs = np.array(err_freqs)

    delay_couples = np.array(delay_couples)
    freq_couples = np.array(freq_couples)
    
    vel_arr = np.array(vel_arr)
    dne_dr_arr = np.array(dne_dr_arr)
    df_dne_arr = np.array(df_dne_arr)



    err_fdrift = np.abs(np.sqrt((err_delays[:]/dif_delays[:])**2 + (err_freqs[:]/dif_freqs[:])**2) )



    return_dict = {"conv_frequencies": freqs,
                   "delays":delays,
                   "delay_pairs":delay_couples,
                   "freq_pairs":freq_couples,
                   "ne_pairs":ne_couples,
                   "r_pairs":r_couples,
                   "drifts":f_drifts,
                   "dne_dr":dne_dr_arr,
                   "df_dne":df_dne_arr,
                   "dr_dt":vel_arr,
                   "sigma_dfdt":err_fdrift,
                   "sigma_f" :err_freqs,
                   "sigma_center":err_delays,
                   "sigma_t":dts,
                   "drift_on":using}

    fit_results['freq_drifts'] = return_dict

    return  fit_results

def rpw_plot_freq_drift(fit_results,errorbars=False,limit_cases=True):
    if( not 'freq_drifts' in fit_results.keys()):
        print('[ERROR] this fit results data does not have a frequency drift element associated.')
        print('        Try using "rpw_freq_drifts" to estimate the frequency drift rates first.')
        return
    freq_drifts = fit_results['freq_drifts']
    freqs = freq_drifts['conv_frequencies']
    f_drifts = freq_drifts["drifts"]

    maxyerr=np.mean([t for t in freq_drifts["sigma_dfdt"] if np.abs(t)!=np.inf ])
    ax = plt.gca()
    interval_centers=(freqs[1:]+freqs[:-1])/2
    fdrifts_mhz = f_drifts/1000. #in MHz s-1
    for i in range(len(fdrifts_mhz)):
        col = "red" if fdrifts_mhz[i]<0 else "blue"
        yerr=freq_drifts["sigma_dfdt"][i]/1000. #in MHz s-1

        yerr = yerr if yerr!=np.inf else maxyerr/1000.
        if(limit_cases and yerr/np.abs(fdrifts_mhz[i])>=1):
            continue
        #print(yerr/np.abs(fdrifts_mhz[i]))


        ax.scatter(interval_centers[i],np.abs(fdrifts_mhz[i]),c=col,s=8)

        bar_alpha=0.3
        if(errorbars=="both"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),xerr=freq_drifts["sigma_f"][i],yerr=yerr,c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
        elif(errorbars=="x"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),xerr=freq_drifts["sigma_f"][i],c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
        elif(errorbars=="y"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),rerr=yerr,c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]



    neg_patch = mpatches.Patch(color="red", label='df/dt < 0')
    pos_patch = mpatches.Patch(color="blue", label='df/dt > 0')
    plt.legend(handles=[neg_patch,pos_patch],fontsize=8)

   # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

    #for i in range(len(interval_centers)):
        #plt.text(interval_centers[i],fdrifts_mhz[i],str(int(interval_centers[i]))+" MHz",
        #         fontsize=10,horizontalalignment="center",verticalalignment="bottom")
    #plt.ylim(1,5)
    plt.yscale("log")
    plt.xscale("log")#,subs=[1,2,3,4,5,6,7,8,9])
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Frequency drift rate  [MHz sec$^{-1}$]")
    plt.grid(which='both',c='#DDDDDD')


def rpw_plot_fit_summary(rpw_psd,fit_results,freq_drifts,fit_limits=True,savepath=None,grid=True,errorbars=False):

    curve_fits = fit_results["curve_fits"]
    cf_meta = fit_results["metadata"]
    fit_interval = cf_meta["time_interval"]

    fig=plt.figure(figsize=(16,4),dpi=120)
    spec3 = gridspec.GridSpec(ncols=4, nrows=1)

    fig.add_subplot(spec3[0, 1:])
    rpw_plot_fit_results(curve_fits,rpw_psd,fit_limits=fit_limits)

    interv_times = curve_fits

    solar_event(event_type="interval",times={'start':fit_interval[0],'end':fit_interval[1]},hl_alpha=0.4).paint()
    fig.add_subplot(spec3[0,0])
    if(grid):
        plt.grid(which='both',color="#EEEEEE")
        plt.grid(color="#CCCCCC")
    rpw_plot_freq_drift(freq_drifts,errorbars)



    fig.tight_layout()
    if(savepath):
        plt.savefig(savepath,bbox_inches='tight')


def convert_RoSec_to_c(vels):
    #c = 299792.458
    #conv_fact=695700  #695700km = 1 R0

    return [(v*km_per_Rs)/speed_c_kms for v in vels]

def convert_c_to_roSec(vels):
    #c = 299792.458
    #conv_fact= 695700  #695700km = 1 R0
    return [(v*speed_c_kms)/km_per_Rs for v in vels]

# def rpw_estimate_exciter_velocity(fit,density_model=ne_from_r_leblanc,r_interv=[0.1,300],n_iter=10,c=0.01,npoints=5000,weight_v_error=1.,only_neg_drifts=True):
    
#     # frequency drifts previously estimated
#     freq_drifts = fit['freq_drifts']

#     # convergent frequencies ( the ones for which we have a center/onset time )
#     freqs = freq_drifts["conv_frequencies"] 
#     # lower b oundaries of frequency bins
#     freqs_low_bound = freqs[:-1]
#     # frequency avg between two contiguous frequencies
#     freqs = (freqs[1:]+freqs[:-1])/2.


#     # times of center/onset in seconds referent to t0
#     delays = freq_drifts["delays"]

#     delays = (delays[1:]+delays[:-1])/2.


#     # freq drifts and Dt
#     dfdt = freq_drifts["drifts"]
#     dt = freq_drifts["sigma_t"]

#     # if only using negative drifts; select indexes 
#     if(only_neg_drifts):
#         iidx = dfdt<0
#         dfdt = dfdt[iidx]
#         freqs = freqs[iidx]
#         freqs_low_bound = freqs_low_bound[iidx]
#         delays = delays[iidx]



def rpw_estimate_beam_velocity(fit,density_model,r_interv=[0.1,300],n_iter=10,c=0.01,npoints=5000,weight_v_error=1.,only_neg_drifts=True):
    freq_drifts = fit['freq_drifts']
    freqs_conv = freq_drifts["conv_frequencies"]
    freqs_low_bound = freqs_conv[:-1]
    freqs = (freqs_conv[1:]+freqs_conv[:-1])/2.


    delays = freq_drifts["delays"]
    delays = (delays[1:]+delays[:-1])/2.

    dfdt = freq_drifts["drifts"]

    dt = freq_drifts["sigma_t"]

    if(only_neg_drifts):
        iidx = dfdt<0 
        print(len(freqs),' freqs, ',len(dfdt),' dfdts, ',np.sum(iidx),'neg drifts' )
        freqs = freqs[iidx]
        freqs_low_bound = freqs_low_bound[iidx]
        delays = delays[iidx]
        dfdt = dfdt[iidx]



    n_e = ne_from_freq(freqs)
    #print(len(n_e),len(freq_drifts["sigma_f"][:len(freqs)]),len(freqs))
    err_ne = n_e[:]*((2/9)*freq_drifts["sigma_f"][:len(freqs)]/freqs[:])#*((1/6)*(freq_drifts["sigma_f"][:len(freqs)]/freqs[:]))


    rads,err_r = r_from_ne(n_e,density_model,r_interv=r_interv,n_iter=n_iter,c=c,npoints=npoints,error=True)
    err_r = err_r[:] + np.array(rads[:])*(err_ne[:]/n_e[:])

    dfdn=dfdn_from_ne(n_e)
    err_dfdn = dfdn[:]*(freq_drifts["sigma_f"][:len(freqs)]/freqs[:])*np.sqrt(1+(1/36))

    dndr=dndr_from_r(rads)
    err_dndr = np.abs(dndr[:]*(err_ne[:]/n_e[:])*np.sqrt(1+(1/(2*3.3e5+4*4.1e6+6*8.0e7)**2)))

    drdt = []
    drdt_err=[]
    for i in range(len(dfdt)):
        if(dfdt[i]>0):
            continue
        v_trig = dfdt[i]*(dndr[i] **(-1) )*( dfdn[i]**(-1) )
        #print(dfdt[i],dndr[i] ,dfdn[i],v_trig)
        err_v = v_trig*np.sqrt((-freq_drifts["sigma_dfdt"][i]/dfdt[i])**2 + (err_dfdn[i]/dfdn[i])**2 + (err_dndr[i]/dndr[i])**2)

        drdt.append( v_trig )
        drdt_err.append( err_v*weight_v_error)

    if(np.min(np.abs(np.array(drdt)))==np.inf):
        print("Error! Peak convergence failed for every frequency! All the resulting velocities diverge.")
        return


    return_dict = {
        "frequencies":freq_drifts["conv_frequencies"],
        "freq_average":freqs,
        "freq_low_bound":freqs_low_bound,
        "delays":delays,
        "n_e":n_e,
        "r":rads,
        "dfdt":dfdt,
        "drdt":drdt,
        "dndr":dndr,
        "dfdn":dfdn,
        "err_drdt":drdt_err,
        "err_dndr":err_dndr,
        "err_n_e":err_ne,
        "err_r":err_r,
        "dt":dt[:len(freqs)]

    }


    fit['trigger_velocity'] = return_dict
    #print([(len(return_dict[x]),x) for x in list(return_dict.keys())])

    return fit



def rpw_plot_typeIII_diagnostics(rpw_psd,fit_results,freq_drifts,trigger_velocity,figsize=(16,15),dpi=150,errorbars="both",dfdt_errorbars="both",grid=True,fit_limits=False,cmap="jet"):

   # print(freq_drifts["conv_frequencies"])

    peak_fits=fit_results["freq_fits"]
    pf_meta = fit_results["metadata"]
    fit_interval = pf_meta["time_interval"]

    t0 = pf_meta["t0"]

    timeax = sec_t0_to_dt(trigger_velocity["delays"],t0)

    # TIME error
    delta_t = trigger_velocity["dt"]
    delta_t_dt = [dt.timedelta(seconds=t) for t in delta_t]

    terr = freq_drifts["sigma_tpeak"]
    terr_dt=[dt.timedelta(seconds=t) for t in terr]

    r_err = trigger_velocity["err_r"]
    f_err = freq_drifts["sigma_f"]


    cmap = mpl.cm.get_cmap(cmap)
    # create figure and grid
    fig=plt.figure(figsize=figsize,dpi=dpi)
    spec3 = gridspec.GridSpec(ncols=4, nrows=9,wspace=0.4,hspace=0.)

    # PLOT SPECTROGRAM
    ax=fig.add_subplot(spec3[:2, 1:])

    rpw_plot_fit_results(fit_results,rpw_psd,fit_limits=fit_limits)
    plt.gca().invert_yaxis()
    solar_event(event_type="interval",times={'start':fit_interval[0],'end':fit_interval[1]},hl_alpha=0.4,color="#AADDAA").paint()
    ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()
    plt.xlabel("start time: {}".format(datetime.strftime(rpw_psd["time"][0],"%d-%b-%Y %H:%M:%S")))
    #PLOT FREQ. DRIFTS
    ax=fig.add_subplot(spec3[:2,0])
    if(grid):
        plt.grid(which='both',color="#EEEEEE")
        plt.grid(color="#CCCCCC")
    rpw_plot_freq_drift(freq_drifts,errorbars)
    ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()


    # VELOCITY DIAGNOSTICS
    vels = convert_RoSec_to_c(trigger_velocity["drdt"])
    err_vels = convert_RoSec_to_c(trigger_velocity["err_drdt"])



    # PLOT DIAGNOSTICS VS TIME
    ax=fig.add_subplot(spec3[3:4,:2])


    #select in range velocities (physical datapoints)

    selected_datapoints = [x for x in range(len(vels)) if (vels[x]>0 and vels[x]<=1) ]
    mean_selected = np.mean([vels[x] for x in selected_datapoints])
    mean_selected_kev = vc_to_k(mean_selected,units="kev",relativistic=True)

    maxs = [vels[i]+err_vels[i] for i in selected_datapoints]
    mins =[max(vels[i]-err_vels[i],1e-10) for i in selected_datapoints]
    #print(trigger_velocity["drdt"],trigger_velocity["err_drdt"],selected_datapoints)
    bot_avg = np.mean(mins)
    top_avg = np.mean(maxs)


    #ax.axhline(top_avg,c="grey",linestyle="--")
    #ax.axhline(bot_avg,c="grey",linestyle="--")

    plt.axhspan(bot_avg, top_avg, color="lightgrey", alpha=0.5)

    ax.axhline(mean_selected,c="k",linestyle="--",label="average = {:.2f} c = {:.2f} keV".format(mean_selected,mean_selected_kev))
    #ax.axhline(1,c="r",linestyle="--",label="speed of light")

    ax.set_xlabel("$t-t_0$ [sec] ",fontsize=13)
    ax.set_ylabel("v/c",fontsize=13)


    for f_i in range(len(trigger_velocity["freq_average"])):
        if(vels[f_i]<0 or vels[f_i]>1):
            err_vels[f_i]=0
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax.scatter(trigger_velocity["delays"][f_i],vels[f_i],color=rgba,marker="o",s=20)
        ax.errorbar(trigger_velocity["delays"][f_i],vels[f_i],xerr=terr[f_i]+delta_t[f_i],yerr=np.abs(err_vels[f_i]),c=rgba,markersize=2)


    ax.xaxis.tick_top()
    #plt.ylim(1e-3,10)

    bot_lim = np.min(mins)/2. if np.min(mins) else 1e-3
    bot_lim = max(bot_lim,1e-3)
    plt.ylim(bot_lim,1)
    ax.set_yscale("log")
    ax.xaxis.set_label_position('top')
    xmin, xmax = ax.get_xlim()
    xmin,xmax = sec_t0_to_dt([xmin,xmax],t0)

    #ax.legend()


    ax2=fig.add_subplot(spec3[4:5,:2])
    ne_err = trigger_velocity["err_n_e"]
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax2.scatter(timeax[f_i],trigger_velocity["n_e"][f_i],color=rgba,marker="o",s=20)
        ax2.errorbar(timeax[f_i],trigger_velocity["n_e"][f_i],yerr=np.abs(ne_err[f_i]),xerr=terr_dt[f_i]+delta_t_dt[f_i],c=rgba)
    ax2.set_yscale("log")
    ax2.set_ylabel("$n_e$ [cm$^{-3}$]",fontsize=13)
    ax2.set_xticks([])
    plt.xlim(xmin,xmax)

    fig.add_subplot(spec3[5:6,:2])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        plt.scatter(timeax[f_i],trigger_velocity["r"][f_i],color=rgba,marker="o",s=20,label="{} kHz".format(int(trigger_velocity["frequencies"][f_i])))
        plt.errorbar(timeax[f_i],trigger_velocity["r"][f_i],xerr=terr_dt[f_i]+delta_t_dt[f_i],yerr=np.abs(r_err[f_i]),c=rgba,markersize=2)
    plt.xlabel("Time (UT)  $t_0$ = {}".format(datetime.strftime(t0,std_date_fmt)),fontsize=13)
    plt.ylabel("r $[R_o]$",fontsize=13)
    plt.xticks()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    #plt.legend(fontsize=7,ncol=4)
    plt.xlim(xmin,xmax)


    # PLOT DIAGNOSTICS VS R

    ax=fig.add_subplot(spec3[3:4,2:])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        r_AU = trigger_velocity["r"][f_i]*(1/Rs_per_AU)
        r_err_AU = r_err[f_i]*(1/Rs_per_AU)
        if(vels[f_i]<0):
            err_vels[f_i]=0
        ax.errorbar(r_AU,vels[f_i],yerr=np.abs(err_vels[f_i]),xerr=np.abs(r_err_AU),c=rgba,marker="o",markersize=2)

        ax.scatter(r_AU,vels[f_i],color=rgba,marker="o",s=20)
    plt.yscale("log")

    plt.axhspan(bot_avg, top_avg, color="lightgrey", alpha=0.5)

    ax.axhline(mean_selected,c="k",linestyle="--",label="average = {:.2f} c = {:.2f} keV".format(mean_selected,mean_selected_kev))
    #ax.axhline(1,c="r",linestyle="--",label="speed of light")



    plt.xlabel("r $[AU]$",fontsize=13)
    plt.ylabel("v/c",fontsize=13)
    plt.ylim(max(1e-3,np.min(mins)/2.),1)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.yaxis.tick_right()

    #ax.yaxis.set_label_position('right')


    plt.legend(fontsize=13)

    ax2=fig.add_subplot(spec3[4:5,2:])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax2.errorbar(trigger_velocity["r"][f_i],trigger_velocity["n_e"][f_i],yerr=ne_err[f_i],xerr=r_err[f_i],c=rgba)
        ax2.scatter(trigger_velocity["r"][f_i],trigger_velocity["n_e"][f_i],color=rgba,marker="o",s=20)
    plt.yscale("log")

    plt.ylabel("$n_e$ [cm$^{-3}$]",fontsize=13)
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    #ax2.yaxis.set_label_position('right')


    #print(trigger_velocity["freq_average"])
    ax3=fig.add_subplot(spec3[5:6,2:],)
    for f_i in range(len(trigger_velocity["freq_low_bound"])):
        #print(freq_drifts["conv_frequencies"][f_i])
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_low_bound"]))
        lbl = "{} MHz".format(round(trigger_velocity["freq_low_bound"][f_i]/1000.,2))
        if(len(trigger_velocity["freq_average"])>15 and f_i%2!=0 and f_i!=len(trigger_velocity["freq_average"])-1):
                lbl=None
        plt.scatter(trigger_velocity["r"][f_i],trigger_velocity["freq_average"][f_i],color=rgba,marker="o",s=20,label=lbl)
        ax3.errorbar(trigger_velocity["r"][f_i],trigger_velocity["freq_average"][f_i],xerr=r_err[f_i],yerr=f_err[f_i],c=rgba)
    plt.xlabel("r $[R_o]$",fontsize=13)
    plt.ylabel("Frequency [Hz]",fontsize=13)
    plt.yscale("log")
    plt.legend(fontsize=9,ncol=3)
    #plt.yscale("log")
    #plt.xscale("log")
    ax3.yaxis.tick_right()

    #ax3.yaxis.set_label_position('right')



def plot_curve_fits(fit_results,onset=True,metric="rmse",ax=None,colorbar=True,curve_color=None,xylog=[False,False]):

    ax = ax if ax else plt.gca()

    color_list = curve_color if curve_color else  ["dodgerblue","tomato","c"]

    freqss = [list(fp["freq_fits"].keys()) for fp in fit_results]
    freqss = np.sort([item for sublist in freqss for item in sublist])

    compared_fits = best_fit_values(fit_results)

    t_= compared_fits['metadata']['t']
    f = compared_fits['metadata']['f']
    z = compared_fits['metadata']['v']

    cm= ax.pcolormesh(t_,f,np.log10(z),shading="auto",cmap="binary_r")
    if(colorbar):
        plt.colorbar(cm,label="$Log_{10}$ PSD (V)",format=scaled_int_ax_formatter(out='formatter'))



    for i in range(len(fit_results)):
        fit_result= fit_results[i]
        fit_name = fit_result['metadata']['name']

        freqs = list(fit_result['freq_fits'].keys())

        centers = [fit_result['freq_fits'][x]['center'] for x in freqs]
        ax.scatter(centers,freqs,c=color_list[i],label=f'{fit_name} (peak)')

        if(onset):
            onsets = [fit_result['freq_fits'][x]['onset'] for x in freqs]
            ax.scatter(onsets,freqs,facecolor="None",edgecolor=color_list[i],label=f'{fit_name} (onset)',lw=2)

    freqs_bf = list(compared_fits['freq_fits'].keys())
    centers_bf = [compared_fits['freq_fits'][x]['center'] for x in freqs_bf]
    ax.plot(centers_bf,freqs_bf,c="limegreen",label=f'{metric} best fit (peak)')

    if(onset):
        onsets_bf = [compared_fits['freq_fits'][x]['onset'] for x in freqs_bf]
        ax.plot(onsets_bf,freqs_bf,c="limegreen",ls="--",label=f'{metric} best fit (onset)')

    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylabel('Frequency (kHz)')



    ax.grid()
    ax.invert_yaxis()
    if xylog[1]:
        ax.set_yscale("log")
    if xylog[0]:
        ax.set_xscale("log")





def plot_velocity_results(vel_fits,color_list=None,reg_last_n=None,fdr_last_n=None,figsize=[11,9],errorbars=[True,True]):

    color_list = color_list if color_list else ["r","g","b","m"]
    plt.figure(figsize=figsize,dpi=200)




    plt.subplot(211)
    plt.title("Velocity estimation with Frequency Drift Rate")


    for fp in range(len(vel_fits)):
        fit = vel_fits[fp]
        fitname = f"{fit['metadata']['name']} ({fit['freq_drifts']['drift_on']})"

        freqs = fit["trigger_velocity"]["delays"]

        vel = fit["trigger_velocity"]["drdt"]
        vel_c = np.array(convert_RoSec_to_c(vel))
        iix = vel_c<1
        vel_c= vel_c[iix]
        freqs=np.array(freqs)[iix]
        vel_c_err = np.array(convert_RoSec_to_c(fit["trigger_velocity"]["err_drdt"]))[iix]
        err_delays = np.array(fit["trigger_velocity"]['dt'])[iix]

        if(fdr_last_n):
            vel_avg = np.mean(vel_c[:fdr_last_n])
        else:
            vel_avg = np.mean(vel_c)
        energ_avg = vc_to_k(vel_avg,units="kev")


        plt.axhline(vel_avg,c=color_list[fp],ls=":",label=f"Avg velocity ={round(vel_avg,2)} c ({round(energ_avg,2)} keV)")
        if(errorbars[0]):
            plt.errorbar(freqs,vel_c,xerr=err_delays,yerr=vel_c_err,c=color_list[fp],marker="o",lw=1.5,label=fitname,linestyle='None')
        else:
            plt.scatter(freqs,vel_c,c=color_list[fp],marker="o",lw=1.5,label=fitname)
        plt.ylim(0.01,1)
        plt.yscale("log")
        #plt.xscale("log")
        plt.ylabel("Velocity (v/c)",fontsize=14)
        #plt.xlabel("Time (sec)",fontsize=14)
        plt.legend()
        plt.grid(True)


    plt.subplot(212)
    plt.title("Velocity estimation with Linear Regression")


    for fp in range(len(vel_fits)):
        fit = vel_fits[fp]
        fitname = f"{fit['metadata']['name']} ({fit['freq_drifts']['drift_on']})"

        freqs = fit["trigger_velocity"]["freq_average"]
        delays = fit["trigger_velocity"]["delays"]
        rads = fit["trigger_velocity"]["r"]

        err_delays = fit["trigger_velocity"]['dt']
        err_rads = fit["trigger_velocity"]['err_r']

        # linear regression
        m=0
        b=0
        if(reg_last_n):
            print(f"[!] using first {reg_last_n} points to estimate regression (lower freqs)")
            m, b = np.polyfit(delays[:reg_last_n], rads[:reg_last_n], 1)
        else:
            m, b = np.polyfit(delays, rads, 1)

        t_Ro = m+b
        onset_time = sec_t0_to_dt([t_Ro],fit['metadata']['t0'])[0]

        print( fitname, 'Ro = 1 Rs at t =',round(t_Ro,2),f"s ({onset_time})")

        avg_vel = convert_RoSec_to_c([m])[0]

        avg_energy = vc_to_k(avg_vel,units="kev")


        if(errorbars[1]):
            plt.errorbar(delays,rads,xerr=err_delays,yerr=err_rads,c=color_list[fp],label=f"{fitname}",marker="o",linestyle='None')
        else:
            plt.scatter(delays,rads,c=color_list[fp],label=f"{fitname}",marker="o")
        plt.ylabel("Distance ($R_\odot$)",fontsize=14)
        plt.xlabel("Time (sec)",fontsize=14)
        plt.plot(delays,m*delays+b,c=color_list[fp],ls=":",label=f"Velocity = {round(avg_vel,2)} c ({round(avg_energy,2)} keV)")
        plt.grid(True)
        plt.legend()




# TODO move to final loc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def flatten_pairs(pairs_list):
    els = []
    for i in range(len(pairs_list)):
        pl = pairs_list[i]
        for j in range(len(pl)):
            
            if(not pl[j] in els):
                els.append(pl[j])

    return els
def test_plot_velocity_results(vel_fits,color_list=None,reg_last_n=None,fdr_last_n=None,figsize=[9,9],errorbars=[True,True]):

    color_list = color_list if color_list else ["r","g","b","m"]
    plt.figure(figsize=figsize,dpi=250)




    ax1=plt.subplot(211)
    ax1.set_title("Velocity estimation with Frequency Drift Rate")

    vel_groups= []

    for fp in range(len(vel_fits)):
        fit = vel_fits[fp]
        fitname = f"{fit['metadata']['name']} ({fit['freq_drifts']['drift_on']})"

        freqs = fit["freq_drifts"]["delay_pairs"]

        vel = fit["freq_drifts"]["dr_dt"]
        vel_c = np.array(convert_RoSec_to_c(vel))

        vel_groups.append(vel_c)
        
        #vel_c= vel_c
        freqs=np.array(freqs)
        
        #TODO REMOVE/CORRECT
        vel_c_err = np.array(convert_RoSec_to_c(fit["freq_drifts"]["sigma_f"]))
        err_delays = np.array(fit["freq_drifts"]['sigma_center'])


        
        if(fdr_last_n):
            vel_avg_a = np.mean(vel_c[:fdr_last_n])
            vel_avg_g = 10**np.mean(np.log10(vel_c[:fdr_last_n]))
        else:
            vel_avg_a = np.mean(vel_c)
            vel_avg_g = 10**np.mean(np.log10(vel_c))
        energ_avg_a = vc_to_k(vel_avg_a,units="kev")
        energ_avg_g = vc_to_k(vel_avg_g,units="kev")


        ax1.axhline(vel_avg_a,c=color_list[fp],ls="--",label=f"A mean V ={round(vel_avg_a,2)} c ({round(energ_avg_a,2)} keV)")
        ax1.axhline(vel_avg_g,c=color_list[fp],ls=":",label=f"G  mean V ={round(vel_avg_g,2)} c ({round(energ_avg_g,2)} keV)")
        if(errorbars[0]):
            ax1.errorbar(freqs[:,0],vel_c,xerr=err_delays,yerr=vel_c_err,c=color_list[fp],marker="o",lw=1.5,label=fitname,linestyle='None')
        else:
            ax1.scatter(freqs[:,0],vel_c,c=color_list[fp],marker="o",lw=1.5,label=fitname)
        ax1.set_ylim(top=1)
        ax1.set_yscale("log")
        #plt.xscale("log")
        ax1.set_ylabel("Velocity (v/c)",fontsize=14)
        #plt.xlabel("Time (sec)",fontsize=14)
        ax1.legend(loc="upper left",ncols=2)
        ax1.grid(True)


    subax1 = inset_axes(ax1,width='35%',height='25%',loc='lower right',
                       borderpad=2,axes_kwargs={"alpha":0.7})
    xx = np.linspace(-2,0)
    vel_avgs = []
    for fp in range(len(vel_fits)):
        vels = vel_groups[fp]
        kde_l = lambda x: get_kde(np.log10(vels),xx,l=x)

        avg_a = np.log10(np.mean(vels))
        avg_g = np.mean(np.log10(vels))
        vel_avgs.append([avg_a,avg_g])

        subax1.plot(xx,kde_l(0.2),c=color_list[fp])
        subax1.axvline(avg_a,c=color_list[fp],ls='--')
        subax1.axvline(avg_g,c=color_list[fp],ls=":")

    
    subax1.set_yticklabels([])
    ymin_plot, ymax_plot = subax1.get_ylim()
    plt.text(-2,ymin_plot*1.1,r'KDE Log$_{10}$(v/c)', horizontalalignment='left',verticalalignment='bottom' ,weight='bold',color='k')

    for fp in range(len(vel_fits)):
        avg_a,avg_g = vel_avgs[fp]
        

        plt.text(avg_g,ymax_plot*1.1,f'{round(10**avg_g,2)}c', horizontalalignment='center',verticalalignment='bottom' ,rotation=90,weight='bold',color=color_list[fp])
        plt.text(avg_a,ymax_plot*1.1,f'{round(10**avg_a,2)}c', horizontalalignment='center',verticalalignment='bottom' ,rotation=90,weight='bold',color=color_list[fp])

    subax1.grid(True)



    ax2=plt.subplot(212)
    ax2.set_title("Velocity estimation with Linear Regression")



    inst_vels = []
    for fp in range(len(vel_fits)):
        fit = vel_fits[fp]
        fitname = f"{fit['metadata']['name']} ({fit['freq_drifts']['drift_on']})"

        #freqs = np.array([np.mean(x) for x in fit["freq_drifts"]["freq_pairs"] ])
        #delays = fit["freq_drifts"]["delay"]
        #rads = fit["freq_drifts"]["radii"]

        freqs = np.array(flatten_pairs(fit["freq_drifts"]["freq_pairs"]))
        delays = np.array(flatten_pairs(fit["freq_drifts"]["delay_pairs"]))
        rads = np.array(flatten_pairs(fit["freq_drifts"]["r_pairs"]))
        


        instant_vels = (rads[1:]-rads[:-1])/(delays[1:]-delays[:-1])
        inst_vels.append(instant_vels)


        err_delays = fit["freq_drifts"]['sigma_t']
        err_rads = fit["freq_drifts"]['sigma_f']

        # linear regression
        m=0
        b=0
        if(reg_last_n):
            print(f"[!] using first {reg_last_n} points to estimate regression (lower freqs)")
            m, b = np.polyfit(delays[:reg_last_n], rads[:reg_last_n], 1)
        else:
            m, b = np.polyfit(delays, rads, 1)

        t_Ro = -b/m
        onset_time = sec_t0_to_dt([t_Ro],fit['metadata']['t0'])[0]

        print( fitname, 'Ro = 0 Rs at t =',round(t_Ro,2),f"s ({onset_time})")

        avg_vel = convert_RoSec_to_c([m])[0]

        avg_energy = vc_to_k(avg_vel,units="kev")


        if(errorbars[1]):
            ax2.errorbar(delays,rads,xerr=err_delays,yerr=err_rads,c=color_list[fp],label=f"{fitname}",marker="o",linestyle='None')
        else:
            ax2.scatter(delays,rads,c=color_list[fp],label=f"{fitname}",marker="o")
        ax2.set_ylabel("Distance ($R_\odot$)",fontsize=14)
        ax2.set_xlabel("Time (sec)",fontsize=14)
        ax2.plot(delays,m*delays+b,c=color_list[fp],ls="-",lw=1,label=f"Regresion velocity = {round(avg_vel,2)} c ({round(avg_energy,2)} keV)")
    ax2.grid(True)
    ax2.legend(loc="upper left")
    
    
    
    subax2 = inset_axes(ax2,width='35%',height='25%',loc='lower right',borderpad=2)
    xx = np.linspace(-2,0)
    vel_avgs=[]
    for fp in range(len(vel_fits)):
        vels = np.array(inst_vels[fp])
        vels = vels[vels>0]
        kde_l = lambda x: get_kde(np.log10(vels),xx,l=x)

        avg_a=np.log10(np.mean(vels))
        avg_g=np.mean(np.log10(vels))
        vel_avgs.append([avg_a,avg_g])

        subax2.plot(xx,kde_l(0.2),c=color_list[fp])
        subax2.axvline(avg_a,c=color_list[fp],ls='--')
        subax2.axvline(avg_g,c=color_list[fp],ls=":")



    subax2.set_yticklabels([])
    ymin_plot, ymax_plot = subax2.get_ylim()
    plt.text(-2,ymin_plot*1.1,r'KDE Log$_{10}$(v/c)', horizontalalignment='left',verticalalignment='bottom' ,weight='bold',color='k')

    for fp in range(len(vel_fits)):
        avg_a,avg_g = vel_avgs[fp]
        

        plt.text(avg_g,ymax_plot*1.1,f'{round(10**avg_g,2)}c', horizontalalignment='center',verticalalignment='bottom' ,rotation=90,weight='bold',color=color_list[fp])
        plt.text(avg_a,ymax_plot*1.1,f'{round(10**avg_a,2)}c', horizontalalignment='center',verticalalignment='bottom' ,rotation=90,weight='bold',color=color_list[fp])

    subax2.grid(True)

        # TODO 
        # add v/c annotations on top of kde
        # hide kde yaxis
        # create table with plot generator (Amean_FDR,Gmean_FDR,Amean_inst,G_mean_inst,reg)
        # create sns pairplot
