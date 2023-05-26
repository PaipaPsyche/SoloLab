from .values import *
import os
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime,time,timedelta

from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table
import astropy.units as u
import astropy.constants as cs

 # STIX data read

def stix_create_counts(pathfile, is_bkg=False,time_arr=None,correct_flight_time=False,energy_shift=0):

    #is spectrogram file
    spectrogram_file=False

    print("Extracting info: ")
    infos = os.path.basename(pathfile).split("-")[0].split("_")
    #dts = [datetime.strptime(x,"%Y%m%dT%H%M%S") for x in os.path.basename(pathfile).split("_")[3].split("-")]
    #dts = [datetime.strftime(x,dt_fmt)for x in dts]
    txt_type = "{} {}".format(infos[2] ,infos[1]).upper()


    print("  File: ",os.path.basename(pathfile))
    if("spe" in os.path.basename(pathfile)):
        spectrogram_file=True
        print("  Type: ",infos[2],"spectrogram","BKG" if is_bkg else "")
    else:
        print("  Type: ",txt_type,"BKG" if is_bkg else "")
    #output  dict
    return_dict = {}
    #fits info
    hdulist = fits.open(pathfile)
    header = hdulist[0].header
    earth_sc_delay = header["EAR_TDEL"] if correct_flight_time else 0

    # print elapsed time and distance from the sun
    try:
        print("  Obs. elapsed time: ",round((Time(header["DATE_END"])-Time(header["DATE_BEG"])).to(u.s).value/60,2),"minutes")
    except:
        print("  Obs. elapsed time: ",round((Time(header["DATE-END"])-Time(header["DATE-BEG"])).to(u.s).value/60,2),"minutes")


    try:
        dist_sun_sc=(header["DSUN_OBS"]*u.m).to(u.au)
        print(f"  Distance s/c - sun: {np.round(dist_sun_sc,3)}")
    except:
        dist_sun_sc=(header["DSUN-OBS"]*u.m).to(u.au)
        print(f"  Distance s/c - sun: {np.round(dist_sun_sc,3)}")


    data = Table(hdulist[2].data)
    #sum over all detectors and pixels (optional)
    data_counts = None
    if(spectrogram_file):
        data_counts = data["counts"]
    else:
        data_counts = np.sum(data['counts'],axis=(1,2))

    energies = Table(hdulist[3].data)
    if(not "channel" in list(energies.columns)): # optional to detect when frame is not ther
        energies = Table(hdulist[4].data)
        print("  [!] frame 3 had no energy data, using frame 4 instead.")
    # apply energy shift. default 0
    if(energy_shift!=0):
        print(f"  [!] Energy shift : {energy_shift} keV  (Range {list(energies['e_low'])[0]}-{list(energies['e_low'])[-1]} keV is now in range {list(energies['e_low'])[0]+energy_shift}-{list(energies['e_low'])[-1]+energy_shift} keV)")
        energies["e_low"]=[x+energy_shift for x in energies["e_low"].quantity[:].value]
        energies["e_high"]=[x+energy_shift for x in energies["e_high"].quantity[:].value]

    #get UNITS
    data_units = list(hdulist[2].data.columns.units)
    data_names = list(hdulist[2].data.columns.names)
    units_time = u.cs if data_units[data_names.index("time")]=='cs' else u.s
    units_timedel = u.cs if data_units[data_names.index("timedel")]=='cs' else u.s

    # units to seconds
    data["time"] = data["time"]*units_time.to(u.s)
    data["timedel"] = data["timedel"]*units_timedel.to(u.s)



    # get cts_per_sec
    n_energies = len(energies)
    print ("  Energy channels extracted: ",n_energies)

    # number of energy channels available.
    min_channels = min(n_energies,np.shape(data_counts)[-1])
    if(min_channels!=n_energies):
        print(f"  [!] Counts array: only  {min_channels} energy channels available.")
        energies = energies[:min_channels]



    #normalise by time_bin duration ("timedel" keyword)
    if is_bkg and np.shape(data['timedel'])[0]>1:
        data_counts = np.sum(data_counts,axis=0)
        timedel = np.sum(data['timedel'])
        timedel = timedel*units_timedel.to(u.s)
        data_counts_per_sec = np.reshape(data_counts/timedel,(n_energies))
    else:


        data_counts_per_sec = np.reshape(data_counts/data['timedel'],(np.shape(data_counts)[-1])) if is_bkg else data_counts/data['timedel'].reshape(-1,1)

    # for bakground create array of constant bkg cts/sec value per energy bin
    if is_bkg:
        bkg_arr = []
        for i in range(len(time_arr)):
            bkg_arr.append(data_counts_per_sec)
        verbose_dts(np.array(time_arr))
        return_dict = {"time":time_arr,
                       "counts_per_sec":bkg_arr
        }
    # for L1 images, return energy info , cts/sec/bin, time array
    else:

        max_e = np.max(energies["e_low"])
        mean_energy = [(min(max_e+1,e_high)+e_low)/2 for chn,e_low,e_high in energies]


        data_time=None


        try:
            data_time = Time(header['DATE_BEG']) + TimeDelta(data['time'] * u.s)+TimeDelta(earth_sc_delay * u.s)

        except:
            data_time = Time(header['DATE-BEG']) + TimeDelta(data['time'] * u.s)+TimeDelta(earth_sc_delay * u.s)
        data_time = [t.datetime for t in data_time]



        verbose_dts(np.array(data_time))
    # counts object, input  for plotting and spectral analysis routines
        return_dict = {"time":data_time,
                   "counts_per_sec":data_counts_per_sec,
                   "energy_bins":energies,
                   "mean_energy":mean_energy}
    return return_dict






def stix_remove_bkg_counts(pathfile,pathbkg=None,stix_bkg_range=None,correct_flight_time = False,energy_shift=0):
    #import L1 data
    data_L1 = stix_create_counts(pathfile,energy_shift=energy_shift)

    bkg_count_spec = None
    data_counts_per_sec_nobkg = np.array(data_L1["counts_per_sec"])
    min_channels = np.shape(data_L1["counts_per_sec"])[-1]

    if(pathbkg):

        #import BKG data
        data_BKG = stix_create_counts(pathbkg,is_bkg=True, time_arr=data_L1["time"],correct_flight_time =correct_flight_time,energy_shift=energy_shift)

        #subtract background
        min_channels = min(np.shape(data_L1["counts_per_sec"])[-1],np.shape(data_BKG["counts_per_sec"])[-1])
        print(f"  [!] STIX Background subtraction: {min_channels} energy channels used")
        data_counts_per_sec_nobkg = np.array(data_L1["counts_per_sec"])[:,:min_channels]-np.array(data_BKG["counts_per_sec"])[:,:min_channels]

        #create bkg spectrum
        bkg_count_spec=data_BKG["counts_per_sec"][0]

    if(stix_bkg_range is not None):
        date_range=[datetime.strptime(x,std_date_fmt) for x in stix_bkg_range]
        d_idx = np.array([True if np.logical_and(x>=date_range[0],x<=date_range[1]) else False for x in data_L1["time"]])



        bkg_interv_spec = np.mean(data_counts_per_sec_nobkg[d_idx,:],axis=0)[:min_channels]
        bkg_array =  np.array([bkg_interv_spec for i in range(np.shape(data_counts_per_sec_nobkg)[0])])
        data_counts_per_sec_nobkg = data_counts_per_sec_nobkg - bkg_array

        if(bkg_count_spec is not None):
            for i in range(min_channels):
                bkg_count_spec[i] = bkg_count_spec[i] + bkg_interv_spec[i]
        else:
            bkg_count_spec=bkg_interv_spec



    # replace ctc/secinfo with corrected info
    return_dict = data_L1.copy()
    return_dict["counts_per_sec"] = data_counts_per_sec_nobkg
    return_dict["background"]=bkg_count_spec

    return return_dict

def stix_combine_files(filenames,bkgfile=None,correct_flight_time=False,stix_bkg_range=None,force=False,energy_shift=0):
    if(not stix_bkg_range):
        stix_bkg_range = [None for n in filenames]
    if bkgfile:
        return stix_combine_counts([stix_remove_bkg_counts(x,bkgfile,stix_bkg_range=y,energy_shift=energy_shift) for x,y in zip(filenames,stix_bkg_range)],force=force)
    else:
        return stix_combine_counts([stix_create_counts(x,stix_bkg_range=y,correct_flight_time=correct_flight_time,energy_shift=energy_shift) for x,y in zip(filenames,stix_bkg_range)],force=force)

def stix_combine_counts(allcounts,force=False):

    allcounts.sort(key=lambda x: x["time"][0], reverse=False)

    eranges = [x["energy_bins"] for x in allcounts]
    same_eranges = all(len(x)==len(eranges[0]) for x in eranges)


    # merging conditions
    same_energy_bins = same_eranges


    if(not same_eranges):
        print("Warning! count objects to merge do not have the same energy bins")
        print("   number of bins:")
        print("   ",[len(x) for x in eranges])
    else:
        for i in allcounts[0]["energy_bins"]["channel"]:
            all_chan = [(allcounts[j]["energy_bins"]["e_low"][i],allcounts[j]["energy_bins"]["e_high"][i]) for j in range(len(allcounts))]
            same_echan = all(x==all_chan[0] for x in all_chan)
            if(not same_echan):
                print("Warning! count objects to merge do not have the same energy bins")
                print("   inconsistent channel energies: channel ",i)
                print("   energy limits:",all_chan)
                same_energy_bins=False
    timedelts = [(allcounts[a]["time"][0]-allcounts[a-1]["time"][-1]).seconds for a in range(1,len(allcounts))]
    print("Time gaps between files in seconds (negatives mean overlapping):",timedelts)

    if(same_energy_bins):
        new_time = []
        new_cts_per_sec = []
        energy_bins = allcounts[0]["energy_bins"]
        mean_e = allcounts[0]["mean_energy"]
        for i in range(len(allcounts)):
            cts_ = allcounts[i]
            time_=cts_["time"]
            for t in range(len(time_)):
                if(len(new_time)==0 or (time_[t]>new_time[-1])):
                    new_time.append(time_[t])
                    new_cts_per_sec.append(cts_["counts_per_sec"][t,:])



        new_cts_obj={
        "time":new_time,
        "energy_bins":energy_bins,
        "mean_energy":mean_e,
        "counts_per_sec":np.array(new_cts_per_sec),
        }
        if("background" in allcounts[0].keys()):
            new_cts_obj["background"]=allcounts[0]["background"]
        return new_cts_obj
    elif(force):
        print(" Warning! Energy bins inconsistency found. Forcing merge (there could be merging mistakes in energy bins definition)")
        #print("   ",[print(*ac["energy_bins"]) for ac in allcounts])

        new_time = []
        new_cts_per_sec = []

        max_echannels  = np.min([len(ac["energy_bins"]) for ac in allcounts])
        energy_bins = allcounts[0]["energy_bins"][:max_echannels]
        mean_e = allcounts[0]["mean_energy"][:max_echannels]
        for i in range(len(allcounts)):
            cts_ = allcounts[i]
            time_=cts_["time"]
            for t in range(len(time_)):
                if(len(new_time)==0 or (time_[t]>new_time[-1])):
                    new_time.append(time_[t])
                    new_cts_per_sec.append(cts_["counts_per_sec"][t,:max_echannels])
        new_cts_obj={
        "time":new_time,
        "energy_bins":energy_bins,
        "mean_energy":mean_e,
        "counts_per_sec":np.array(new_cts_per_sec),
        }
        if("background" in allcounts[0].keys()):
            new_cts_obj["background"]=allcounts[0]["background"]
        return new_cts_obj
    else:
        print("[!] Error! No return: exception cases for energy bins inconsistency are still under development")
