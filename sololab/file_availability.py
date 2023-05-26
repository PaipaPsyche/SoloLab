import os
import numpy as np
from datetime import datetime
#from values import *

## FILE AVAILABILITY  (CHECK)
def stix_get_all_bkg_files(path,verbose=True):

    filelist = []

    for root, dirs, files in os.walk(path):
        if(not"_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3].split("-")[0]
            date = datetime.strptime(obstime,"%Y%m%dT%H%M%S")
            entry= "["+date.strftime("%Y-%m-%d")+"] "+file
            filelist.append(entry)


    #print all the file names
    filelist.sort()
    print(len(filelist),"BKG files found:")
    for name in filelist:
        print(name)

def stix_get_all_aux_files(path,verbose=True):

    filelist = []

    for root, dirs, files in os.walk(path):
        if("_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file or not "aux" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3].split("-")[0]
            date = datetime.strptime(obstime,"%Y%m%d")
            entry= "["+date.strftime("%Y-%m-%d")+"] "+file
            filelist.append(entry)


    #print all the file names
    filelist.sort()
    print(len(filelist),"AUX files found:")
    for name in filelist:
        print(name)

def stix_suggest_bkg_file_for_date(date,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S",suggestions=1):
    date = datetime.strptime(date,dt_fmt)
    filelist = {}
    file_dt_fmt="%Y%m%dT%H%M%S"

    for root, dirs, files in os.walk(rootpath):
        if(not"_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3].split("-")[0]
            dateobs = datetime.strptime(obstime,file_dt_fmt)

            tdelta = (date-dateobs).days
            filelist[os.path.join(root,file)] = np.abs(tdelta)

    sort_files = sorted(filelist.items(),key=lambda x:x[1])
    sort_files = sort_files[:suggestions]
    for i in range(suggestions):
        print("  [",round(sort_files[i][1]),"days] ",(sort_files[i][0]).replace(rootpath,"PATH + "))
    return sort_files

def stix_suggest_bkg_file_for_file(file,rootpath,suggestions=1):
    obstime = file.split("_")[3].split("-")[0]
    fmtd = "%Y%m%dT%H%M%S"
    return stix_suggest_bkg_file(obstime,rootpath,dt_fmt=fmtd,suggestions=suggestions)



def stix_data_in_interval_exists(date_range,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S"):

    date_range = [datetime.strptime(x,dt_fmt) for x in date_range]
    filelist = {
        "totally":[],
        "partially":[]}

    for root, dirs, files in os.walk(rootpath):
        if("_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file or "aux" in file):
                continue
            #append the file name to the list
            #print(file)
            obstime_1 = file.split("_")[3].split("-")[0]
            obstime_2 = file.split("_")[3].split("-")[1]

            dateinterv = [datetime.strptime(x,"%Y%m%dT%H%M%S") for x in [obstime_1,obstime_2]]

            date1_in_range = date_range[0]>=dateinterv[0] and date_range[0]<=dateinterv[1]
            date2_in_range = date_range[1]>=dateinterv[0] and date_range[1]<=dateinterv[1]

            interv_in_range = date_range[0]<=dateinterv[0] and date_range[1]>=dateinterv[1]

            fileroot=(root+"/"+file).replace(rootpath,"PATH + ")

            if(date1_in_range and date2_in_range):
                filelist["totally"].append(file)
                print(" Totally contained in file:\n  ",
                      fileroot,
                      "\n  File from ",datetime.strftime(dateinterv[0],dt_fmt)," to ",
                      datetime.strftime(dateinterv[1],dt_fmt),"\n")
            elif(date1_in_range or date2_in_range):
                filelist["partially"].append(file)
                print(" Partially contained in file:\n  ",
                      fileroot,
                      "\n  File from ",datetime.strftime(dateinterv[0],dt_fmt)," to ",
                      datetime.strftime(dateinterv[1],dt_fmt),"\n")
            elif(interv_in_range):
                filelist["partially"].append(file)
                print(" Partially contained in file:\n  ",
                      fileroot,
                      "\n  File from ",datetime.strftime(dateinterv[0],dt_fmt)," to ",
                      datetime.strftime(dateinterv[1],dt_fmt),"\n")

    if(len(filelist["totally"])==0 and len(filelist["partially"])==0):
        print("No files containing totally or partially the provided time interval.")

    return filelist


def stix_check_interval_availability(date_range,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S"):
    print("**STIX Science files availability:")
    L1_check = stix_data_in_interval_exists(date_range,rootpath=rootpath,dt_fmt=dt_fmt)
    print("**STIX BKG files availability:")
    BKG_check = stix_suggest_bkg_file_for_date(date_range[0],rootpath=rootpath,dt_fmt=dt_fmt)
    return [L1_check,BKG_check]




def rpw_check_date_availability(date,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S"):
    date = datetime.strptime(date,dt_fmt)
    filelist = []
    simple_dt_fmt="%Y%m%d"

    print("**RPW Science files availability:")
    for root, dirs, files in os.walk(rootpath):
        for file in files:
            if(not ".cdf" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3]

            if(date.strftime(simple_dt_fmt)==obstime):

                print("  Contained in file PATH + ",file)
                filelist.append(file)
    return filelist



def check_combined_availability(date_range,rootpath_stix,rootpath_rpw,dt_fmt="%Y-%m-%d %H:%M:%S"):
    stix_info = stix_check_interval_availability(date_range,rootpath_stix,dt_fmt)
    rpw_info=None
    if(date_range[0].split()[0]==date_range[1].split()[0]):
        rpw_info = rpw_check_date_availability(date_range[0],rootpath_rpw,dt_fmt)
    else:
        print("[!] date range includes two different days, two RPW files might be required")
        rpw_info = [rpw_check_date_availability(date_range[0],rootpath_rpw,dt_fmt),
                   rpw_check_date_availability(date_range[1],rootpath_rpw,dt_fmt)]


    return {"STIX":stix_info,"RPW":rpw_info}


# Manage objects

def save_objects(path,name,counts_spec=None,hfr_psd=None,tnr_psd=None):
    np.savez(path+name+'.npz',counts_spec,hfr_psd,tnr_psd)

def load_objects(path):
    elems = np.load(path,allow_pickle=True)
    ans = [elems[x].all() for x  in elems]
    return ans
