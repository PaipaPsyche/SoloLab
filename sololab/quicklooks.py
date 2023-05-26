
#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
from .values import *
## PLOT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mple
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
import matplotlib.colors as colors

## MATH

import numpy as np


## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table, vstack, hstack
import astropy.units as u




#AUX functions
def scaled_int_ax_formatter(scale=1.,out="function"):

    def int_formatter(x, pos):
        x = x*scale
        if x.is_integer():
            return str(int(x))
        else:
            return str(round(x,1))
    if out == 'function':
        return int_formatter
    elif out =="formatter":
        return FuncFormatter(int_formatter)


def smooth(y,pts):
    ones=np.ones(pts)/pts
    return np.convolve(y,ones,mode="same")
# SOLAR EVENTS CLASS
class solar_event:
    def __init__(self,event_type,times,color=None,linestyle="-",linewidth=2,hl_alpha=0.4,paint_in=None,date_fmt=std_date_fmt):
        self.type = event_type
        #interval,stix_flare,rpw_burst
        try:
            self.start_time = dt.datetime.strptime(times['start'],date_fmt)
        except:
            self.start_time = None
        try:
            self.end_time = dt.datetime.strptime(times['end'],date_fmt)
        except:
            self.end_time = None
        try:
            self.peak_time = dt.datetime.strptime(times['peak'],date_fmt)
        except:
            self.peak_time = None


        #    self.end_time = times['end'] if  times['end'] else None
        #self.peak_time = times['peak'] if  times['peak'] else None
        self.color = color
        self.linestyle=linestyle
        self.linewidth=linewidth
        self.hl_alpha=hl_alpha
        self.paint_in=paint_in
        if(self.paint_in==None):
            if self.type=="rpw_burst":
                self.paint_in = "rpw"
            elif self.type=="stix_flare":
                self.paint_in = "stix"
            else:
                self.paint_in = "both"



    def paint(self):
        if(self.type=="interval"and self.start_time and self.end_time):
            color = self.color if  self.color else "white"
            plt.axvspan(self.start_time, self.end_time, color=color, alpha=self.hl_alpha)
            plt.axvline(self.start_time,c=color,linestyle=self.linestyle,linewidth=self.linewidth)
            plt.axvline(self.end_time,c=color,linestyle=self.linestyle,linewidth=self.linewidth)
        elif(self.type=="stix_flare"):
            color = self.color if  self.color else "white"
            if(self.start_time):
                plt.axvline(self.start_time,c=color,linestyle="--",linewidth=self.linewidth)
            if(self.peak_time):
                plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
        elif(self.type=="rpw_burst" and self.peak_time):
            color = self.color if  self.color else "orange"
            plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
        elif(self.type=="marker" and self.peak_time):
            color = self.color if  self.color else "magenta"
            plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)

class imaging_intervals:
    def __init__(self,intervals,**kwargs):
        self.intervals = intervals
        self.events = []
        self.color=kwargs.get('color','white')
        self.linestyle=kwargs.get("linestyle","-")
        self.linewidth=kwargs.get("linewidth",0.1)
        self.hl_alpha=kwargs.get('hl_alpha',0.3)
        self.date_fmt=kwargs.get('date_fmt',std_date_fmt)
        self.number_color=kwargs.get('number_color','black')
        self.number_pos=kwargs.get('number_pos',7)
        self.fontsize=kwargs.get('fontsize',10)

        self.set_events()

    def set_events(self):
        for i in range(len(self.intervals)):
            times = {'start':self.intervals[i][0],'end':self.intervals[i][1]}
            s_e = solar_event("interval",times,color=self.color,linestyle=self.linestyle,
            linewidth=self.linewidth,hl_alpha=self.hl_alpha,paint_in="stix",date_fmt=self.date_fmt)
            self.events.append(s_e)
    def paint(self):
        for i in range(len(self.events)):
            ev = self.events[i]
            cent_time = ev.start_time + (ev.end_time-ev.start_time)/2
            plt.text(cent_time,self.number_pos,i+1,color=self.number_color,fontsize=self.fontsize,ha="center",fontweight="bold")
            ev.paint()






def rpw_filter_high_values(rpw_psd,threshold=None,hist_bins=20,smoothing_pts=5,replace_with = 1e-16,plot=True):
    print("Filtering RPW high values ...")
    if(plot):
        hh=plt.hist(np.log10(rpw_psd["v"].flatten()),bins=hist_bins)
    else:
        hh=np.histogram(np.log10(rpw_psd["v"].flatten()),bins=hist_bins)


    # bin centers
    temptx = (hh[1][:-1]+hh[1][1:])/2
    #Y value smoothing
    tempty = smooth(hh[0],smoothing_pts)
    if(threshold):
        x_min = np.log10(threshold)
    else:
        idx_min = np.where(tempty==np.min(tempty))[0][0]
        x_min = temptx[idx_min]
    print(f" Inflexion point found at: 10 ** ({round(x_min,1)})")
    print(f" Replacing higher values with 10 ** {np.log10(replace_with)} ...")
    rpw_psd["v"][rpw_psd["v"]>=10**(x_min)]=replace_with
    if(plot):
        plt.plot(temptx,tempty)
        plt.yscale("log")
        plt.axvline(x_min,c="r",ls="--")
    return rpw_psd

def rpw_plot_psd(psd,logscale=True,colorbar=True,cmap="jet",t_format="%H:%M",ax=None,date_range=None,
            axis_fontsize=13,xlabel=True,frequency_range=None):

    t,f,z=psd["time"],psd["frequency"],psd["v"]
    if(logscale):
        z = np.log10(z)


    ax = ax if ax else plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(t_format))

    if(date_range):
        dt_fmt_ = "%d-%b-%Y %H:%M:%S"
        date_range=[datetime.strptime(x,dt_fmt_) for x in date_range]
        ax.set_xlim(*date_range)

    cm= ax.pcolormesh(t,f,z,shading="auto",cmap=cmap)
    if(colorbar):
        plt.colorbar(cm,label="$Log_{10}$ PSD (V)",format=scaled_int_ax_formatter(out='formatter'))



    ax.set_yscale('log')
    ax.set_yticks([], minor=True)

    if(np.min(f)<400):
        ax.set_yticks([x  for x in display_freqs_tnr if np.logical_and(x<=f[-1],x>=f[0])])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:0d}f}}'.format(int(np.maximum(-np.log10(y),1)))).format(y)))
        ax.set_ylabel("RPW - TNR \n Frequency [KHz]",fontsize=axis_fontsize)
    else:
        ax.set_yticks([x  for x in display_freqs if np.logical_and(x<=f[-1],x>=f[0])])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y/1000.),1)))).format(y/1000.)))
        ax.set_ylabel("RPW - HFR \n Frequency [MHz]",fontsize=axis_fontsize)

    if(frequency_range):

        ax.set_ylim(max(frequency_range[0],np.min(f)),min(frequency_range[1],np.max(f)))


    if(xlabel):
        ax.set_xlabel("start time: "+t[0].strftime(std_date_fmt),fontsize=axis_fontsize)


    return ax
# plot rpw curves

def rpw_plot_curves(rpw_psd,savename=None,
                      dt_fmt=std_date_fmt,title=None,ax=None,
                      date_range=None,legend=True,fill_nan=True,lcolor=None,lw=1,ls="-",
                      freqs=None,ylogscale=True,smoothing_pts=None,bias_multiplier=1,return_freq_col=False):

    color_list = ["red","dodgerblue","limegreen","orange","cyan","magenta"]

    t,f,z=rpw_psd["time"],rpw_psd["frequency"],rpw_psd["v"]
    #if(logscale):
    #    z = np.log10(z)

    if(freqs):
        freqs = np.array(freqs)
        freqs = freqs[np.logical_and(freqs>=np.min(f),freqs<=np.max(f))]
        if(len(freqs)==0):
            print(f"Error! None of the selected frequencies to plot are in the frequency range {f[0]} to {f[-1]} kHz. ")
            return ax

    else:
        print("Error! There are no selected frequencies to plot. ")
        return ax



    plot_time=None
    plot_z=None
    if(date_range!=None):
        date_range=[datetime.strptime(x,dt_fmt) for x in date_range]
        d_idx = np.array([True if np.logical_and(x>=date_range[0],x<=date_range[1]) else False for x in t])
        #print(d_idx)

        plot_z = z[:,d_idx]

        #plot_time = plot_time[d_idx]
        plot_time = [i for (i, v) in zip(t, d_idx) if v]
    else:
        plot_time=t
        plot_z=z



    ax = ax if ax else plt.gca()



    myFmt = mdates.DateFormatter(dt_fmt)


    #if(fill_nan):
    #    cts_data=np.nan_to_num(cts_data,nan=0)

    plot_groups = []
    maxval,minval = 0,1

    for sel_freq in freqs:
        idx_close = np.argmin(np.abs(f-sel_freq))
        close_freq = f[idx_close]

        close_intensity = plot_z[idx_close,:]
        maxval =np.max(close_intensity) if np.max(close_intensity)>maxval else maxval
        minval =np.min(close_intensity) if np.min(close_intensity)<minval else minval

        if(smoothing_pts):
            #moving average smoothing
            close_intensity = smooth(close_intensity,smoothing_pts)

        plot_groups.append([close_freq,close_intensity])


    if not lcolor:
        lcolor=color_list[:len(plot_groups)]
    elif len(lcolor)<len(plot_groups):
        print("[!] color list length do not match the number of energy bins plotted. Using default instead")
        lcolor=color_list[:len(plot_groups)]

    lims = [0,0]
    for g in range(len(plot_groups)):
        pg = plot_groups[g]

        if(bias_multiplier):
            plot_y = pg[1]*bias_multiplier**(len(plot_groups)-g)
            ax.plot(plot_time,plot_y,label="{} kHz".format(int(pg[0])),c=lcolor[g],lw=lw,ls=ls)
            ax.plot([plot_time[0],plot_time[-1]],[np.min(plot_y),np.min(plot_y)],c="gray",lw=0.5,ls=":")

            text_height =10*np.sqrt(np.max(plot_y)*np.min(plot_y))
            text_x = plot_time[0]+TimeDelta(120*u.s).to_datetime()
            ax.text(text_x, text_height, "{} kHz".format(int(pg[0])), horizontalalignment='left',verticalalignment='top',color = lcolor[g],fontweight="bold")#, transform=ax.transAxes)
            #ax.plot([plot_time[0],plot_time[-1]],[int(pg[0]),int(pg[0])],c=lcolor[g],lw=0.5,ls=ls)
            if(np.max(plot_y)>lims[1] or lims[1]==0):
                lims[1] = 2*np.max(plot_y)
            if(np.min(plot_y)<lims[0]or lims[0]==0):
                lims[0] = np.min(plot_y)
        else:

            ax.plot(plot_time,pg[1],label="{} kHz".format(int(pg[0])),c=lcolor[g],lw=lw,ls=ls)
            if(np.max(pg[1])>lims[1]or lims[1]==0):
                lims[1] = 2*np.max(pg[1])
            if(np.min(pg[1])<lims[0]or lims[0]==0):
                lims[0] = np.min(pg[1])
            if(legend):
                plt.legend(loc=1,ncol= 4)
    scale = np.log10(maxval/minval)
    if(ylogscale or scale>2):
        plt.yscale("log")
    plt.ylim(lims[0],lims[1])
    if(date_range):
        plt.xlim(plot_time[0],plot_time[-1])

    if(return_freq_col):
        close_freqs = [pg[0] for pg  in plot_groups]
        return ax,zip(close_freqs,lcolor)
    else:
        return ax


def stix_plot_spectrogram(counts_dict,savename=None,colorbar=True,
                      xfmt=" %H:%M",title=None,cmap="jet",fill_nan=True,
                      date_range=None,energy_range=None,x_axis=False,ax=None,
                      logscale=True,ylogscale=False,**kwargs):
    # date_ranges param is used for visualizing delimiters for date range selection of the
    # background and sample pieces (interactive plotting)
    # date_ranges = [[bkg_initial, bkg_final],[smpl_initial, smpl_final]]

    plot_time = counts_dict["time"]
    cts_per_sec = counts_dict["counts_per_sec"]
    min_channels = np.shape(cts_per_sec)[-1]
    energies = counts_dict ["energy_bins"][:min_channels]
    mean_e = counts_dict["mean_energy"][:min_channels]
    ax = ax if ax else plt.gca()


    myFmt = mdates.DateFormatter(xfmt)


    ax.xaxis.set_major_formatter(myFmt)

    cts_data = np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec>0)).T if logscale else cts_per_sec.T
    if(fill_nan):
        cts_data=np.nan_to_num(cts_data,nan=0)
    cm= ax.pcolormesh(plot_time,mean_e,cts_data,shading="auto",cmap=cmap,vmin=0)
    if(colorbar):
        cblabel = "$Log_{10}$ Counts $s^{-1}$" if logscale else "Counts $s^{-1}$"
        plt.colorbar(cm,label=cblabel,format = scaled_int_ax_formatter(out='formatter'))
    if(x_axis):
        ax.set_xlabel("Observation time [@ SolO]: "+plot_time[0].strftime(std_date_fmt),fontsize=14)
        #plt.xlabel("start time "+date_ranges[0],fontsize=14)
    ax.set_ylabel('STIX \n Energy bins [KeV]',fontsize=14)
    if(energy_range):
        ax.set_ylim(*energy_range)
    if(title):
        plt.title(title)
    if(date_range):

        date_range=[datetime.strptime(x,std_date_fmt) for x in date_range]
        ax.set_xlim(*date_range)
    if(ylogscale):
        ax.set_yscale('log')
    #return fig, axes
    if(savename):
        plt.savefig(savename,bbox_inch="tight")


    return ax


def stix_plot_bkg(l1_counts,ax=None):

    ax= ax if ax else plt.gca()
    if not"background" in l1_counts.keys():
        print("This L1 counts object has no removed background.")
        return ax


    energies = l1_counts["mean_energy"]
    bkg_counts = l1_counts["background"]

    min_channels = min(len(energies),len(bkg_counts))

    ax.plot(energies[:min_channels],bkg_counts[:min_channels]/energies[:min_channels])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Energy[kev]")
    ax.set_ylabel("Counts / sec / keV")

    if(energies[min_channels-1]>=31):
        ax.axvline(31,c="r",ls="--")
    if(energies[min_channels-1]>=81):
        ax.axvline(81,c="r",ls="--",label="Callibration lines \n  31 and 81 kev")
    ax.grid()
    if(energies[-1]>31):
        ax.legend(fontsize=12)


def stix_plot_counts(counts_dict,savename=None,
                      dt_fmt=std_date_fmt,title=None,e_range=None,ax=None,
                      date_range=None,legend=True,fill_nan=True,lcolor=None,lw=1,ls="-",smoothing_pts=1,
                      integrate_bins=None,zlogscale=True,ylogscale=True,verbose=True):

    color_list = ["red","dodgerblue","limegreen","cyan","magenta"]




    #get data
    plot_time = counts_dict["time"]
    cts_per_sec = counts_dict["counts_per_sec"]
    min_channels = np.shape(cts_per_sec)[-1]
    energies = counts_dict ["energy_bins"][:min_channels]
    mean_e = np.array(counts_dict["mean_energy"][:min_channels])

    cts_data = cts_per_sec #np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec>0)) if zlogscale else cts_per_sec

    myFmt = mdates.DateFormatter(dt_fmt)

    if(fill_nan):
        cts_data=np.nan_to_num(cts_data,nan=0)


    # select data
    if(e_range!=None):
        e_idx = np.logical_and(mean_e>=e_range[0],mean_e<=e_range[1])
        #print(e_idx)

        cts_data = cts_data[:,e_idx]
        energies = energies[e_idx]

        mean_e = mean_e[e_idx]
    if(date_range!=None):
        date_range=[datetime.strptime(x,dt_fmt) for x in date_range]
        d_idx = np.array([True if np.logical_and(x>=date_range[0],x<=date_range[1]) else False for x in plot_time])
        #print(d_idx)
        cts_data = cts_data[d_idx,:]

        #plot_time = plot_time[d_idx]
        plot_time = [i for (i, v) in zip(plot_time, d_idx) if v]



    plot_groups = []

    if(integrate_bins!=None):
        for e_bin in integrate_bins:

            e_idx = np.logical_and(mean_e>=e_bin[0],mean_e<=e_bin[1])

            cts_per_sec_g = cts_data[:,e_idx]
            energies_g = energies[e_idx]
            mean_e_g= mean_e[e_idx]

            energy_g = [energies_g[0]["e_low"],energies_g[-1]["e_high"]]
            m_energy_g = np.mean([energies_g[0]["e_low"],energies_g[0]["e_high"]])
            cts_sec_g = np.sum(cts_per_sec_g,axis=1)


            plot_groups.append([cts_sec_g,energy_g,m_energy_g])
    else:
        for e in range(len(energies)):
            plot_groups.append([cts_per_sec[:,e],[energies[e]["e_low"],energies[e]["e_high"]],mean_e[e]])

    if not lcolor:
        lcolor=color_list[:len(plot_groups)]
    elif len(lcolor)<len(plot_groups):
        print("[!] color list length do not match the number of energy bins plotted. Using default instead")
        lcolor=color_list[:len(plot_groups)]

    ax = ax if ax else plt.gca()
    #ax.xaxis.set_major_formatter(myFmt)

    lims = [2,10]


    if(verbose):
        print("STIX energy bins info:")
    for g in range(len(plot_groups)):
        pg = plot_groups[g]
        counts_plot=pg[0]

        lbl="{} - {} keV".format(int(pg[1][0]),int(pg[1][1]))
        if(verbose):
            maxcts = round(np.max(counts_plot))
            maxtime = dt.datetime.strftime(plot_time[np.argmax(counts_plot)],std_date_fmt)
            print(lbl,"max. cts/sec = {}  Time: {}".format(maxcts,maxtime))

        smooth_counts = smooth(counts_plot,smoothing_pts)




        ax.plot(plot_time,smooth_counts,label=lbl,c=lcolor[g],lw=lw,ls=ls)
        if(np.max(pg[0]+1)>lims[1]):
            lims[1] = 1.1*np.max(pg[0]+1)
    if(legend):
        plt.legend(ncol= 4,loc=2)
    if(ylogscale):
        #plt.ylim(0.5,np.max(cts_per_sec))
        plt.yscale("log")
    #plt.ylim(lims[0],lims[1])
    plt.ylim(bottom=3)
    if(date_range):
        plt.xlim(plot_time[0],plot_time[-1])

    return ax



# Combined views
def paint_markers(ax,markers,dt_fmt=std_date_fmt):
    for mk in markers:
        ax.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=1.5)
def ax_paint_grid(ax,which='major',axis="both",color="lightgray",linestyle=":"):
    ax.grid(b=True, which=which, axis=axis, color=color, linestyle=linestyle)


def stix_rpw_combinedQuickLook(l1_cts,rpw_psd,tnr_psd=None,energy_range=[4,28],frequency_range=None,energy_bins=[[4,12],[16,28]],
                               frequencies=[500,3500,13000],stix_cmap="bone",rpw_cmap="bone",cmap=None,date_fmt="%H:%M",
                               date_range=None,stix_ylogscale=False,smoothing_points=5,stix_lcolor=None,ylogscale_stix_curves=True,
                               stix_zlogscale=True,
                               rpw_lcolor=None,rpw_units="watts",rpw_repeated_freqs=None,rpw_join_xaxis=True,
                               figsize=(15,7),mode="overlay",curve_overlay="both",rpw_plot_bias=False,curve_lw=2,
                               timegrid=True,fontsize=13,markers={},imaging_intervals=None,savename=None):
    #font = { #'family' : 'normal',
        #'weight' : 'normal',
    #    'size'   : 15}


    hfr_freq_range = [rpw_psd["frequency"][0],rpw_psd["frequency"][-1]]
    tnr_freq_range = None

    hfr_psd_cp=rpw_psd.copy()
    tnr_psd_cp = None




    if(not tnr_psd):
        tnr_psd_cp = None
    else:
        tnr_psd_cp = tnr_psd.copy()
        tnr_freq_range = [tnr_psd["frequency"][0],tnr_psd["frequency"][-1]]


    if(frequency_range):
        hfr_freq_range=[max(hfr_freq_range[0],frequency_range[0]) , min(hfr_freq_range[-1],frequency_range[-1])]
        if(tnr_psd):
            tnr_freq_range=[max(tnr_freq_range[0],frequency_range[0]) , min(tnr_freq_range[-1],frequency_range[-1])]

    # depending on rpw_repeated_freqs value, crop the spectrograms so they dont overlapping
    if(tnr_psd and rpw_repeated_freqs):

        #crop hfr spectrogram: if the upper bound of tnr spectrogram freq is
        # bigger than the lower bound of hfr, crop hfr lower bound until it matches tnr upper bound
        if(rpw_repeated_freqs=="only_tnr"):
            hfr_freq_range[0] = max(hfr_freq_range[0],tnr_freq_range[1])

        #crop tnr spectrogram: if the lower bound of hfr spectrogram freq is
        # smaller than the higher bound of tnr, crop tnr upper bound until it matches hfr lower bound
        elif(rpw_repeated_freqs=="only_hfr"):
            tnr_freq_range[1] = min(hfr_freq_range[0],tnr_freq_range[1])





    rpw_ylabel = "I "
    if rpw_units=="watts":
        rpw_ylabel+="[W m$^{-2}$ Hz$^{-1}$]"
    elif rpw_units=="sfu":
        hfr_psd_cp["v"]= rpw_psd_cp["v"]/1e-22
        rpw_ylabel += " [SFU]"



    plt.rcParams.update({'font.size': fontsize})

    # palette for spectrograms
    stix_cmap = cmap if cmap else stix_cmap
    rpw_cmap = cmap if cmap else rpw_cmap


     #set date range([str,str]): if not provided, then, use max possible
    dt_mins = [ np.min(l1_cts["time"]),np.min(hfr_psd_cp["time"]) ]
    dt_maxs = [np.max(l1_cts["time"]),np.max(hfr_psd_cp["time"])]
    if date_range:
        dt_mins.append(datetime.strptime(date_range[0],std_date_fmt))
        dt_maxs.append(datetime.strptime(date_range[-1],std_date_fmt))

    d_range = [np.max(dt_mins),np.min(dt_maxs)]
    d_range_str = [datetime.strftime(x,std_date_fmt) for x in d_range]

    myFmt = mdates.DateFormatter(date_fmt)



    fig=plt.figure(figsize=figsize,dpi=250)
    if(not tnr_psd):



        if(mode in ["overlay","spectrograms"]):
            #CHANGe

            ax1 = fig.add_subplot(211)
            rpw_plot_psd(hfr_psd_cp,xlabel=False,frequency_range=hfr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax1)
            if(np.min(hfr_psd_cp["frequency"])<400):
                ax1.yaxis.set_major_formatter(scaled_int_ax_formatter(out="formatter"))
            else:
                ax1.yaxis.set_major_formatter(scaled_int_ax_formatter(scale=1/1000.,out="formatter"))

            if(curve_overlay in ["both","rpw"] and mode=="overlay"):

                ax1b = ax1.twinx()

                #CHANGE
                multip=round(5*10**np.interp(len(frequencies),[2,7],[3.5,2]),-2) if rpw_plot_bias else None
                _,fqcol = rpw_plot_curves(hfr_psd_cp,freqs=frequencies,ax=ax1b,date_range=date_range,lw=curve_lw,smoothing_pts=smoothing_points,
                                bias_multiplier=multip,return_freq_col=True)
                for c_freq,c_col in fqcol:
                    ax1b.axhline(c_freq,color=c_col,lw=0.5)
                if(multip != None):
                    ax1b.get_yaxis().set_ticks([])
                    ax1b.set_yticklabels([])
            paint_markers(ax1,markers)
            ax1.invert_yaxis()


            ax2 = fig.add_subplot(212)
            stix_plot_spectrogram(l1_cts,ax=ax2,cmap=stix_cmap,energy_range=energy_range,date_range=d_range_str,x_axis=True,logscale=stix_zlogscale)
            if(stix_ylogscale):
                plt.yscale('log')
            paint_markers(ax2,markers)
            if(imaging_intervals):
                imaging_intervals.paint()
            if(curve_overlay in ["both","stix"] and mode=="overlay"):
                ax3=ax2.twinx()
                stix_plot_counts(l1_cts,smoothing_pts=smoothing_points,integrate_bins=energy_bins,
                                     lcolor=stix_lcolor,ax=ax3,lw=curve_lw,ylogscale=ylogscale_stix_curves)
            plt.subplots_adjust(hspace=.02)
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position('top')
            if(timegrid):
                for axx  in [ax1,ax2]:
                    ax_paint_grid(axx,which="major",axis="x")
        elif(mode=="curves"):
            stix_rows=max(1,int(len(frequencies)/2)-1)
            rows = stix_rows+len(frequencies)
            ax0 = plt.subplot2grid((rows, 1), (rows-stix_rows, 0), rowspan=stix_rows)
            stix_plot_counts(l1_cts,integrate_bins=energy_bins,
                                     lcolor=stix_lcolor,ax=ax0,lw=curve_lw,date_range=d_range_str,ylogscale=ylogscale_stix_curves)
            ax0.xaxis.set_major_formatter(myFmt)
            ax0.grid()
            ax0.set_xlim(d_range)
            paint_markers(ax0,markers)
            freq_axs=[]
            for i in range(len(frequencies)):
                freq_axs.append(plt.subplot2grid((rows, 1), (i, 0)))

                rpw_plot_curves(hfr_psd_cp,freqs=[frequencies[i]],ax=freq_axs[-1],lw=curve_lw,smoothing_pts=smoothing_points,lcolor=["k"],
                                bias_multiplier=None)
                paint_markers(freq_axs[-1],markers)
                if(i==0):
                    freq_axs[-1].xaxis.tick_top()

                    freq_axs[-1].xaxis.set_major_formatter(myFmt)
                    freq_axs[-1].set_ylabel(rpw_ylabel)

                else:
                    freq_axs[-1].set_xticklabels([])

                freq_axs[-1].set_xlim(d_range)
                freq_axs[-1].grid()



                #plt.tight_layout()
                #plt.savefig('grid_figure.pdf')
        #plt.xlim()
        return fig




    else:

        if(mode in ["overlay","spectrograms"]):



            ax1 = fig.add_subplot(311)
            rpw_plot_psd(tnr_psd_cp,xlabel=False,frequency_range=tnr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax1)
            ax1.yaxis.set_major_formatter(scaled_int_ax_formatter(out="formatter"))
            if(curve_overlay in ["both","rpw"] and mode=="overlay"):

                ax1b = ax1.twinx()

                #CHANGE
                multip=round(5*10**np.interp(len(frequencies),[2,7],[3.5,2]),-2) if rpw_plot_bias else None
                _,fqcol=rpw_plot_curves(tnr_psd_cp,freqs=frequencies,ax=ax1b,date_range=date_range,lw=curve_lw,smoothing_pts=smoothing_points,
                                bias_multiplier=multip,return_freq_col=True)
                for c_freq,c_col in fqcol:
                    ax1b.axhline(c_freq,color=c_col,lw=0.5)
                if(multip != None):
                    ax1b.get_yaxis().set_ticks([])
                    ax1b.set_yticklabels([])
            paint_markers(ax1,markers)
            ax1.invert_yaxis()



            ax2 = fig.add_subplot(312,sharex = ax1)
            rpw_plot_psd(hfr_psd_cp,xlabel=False,frequency_range=hfr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax2)
            ax2.yaxis.set_major_formatter(scaled_int_ax_formatter(scale=1/1000,out="formatter"))

            if(curve_overlay in ["both","rpw"] and mode=="overlay"):

                ax2b = ax2.twinx()


                #CHANGE
                multip=round(5*10**np.interp(len(frequencies),[2,7],[3.5,2]),-2) if rpw_plot_bias else None
                _,fqcol=rpw_plot_curves(hfr_psd_cp,freqs=frequencies,ax=ax2b,date_range=date_range,lw=curve_lw,smoothing_pts=smoothing_points,
                                bias_multiplier=multip,return_freq_col=True)

                for c_freq,c_col in fqcol:
                    ax2b.axhline(c_freq,color=c_col,lw=0.5)
                if(multip != None):
                    ax2b.get_yaxis().set_ticks([])
                    ax2b.set_yticklabels([])
            if(rpw_join_xaxis):
                ax1.sharex(ax2)
                ax1.get_shared_x_axes().join(ax1, ax2)
                ax1.xaxis.tick_top()
                ax1.xaxis.set_label_position('top')
            paint_markers(ax2,markers)
            ax2.invert_yaxis()
            plt.subplots_adjust(hspace=.02)
            ax2.set_xticklabels([])


            ax3 = fig.add_subplot(313)
            ax3.margins(x = 0, y =-0.4)
            stix_plot_spectrogram(l1_cts,ax=ax3,cmap=stix_cmap,energy_range=energy_range,date_range=d_range_str,x_axis=True,logscale=stix_zlogscale)

            if(stix_ylogscale):
                plt.yscale('log')
            if(imaging_intervals):
                imaging_intervals.paint()
            paint_markers(ax3,markers)
            if(curve_overlay in ["both","stix"] and mode=="overlay"):
                ax3b=ax3.twinx()
                stix_plot_counts(l1_cts,smoothing_pts=smoothing_points,integrate_bins=energy_bins,
                                     lcolor=stix_lcolor,ax=ax3b,lw=curve_lw,ylogscale=ylogscale_stix_curves)
            if(timegrid):
                for axx  in [ax1,ax2,ax3]:
                    ax_paint_grid(axx,which="major",axis="x")
        elif(mode=="curves"):
            stix_rows=max(1,int(len(frequencies)/2)-1)
            rows = stix_rows+len(frequencies)
            ax0 = plt.subplot2grid((rows, 1), (rows-stix_rows, 0), rowspan=stix_rows)
            stix_plot_counts(l1_cts,integrate_bins=energy_bins,
                                     lcolor=stix_lcolor,ax=ax0,lw=curve_lw,date_range=d_range_str,ylogscale=ylogscale_stix_curves)
            ax0.xaxis.set_major_formatter(myFmt)
            ax0.grid()
            ax0.set_xlim(d_range)
            paint_markers(ax0,markers)
            freq_axs=[]
            for i in range(len(frequencies)):
                freq_axs.append(plt.subplot2grid((rows, 1), (i, 0)))

                if(frequencies[i]<hfr_psd_cp["frequency"][0] and tnr_psd_cp):
                    rpw_plot_curves(tnr_psd_cp,freqs=[frequencies[i]],ax=freq_axs[-1],lw=curve_lw,
                                    smoothing_pts=smoothing_points,lcolor=["k"],
                                    bias_multiplier=None)
                else:
                    rpw_plot_curves(hfr_psd_cp,freqs=[frequencies[i]],ax=freq_axs[-1],lw=curve_lw,
                                    smoothing_pts=smoothing_points,lcolor=["k"],
                                    bias_multiplier=None)
                paint_markers(freq_axs[-1],markers)
                if(i==0):
                    freq_axs[-1].xaxis.tick_top()

                    freq_axs[-1].xaxis.set_major_formatter(myFmt)
                    freq_axs[-1].set_ylabel(rpw_ylabel)

                else:
                    freq_axs[-1].set_xticklabels([])

                freq_axs[-1].set_xlim(d_range)
                freq_axs[-1].grid()



                #plt.tight_layout()
                #plt.savefig('grid_figure.pdf')
        #plt.xlim()
        return fig


def plot_stix_rpw_epd(l1_cts,rpw_psd,ept_df,energies_ept,tnr_psd=None,energy_range=[4,28],frequency_range=None,energy_bins=[[4,12],[16,28]],
                               frequencies=[500,3500,13000],stix_cmap="bone",rpw_cmap="bone",cmap=None,date_fmt="%H:%M",
                               date_range=None,stix_ylogscale=False,smoothing_points=5,stix_lcolor=None,ylogscale_stix_curves=True,
                               ept_resample="5min",ept_channels=[0,2,4,8,12],ept_particle='Electron',round_epd_label=True,
                               rpw_lcolor=None,rpw_units="watts",rpw_repeated_freqs=None,rpw_join_xaxis=True,
                               figsize=(15,7),mode="overlay",curve_overlay="both",rpw_plot_bias=False,curve_lw=2,
                               timegrid=True,fontsize=13,markers={},imaging_intervals=None,savename=None):
    #font = { #'family' : 'normal',
        #'weight' : 'normal',
    #    'size'   : 15}


    hfr_freq_range = [rpw_psd["frequency"][0],rpw_psd["frequency"][-1]]
    tnr_freq_range = None

    hfr_psd_cp=rpw_psd.copy()
    tnr_psd_cp = None




    if(not tnr_psd):
        tnr_psd_cp = None
    else:
        tnr_psd_cp = tnr_psd.copy()
        tnr_freq_range = [tnr_psd["frequency"][0],tnr_psd["frequency"][-1]]


    if(frequency_range):
        hfr_freq_range=[max(hfr_freq_range[0],frequency_range[0]) , min(hfr_freq_range[-1],frequency_range[-1])]
        if(tnr_psd):
            tnr_freq_range=[max(tnr_freq_range[0],frequency_range[0]) , min(tnr_freq_range[-1],frequency_range[-1])]

    # depending on rpw_repeated_freqs value, crop the spectrograms so they dont overlapping
    if(tnr_psd and rpw_repeated_freqs):

        #crop hfr spectrogram: if the upper bound of tnr spectrogram freq is
        # bigger than the lower bound of hfr, crop hfr lower bound until it matches tnr upper bound
        if(rpw_repeated_freqs=="only_tnr"):
            hfr_freq_range[0] = max(hfr_freq_range[0],tnr_freq_range[1])

        #crop tnr spectrogram: if the lower bound of hfr spectrogram freq is
        # smaller than the higher bound of tnr, crop tnr upper bound until it matches hfr lower bound
        elif(rpw_repeated_freqs=="only_hfr"):
            tnr_freq_range[1] = min(hfr_freq_range[0],tnr_freq_range[1])






    rpw_ylabel = "I "
    if rpw_units=="watts":
        rpw_ylabel+="[W m$^{-2}$ Hz$^{-1}$]"
    elif rpw_units=="sfu":
        hfr_psd_cp["v"]= rpw_psd_cp["v"]/1e-22
        rpw_ylabel += " [SFU]"



    plt.rcParams.update({'font.size': fontsize})

    # palette for spectrograms
    stix_cmap = cmap if cmap else stix_cmap
    rpw_cmap = cmap if cmap else rpw_cmap


     #set date range([str,str]): if not provided, then, use max possible
    dt_mins = [ np.min(l1_cts["time"]),np.min(hfr_psd_cp["time"]) ]
    dt_maxs = [np.max(l1_cts["time"]),np.max(hfr_psd_cp["time"])]
    if date_range:
        dt_mins.append(datetime.strptime(date_range[0],std_date_fmt))
        dt_maxs.append(datetime.strptime(date_range[-1],std_date_fmt))

    d_range = [np.max(dt_mins),np.min(dt_maxs)]
    d_range_str = [datetime.strftime(x,std_date_fmt) for x in d_range]

    myFmt = mdates.DateFormatter(date_fmt)

    if(not tnr_psd):
        fig, (ax1, ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=figsize,
            constrained_layout=True,dpi=250)


        # AX 1
        plot_ept_data(ept_df,energies_ept,ax=ax1,particle=ept_particle,channels=ept_channels,
                      resample=ept_resample,date_range=d_range_str,round_epd_label=round_epd_label)


        #AX2
        rpw_plot_psd(hfr_psd_cp,xlabel=False,frequency_range=hfr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax2)
        if(np.min(hfr_psd_cp["frequency"])<400):
            ax2.yaxis.set_major_formatter(scaled_int_ax_formatter(out="formatter"))
        else:
            ax2.yaxis.set_major_formatter(scaled_int_ax_formatter(scale=1/1000.,out="formatter"))


        paint_markers(ax2,markers)
        ax2.invert_yaxis()



        #AX3
        stix_plot_spectrogram(l1_cts,ax=ax3,cmap=stix_cmap,energy_range=energy_range,date_range=d_range_str,x_axis=True)
        if(stix_ylogscale):
            plt.yscale('log')
        paint_markers(ax3,markers)
        if(imaging_intervals):
            imaging_intervals.paint()

        ax3a=ax3.twinx()
        stix_plot_counts(l1_cts,smoothing_pts=smoothing_points,integrate_bins=energy_bins,
                             lcolor=stix_lcolor,ax=ax3a,lw=curve_lw,ylogscale=ylogscale_stix_curves)

        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        if(timegrid):
            for axx  in [ax1,ax2,ax3]:
                ax_paint_grid(axx,which="major",axis="x")

    else:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,figsize=figsize,
            constrained_layout=True,dpi=250 ,gridspec_kw ={'hspace':.0},)


        # AX1
        plot_ept_data(ept_df,energies_ept,ax=ax1,particle=ept_particle,channels=ept_channels,
                      resample=ept_resample,date_range=d_range_str,round_epd_label=round_epd_label)


        #AX2
        rpw_plot_psd(tnr_psd_cp,xlabel=False,frequency_range=tnr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax2)
        ax2.yaxis.set_major_formatter(scaled_int_ax_formatter(out="formatter"))
        paint_markers(ax2,markers)
        ax2.invert_yaxis()

        #AX3
        rpw_plot_psd(hfr_psd_cp,xlabel=False,frequency_range=hfr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax3)

        paint_markers(ax3,markers)
        ax3.invert_yaxis()



        #AX4
        stix_plot_spectrogram(l1_cts,ax=ax4,cmap=stix_cmap,energy_range=energy_range,date_range=d_range_str,x_axis=True)
        if(stix_ylogscale):
            plt.yscale('log')
        paint_markers(ax4,markers)
        if(imaging_intervals):
            imaging_intervals.paint()

        ax4a=ax4.twinx()
        stix_plot_counts(l1_cts,smoothing_pts=smoothing_points,integrate_bins=energy_bins,
                             lcolor=stix_lcolor,ax=ax4a,lw=curve_lw ,ylogscale=ylogscale_stix_curves)


        #plt.subplots_adjust(hspace=.02)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        if(timegrid):
            for axx  in [ax1,ax2,ax3,ax4]:
                ax_paint_grid(axx,which="major",axis="x")


        return fig


def plot_ept_data(ept_df,energies_ept,ax,particle='Electron',channels=[0,4,8,16,22,26],
                  resample='20min',date_range=None,round_epd_label=True):

    ax = ax if ax else plt.gca()
    ax.set_prop_cycle('color', plt.cm.jet_r(np.linspace(0,1,7)))

    if(date_range):
        fromstd = lambda x : dt.datetime.strptime(x,std_date_fmt)

        start_date,end_date = date_range[0],date_range[1]
        start_date,end_date = fromstd(start_date),fromstd(end_date)



        getmask =lambda x: (x.index > start_date) & (x.index <= end_date)


        mask = getmask(ept_df)
        ept_df=ept_df.loc[mask]



    for channel in channels:
        leg_elems=energies_ept[f"{particle}_Bins_Text"][channel][0].split()
        to_round = 0 if round_epd_label else 2
        ktype = int if round_epd_label else float
        new_leg = f'{ktype(round(float(leg_elems[0])*1000,to_round))} - {ktype(round(float(leg_elems[2])*1000,to_round))} keV'



        y =  ept_df[f'{particle}_Flux'][f'{particle}_Flux_{channel}']
        x = ept_df.index
        #print(len(x),len(y),"LEN")
        #print(x,y)
        ax.plot(x,smooth(y=y,pts=600),label = new_leg)
    ax.set_yscale('log')
    ax.set_ylabel("Electron flux\n"+r"(cm$^2$ sr s MeV)$^{-1}$")
    ax.grid()
    ax.set_ylim(bottom=100)
    #ax.legend(ncols=1,loc='center left', bbox_to_anchor=(1, 0.5))
    cols = int(len(channels)/2) if len(channels)>4 else len(channels)
    ax.legend(ncols=cols,loc='upper center', bbox_to_anchor=(0.5,1.2))
    return ax


def plot_rpw_stix_spec(l1_cts,hfr_psd,specdata,tnr_psd=None,energy_range=[4,28],frequency_range=None,energy_bins=[[4,10],[10,16],[16,22]],
                               frequencies=[500,3500,13000],stix_cmap="bone",stix_spec_cmap="binary_r",rpw_cmap="bone",cmap='inferno',date_fmt="%H:%M",
                               date_range=None,stix_ylogscale=False,smoothing_points=5,stix_lcolor=None,NTplot_cuts=[],
                               rpw_lcolor=None,rpw_units="watts",rpw_repeated_freqs=None,rpw_join_xaxis=True,
                               figsize=(14,12),mode="overlay",curve_overlay="both",rpw_plot_bias=False,curve_lw=1.5,
                               timegrid=True,fontsize=13,markers={},imaging_intervals=None,savename=None):



    #retrieve energy thresholds used
    energy_thresholds=[int(x.split("_")[-1].split('kev')[0]) for x in specdata.keys() if 'integrated_above_' in x]


    hfr_freq_range = [hfr_psd["frequency"][0],hfr_psd["frequency"][-1]]
    tnr_freq_range = None

    hfr_psd_cp=hfr_psd.copy()
    tnr_psd_cp = None




    if(not tnr_psd):
        tnr_psd_cp = None
    else:
        tnr_psd_cp = tnr_psd.copy()
        tnr_freq_range = [tnr_psd["frequency"][0],tnr_psd["frequency"][-1]]


    if(frequency_range):
        hfr_freq_range=[max(hfr_freq_range[0],frequency_range[0]) , min(hfr_freq_range[-1],frequency_range[-1])]
        if(tnr_psd):
            tnr_freq_range=[max(tnr_freq_range[0],frequency_range[0]) , min(tnr_freq_range[-1],frequency_range[-1])]

    # depending on rpw_repeated_freqs value, crop the spectrograms so they dont overlapping
    if(tnr_psd and rpw_repeated_freqs):

        #crop hfr spectrogram: if the upper bound of tnr spectrogram freq is
        # bigger than the lower bound of hfr, crop hfr lower bound until it matches tnr upper bound
        if(rpw_repeated_freqs=="only_tnr"):
            hfr_freq_range[0] = max(hfr_freq_range[0],tnr_freq_range[1])

        #crop tnr spectrogram: if the lower bound of hfr spectrogram freq is
        # smaller than the higher bound of tnr, crop tnr upper bound until it matches hfr lower bound
        elif(rpw_repeated_freqs=="only_hfr"):
            tnr_freq_range[1] = min(hfr_freq_range[0],tnr_freq_range[1])






    rpw_ylabel = "I "
    if rpw_units=="watts":
        rpw_ylabel+="[W m$^{-2}$ Hz$^{-1}$]"
    elif rpw_units=="sfu":
        hfr_psd_cp["v"]= rpw_psd_cp["v"]/1e-22
        rpw_ylabel += " [SFU]"



    plt.rcParams.update({'font.size': fontsize})

    # palette for spectrograms
    stix_cmap = cmap if cmap else stix_cmap
    rpw_cmap = cmap if cmap else rpw_cmap


     #set date range([str,str]): if not provided, then, use max possible
    dt_mins = [ np.min(l1_cts["time"]),np.min(hfr_psd_cp["time"]) ]
    dt_maxs = [np.max(l1_cts["time"]),np.max(hfr_psd_cp["time"])]
    if date_range:
        dt_mins.append(datetime.strptime(date_range[0],std_date_fmt))
        dt_maxs.append(datetime.strptime(date_range[-1],std_date_fmt))

    d_range = [np.max(dt_mins),np.min(dt_maxs)]
    d_range_str = [datetime.strftime(x,std_date_fmt) for x in d_range]

    myFmt = mdates.DateFormatter(date_fmt)

    if(not tnr_psd):
        fig, (ax1, ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=figsize,
            constrained_layout=True,dpi=250)


        # AX 1

        rpw_plot_psd(hfr_psd_cp,xlabel=False,frequency_range=hfr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax1)
        if(np.min(hfr_psd_cp["frequency"])<400):
            ax1.yaxis.set_major_formatter(scaled_int_ax_formatter(out="formatter"))
        else:
            ax1.yaxis.set_major_formatter(scaled_int_ax_formatter(scale=1/1000.,out="formatter"))


        paint_markers(ax1,markers)
        ax1.invert_yaxis()

        ax1b = ax1.twinx()
        stix_plot_counts(l1_cts,ax=ax1b,integrate_bins=energy_bins,smoothing_pts=smoothing_points,lw=curve_lw)


        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')




        #AX2
        stix_plot_spectrogram(l1_cts,ax=ax2,cmap=stix_cmap,energy_range=energy_range,date_range=d_range_str,x_axis=True)
        if(stix_ylogscale):
            plt.yscale('log')
        paint_markers(ax2,markers)
        if(imaging_intervals):
            imaging_intervals.paint()

        ax2a=ax2.twinx()

        ax2a.scatter(specdata['start_date'],-specdata['DeltaLow'],s = 8,c="white",label='Electron Spectral Index')

        ax2a.grid()
        ax2a.legend()


        #AX3

        stix_plot_spectrogram(l1_cts,energy_range=energy_range,ylogscale=True,cmap=stix_spec_cmap,ax=ax3,date_range=date_range)

        ax3.set_xlim(d_range)
        ax3a = ax3.twinx()
        ax3a.set_xlim(d_range)


        #normalize item number values to colormap
        norm = colors.Normalize(vmin=-0.6, vmax=len(energy_thresholds)+0.7)
        #cuts = ["09-May-2021 14:10:00"]
        dt_cuts = [dt.datetime.strptime(cut,std_date_fmt) for cut in NTplot_cuts]
        dt_cuts = [list(specdata['start_date'])[0].to_pydatetime()-dt.timedelta(seconds=30)] + dt_cuts + [list(specdata['start_date'])[-1].to_pydatetime()+dt.timedelta(seconds=30)]
        for i in range(len(energy_thresholds)):
            en = energy_thresholds[i]

            #colormap possible values = viridis, jet, spectral
            rgba_color = cm.nipy_spectral_r(norm(i),bytes=True)[:3]
            rgba_color= [x/255. for x in rgba_color]
            #/255
            for c in range(len(NTplot_cuts)+1):
                ixx_cut = np.logical_and(specdata['start_date']>=dt_cuts[c],specdata['start_date']<dt_cuts[c+1])
                lbl = f"Above {en} keV" if c==0 else None

                ax3a.plot(specdata['start_date'][ixx_cut],specdata[f'integrated_above_{en}kev'][ixx_cut],label=lbl,color=rgba_color,ls="-",lw=1.5)
                ax2a.plot(specdata['start_date'][ixx_cut],-specdata['DeltaLow'][ixx_cut],c="white",lw=curve_lw,ls="--")


        ax3a.grid()
        ax3a.set_yscale("log")
        ax3a.legend()

        if(timegrid):
            for axx  in [ax1,ax2,ax3]:
                ax_paint_grid(axx,which="major",axis="x")

    else:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,figsize=figsize,
            constrained_layout=True,dpi=250 ,gridspec_kw ={'hspace':.0},)


        # AX1

        rpw_plot_psd(tnr_psd_cp,xlabel=False,frequency_range=tnr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax1)
        ax1.yaxis.set_major_formatter(scaled_int_ax_formatter(out="formatter"))
        paint_markers(ax1,markers)
        ax1.invert_yaxis()

        #AX2
        rpw_plot_psd(hfr_psd_cp,xlabel=False,frequency_range=hfr_freq_range,date_range=d_range_str,
                                 cmap=rpw_cmap,t_format="%H:%M",ax=ax2)

        paint_markers(ax2,markers)
        ax2.invert_yaxis()

        ax2b = ax2.twinx()
        stix_plot_counts(l1_cts,ax=ax2b,integrate_bins=energy_bins,smoothing_pts=smoothing_points,lw=curve_lw)



        #AX3
        stix_plot_spectrogram(l1_cts,ax=ax3,cmap=stix_cmap,energy_range=energy_range,date_range=d_range_str,x_axis=True)
        if(stix_ylogscale):
            plt.yscale('log')
        paint_markers(ax3,markers)
        if(imaging_intervals):
            imaging_intervals.paint()

        ax3a=ax3.twinx()

        ax3a.scatter(specdata['start_date'],-specdata['DeltaLow'],s = 8,c="white",label='Electron Spectral Index')

        ax3a.grid()
        ax3a.legend()


        #AX4

        stix_plot_spectrogram(l1_cts,energy_range=energy_range,ylogscale=True,cmap=stix_spec_cmap,ax=ax4,date_range=date_range)

        ax4.set_xlim(d_range)
        ax4a = ax4.twinx()
        ax4a.set_xlim(d_range)


        #normalize item number values to colormap
        norm = colors.Normalize(vmin=-0.6, vmax=len(energy_thresholds)+0.7)
        dt_cuts = [dt.datetime.strptime(cut,std_date_fmt) for cut in NTplot_cuts]
        dt_cuts = [list(specdata['start_date'])[0].to_pydatetime()-dt.timedelta(seconds=30)] + dt_cuts + [list(specdata['start_date'])[-1].to_pydatetime()+dt.timedelta(seconds=30)]
        for i in range(len(energy_thresholds)):
            en = energy_thresholds[i]

            rgba_color = cm.nipy_spectral_r(norm(i),bytes=True) [:3]
            rgba_color= [x/255. for x in rgba_color]
            for c in range(len(NTplot_cuts)+1):
                ixx_cut = np.logical_and(specdata['start_date']>=dt_cuts[c],specdata['start_date']<dt_cuts[c+1])
                lbl = f"Above {en} keV" if c==0 else None

                ax4a.plot(specdata['start_date'][ixx_cut],specdata[f'integrated_above_{en}kev'][ixx_cut],label=lbl,color=rgba_color,ls="-",lw=1.5)
                ax3a.plot(specdata['start_date'][ixx_cut],-specdata['DeltaLow'][ixx_cut],c="white",lw=curve_lw,ls="--")


        ax4a.grid()
        ax4a.set_yscale("log")
        ax4a.legend(ncols=2,title="Integrated electron Flux [s$^{-1}$]")

        if(timegrid):
            for axx  in [ax1,ax2,ax3,ax4]:
                ax_paint_grid(axx,which="major",axis="x")








        return fig
