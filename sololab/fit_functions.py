#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
from .values import *
from .quicklooks import *

#math
import numpy as np
import scipy.special as spf
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde



class intensity_model:
    def __init__(self,f,f_center,f_dev,f_base,f_onset=None,err_center=None,err_onset=None,err_base=None):
        self.f = f

        self.center = f_center
        self.err_center=err_center

        self.dev = f_dev

        self.base  = f_base
        self.err_base=err_base

        self.f_onset=f_onset
        self.err_onset=err_onset

        self.func_name = ""


    # name definition
    def name(self,name=None):
        if(self.func_name != ""):
            return self.func_name
        elif(name):
            self.func_name = name.lower()
        else:
            print(" [Error] Function name was not defined/provided!")
            return ""

    # height value functions
    def maxval(self,par):
        return self.f(self.center(par),*par)
    def peak_height(self,par):
        return self.maxval(par)-self.base(par)
    def midval(self,par):
        return self.base(par)+self.peak_height(par)/2
    # dev functions
    def mindev(self,par):
        return self.center(par)-self.dev(par)
    def maxdev(self,par):
        return self.center(par)+self.dev(par)
    # onset
    def onset(self,par):
        if(not self.f_onset):
            print(" [Error] This intensity model has no onset function defined")
            return
        else:
            return self.f_onset(par)


#=============== METRICS ============================

def get_kde(data,x,l=0.2):

    density = gaussian_kde(data)

    density.covariance_factor = lambda : l
    density._compute_covariance()
    return density(x)

def estimate_rmse(x,y,model,params):
    rmse = 0
    for i in range(len(x)):
        rmse += (y-model(x,*params))**2
    return np.sqrt(np.mean(rmse))


def estimate_weighted_rmse(x,y,model,params):
    rmse = 0

    w_range = 0.5
    map_y = interp1d([np.min(y),np.max(y)],[1-w_range,1+w_range])

    for i in range(len(x)):
        rmse += map_y(y[i])*(y-model(x,*params))**2
    return np.sqrt(np.mean(rmse))


def get_metric(name):
    if(name == "rmse"):
        return estimate_rmse
    elif(name == "wrmse"):
        return estimate_weighted_rmse


# ================== FUNCTIONS==============================
# GAUSSIAN FIT FUNCTIION
def gaussian(x,m,s,a,b):
    res = b+a*np.exp(-((x-m)**2)/(2*s**2))
    return res
def gaussian_mode(par):
    m,s,a,b = par[:4]
    return m
def gaussian_dev(par):
    m,s,a,b = par[:4]
    return s
def gaussian_bkg(par):
    m,s,a,b = par[:4]
    return b
def gaussian_onset(par):
    m,s,a,b = par[:4]
    return m-2*gaussian_dev(par)

def gaussian_err_mode(cov):
    return np.sqrt(cov[0][0])
def gaussian_err_onset(cov):
    return np.sqrt(cov[0][0]+2*cov[1][1])
def gaussian_err_bkg(cov):
    return np.sqrt(cov[3][3])



#GAUSS_EXP (rise/decay times) FIT FUNCTIION
def gauss_exp(x, ton,toff,tpeak,A,b):
    if(isinstance(x, (list, tuple, np.ndarray))):
        y = np.zeros(len(x))
    else:
        x = np.array([x])
        y = np.zeros(len(x)) 
    for i, xi in enumerate(x): 
        if xi <= tpeak:
            y[i] =  b + A*np.exp(-(xi-tpeak)**2/(2*ton**2))
        elif xi >= tpeak:
            y[i] =  b + A*np.exp(-(np.abs(xi-tpeak))/(toff))
    return y

def gauss_exp_mode(pars):
    ton,toff,tpeak,A,b = pars
    return tpeak
def gauss_exp_onset(pars):
    ton,toff,tpeak,A,b = pars
    return tpeak - ton*np.sqrt(2*np.log10(2))
def gauss_exp_bkg(pars):
    ton,toff,tpeak,A,b = pars
    return b
def gauss_exp_dev(pars):
    ton,toff,tpeak,A,b = pars
    return (ton+toff)/2

def gauss_exp_err_mode(covs):
    return np.sqrt(covs[2][2])
def gauss_exp_err_onset(covs):
    return np.sqrt(covs[0][0])
def gauss_exp_err_bkg(covs):
    return np.sqrt(covs[4][4])


#GAUSS_GAUSS (rise/decay times) FIT FUNCTIION


def gauss_gauss(x, ton,toff,tpeak,A,b):
    if(isinstance(x, (list, tuple, np.ndarray))):
        y = np.zeros(len(x))
    else:
        x = np.array([x])
        y = np.zeros(len(x)) 
    for i, xi in enumerate(x): 
        if xi <= tpeak:
            y[i] =  b + A*np.exp(-(xi-tpeak)**2/(2*ton**2))
        elif xi >= tpeak:
            y[i] =  b + A*np.exp(-(xi-tpeak)**2/(2*toff**2))
    return y

def gauss_gauss_mode(pars):
    ton,toff,tpeak,A,b = pars
    return tpeak
def gauss_gauss_onset(pars):
    ton,toff,tpeak,A,b = pars
    return tpeak - ton*np.sqrt(2*np.log10(2))

def gauss_gauss_bkg(pars):
    ton,toff,tpeak,A,b = pars
    return b
def gauss_gauss_dev(pars):
    ton,toff,tpeak,A,b = pars
    return (ton+toff)/2

def gauss_gauss_err_mode(covs):
    return np.sqrt(covs[2][2])
def gauss_gauss_err_onset(covs):
    return np.sqrt(covs[0][0])
def gauss_gauss_err_bkg(covs):
    return np.sqrt(covs[4][4])










# FLUORESCENCE (rise/decay times) FIT FUNCTIION
def fluorescence(x,ton,toff,a,b,p):
    try:
        if (x-p)<0:
            return b
        else:
            res =  b+a*np.exp(-(x-p)/toff)*(1-np.exp(-(x-p)/ton))
            if x>0:
                return res
            else:
                return b
    except:
        return np.array([fluorescence(y,ton,toff,a,b,p) if y>0 else b for i,y in enumerate(x) ])
    res = b+a*np.exp(-(x-p)/toff)*(1-np.exp(-(x-p)/ton))
    return res
def fluorescence_mode(par):
    ton,toff,a,b,p = par[:5]
    return ton*np.log((toff+ton)/ton)+p
def fluorescence_dev(par):
    ton,toff,a,b,p = par[:5]
    return ton
def fluorescence_bkg(par):
    ton,toff,a,b,p = par[:5]
    return b
def fluorescence_onset(par):
    ton,toff,a,b,p = par[:5]
    #return p
    #return fluorescence_mode(par)-ton*np.log10(2)
    return (p + fluorescence_mode(par))/2
def fluorescence_err_mode(cov):
    return np.sqrt(cov[0][0]*np.log((cov[1][1]+cov[0][0])/cov[0][0])+cov[4][4])
def fluorescence_err_onset(cov):
    return np.sqrt(cov[4][4])
def fluorescence_err_bkg(cov):
    return np.sqrt(cov[3][3])
# WEIBULL FIT FUNCTION
def weibull(x,l,k,s,b,p):

    try:
        if (x-p)<0:
            return b
        else:
            res = s*(k/l)*(((x-p)/l)**(k-1))*np.exp(-((x-p)/l)**k)+b
            if x>0:
                return res
            else:
                return b
    except:
        return np.array([weibull(y,l,k,s,b,p) if y>0 else b for i,y in enumerate(x) ] )
def weibull_mode(par):
    l,k,s,b,p = par[:5]
    return l*((k-1)/k)**(1/k)+p
def weibull_dev(par):
    l,k,s,b,p = par[:5]
    var = l*l*( (spf.gamma(1+(2/k)) - (spf.gamma(1+(1/k)))**2 ))
    return np.sqrt(var)
def weibull_bkg(par):
    l,k,s,b,p = par[:5]
    return b
def weibull_onset(par):
    l,k,s,b,p = par[:5]
    return max(weibull_mode(par)-3*weibull_dev(par),p)


# ================== DEFINITIONS==============================
f_gaussian=  intensity_model(gaussian,gaussian_mode,gaussian_dev,gaussian_bkg,gaussian_onset,
                            err_center=gaussian_err_mode,err_onset = gaussian_err_onset,err_base = gaussian_err_bkg)
f_gaussian.name("Gaussian")

f_fluorescence=  intensity_model(fluorescence,fluorescence_mode,fluorescence_dev,fluorescence_bkg,fluorescence_onset,
                                err_center=fluorescence_err_mode,err_onset = fluorescence_err_onset,err_base = fluorescence_err_bkg)
f_fluorescence.name("Fluorescence")


f_gauss_exp=  intensity_model(gauss_exp,gauss_exp_mode,gauss_exp_dev,gauss_exp_bkg,gauss_exp_onset,
                                err_center=gauss_exp_err_mode,err_onset = gauss_exp_err_onset,err_base = gauss_exp_err_bkg)
f_gauss_exp.name("Gauss_exp")

f_gauss_gauss=  intensity_model(gauss_gauss,gauss_gauss_mode,gauss_gauss_dev,gauss_gauss_bkg,gauss_gauss_onset,
                                err_center=gauss_gauss_err_mode,err_onset = gauss_gauss_err_onset,err_base = gauss_gauss_err_bkg)
f_gauss_gauss.name("Gauss_gauss")


f_weilbull=  intensity_model(weibull,weibull_mode,weibull_dev,weibull_bkg,weibull_onset)
f_weilbull.name("Weibull")


# CATALOG ====================================================
FUNCTION_CATALOG = [f_gaussian,f_fluorescence,f_weilbull,f_gauss_exp,f_gauss_gauss]
FUNCTION_CATALOG = {x.name():x for  x in FUNCTION_CATALOG}

def get_fit_function(name):
    try:
        return FUNCTION_CATALOG[name]
    except:
        print("[ERROR] provided name [{}] not found in function catalog. Available functions:".format(name))
        print(*list(FUNCTION_CATALOG.keys()))
