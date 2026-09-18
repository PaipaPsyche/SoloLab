import numpy as np
from astropy import constants as _const
from astropy import units as _u
from scipy import stats

# CONSTANTS

#date
std_date_fmt = '%d-%b-%Y %H:%M:%S'
simple_date_fmt='%Y-%m-%d'
numeric_date_fmt='%Y-%m-%d %H:%M:%S'
fits_date_fmt="%Y%m%dT%H%M%S"



# Physical constants, derived from astropy.constants/units instead of
# hand-transcribed literals: m_e_kg used to be hardcoded as 9.31e-31 (~2.2%
# off from CODATA), silently biasing every FDRA beam-energy result until
# that was caught - deriving from astropy means the value always tracks
# the current CODATA release with no manual-transcription step to get
# wrong again. WmHz_per_sfu is a domain unit definition (radio astronomy's
# "solar flux unit"), not a measured physical constant, so it stays a
# literal - astropy has no built-in equivalent.
speed_c_kms = _const.c.to(_u.km / _u.s).value  # km/s
m_e_kg = _const.m_e.to(_u.kg).value  # kg
Rs_per_AU = (1 * _u.AU).to(_u.R_sun).value
km_per_Rs = _const.R_sun.to(_u.km).value
WmHz_per_sfu = 1e-22
ev_per_joule = (1 * _u.J).to(_u.eV).value  # eV per joule
ergs_per_kev = (1 * _u.keV).to(_u.erg).value  # ergs per keV
kb_in_kev_per_K = _const.k_B.to(_u.keV / _u.K).value  # keV/K


# rpw indexes
rpw_suggested_freqs_idx =[437,441,442,448,453,458,465,470,477,482,
                      493,499,511,519,526,533,
                      538,545,552,559,566,576,588,592,600,612,
                      656,678,696,716,734,741,750,755,505,629,649,
                      673,703,727]

tnr_remove_idx =[75,76,77,78,79,80,85,86,103,104,110,111,115,116,118,119]
#rpw_suggested_freqs_idx=[ 437,441,442,448,453,458,465,470,
#                         477,482,493,499,511,519,526,533,
#                         538,545,552,559,566,600,
#                         612,678,696,#,576,588,592,649,629,656,673
#                         703,716,727,734,741,750,755]
rpw_idx_hfr=436
rpw_suggested_indexes = np.array(rpw_suggested_freqs_idx)-rpw_idx_hfr

# Sweep counter jumps larger than this (between consecutive samples) are
# treated as a hardware sweep-counter reset/wrap, not real sweep progress.
rpw_sweep_reset_threshold = 100

# HFR channel-to-frequency mapping: channel 0 starts at this frequency,
# with this fixed step between channels (both in kHz).
rpw_hfr_freq_min_khz = 375.0
rpw_hfr_freq_step_khz = 50.0

# Valid instrument bounds used to validate/clamp user-entered ranges in the
# GUI (STIX energy in keV, RPW frequency in kHz).
stix_energy_range_min_kev = 4
stix_energy_range_max_kev = 150
rpw_freq_range_min_khz = 1
rpw_freq_range_max_khz = 16400

#display_freqs=[0,100,500,1000,2500,5000,10000,15000]
display_freqs=[500,1000,2000,3000,5000,8000,100000,16000]
display_freqs_tnr=[20,50,100,200,400,600,900]

def filter_nan(A, B):
    if len(A) != len(B):
        raise ValueError("Lists A and B must have the same length")
        
    filtered_A = []
    filtered_B = []
    
    for a, b in zip(A, B):
        if not np.isnan(a):
            filtered_A.append(a)
            filtered_B.append(b)
            
    return filtered_A, filtered_B



def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def verbose_dts(timearr):
    timearr = np.asarray(timearr)
    if timearr.dtype == object:
        try:
            timearr = np.array(timearr, dtype="datetime64[ns]")
        except ValueError:
            print("  Data info: time array dtype not supported for dt stats")
            return
    if len(timearr) < 2:
        print("  Data info: not enough samples for dt stats")
        return
    tdts = timearr[1:] - timearr[:-1]
    tdts = tdts / np.timedelta64(1, 's')
    tdts = tdts.astype(float)

    print('  Data info:')
    print(f'   Dt : Max = {np.max(tdts)}s  Min = {np.min(tdts)}s  Avg = {round(np.mean(tdts),1)}s  std = {round(np.std(tdts),1)}s  Median:{np.median(tdts)}s   Mode: {stats.mode(tdts,keepdims=True)[0][0]}s ({round(100*stats.mode(tdts,keepdims=True)[1][0]/len(tdts),1)}%)')




polling_functions =  {
            'mean':np.mean,
            'geo_mean':geo_mean,
            'median':np.median,
            'min':np.min,
            'max':np.max,
        }

def get_poll_func(f):
    if(not f in polling_functions.keys() and f.startswith('P_')):
        p_value = float(f.split('_')[1])
        def percentile_poll(arr,axis=1):
            return np.percentile(arr,p_value,axis=axis)
        return percentile_poll

    kk = list(polling_functions.keys())
    if(not f in kk):
        print("  [Error] background polling function is not defined. you can try instead: ",*kk)
    return polling_functions[f]