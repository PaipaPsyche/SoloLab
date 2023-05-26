import numpy as np
from scipy import stats

# CONSTANTS

#date
std_date_fmt = '%d-%b-%Y %H:%M:%S'
simple_date_fmt='%Y-%m-%d'
numeric_date_fmt='%Y-%m-%d %H:%M:%S'
fits_date_fmt="%Y%m%dT%H%M%S"



speed_c_kms = 299792.458 #km/s
m_e_kg = 9.31e-31 #kg
Rs_per_AU = 215.032
km_per_Rs = 695700.
WmHz_per_sfu=1e-22
ev_per_joule = 6.24e18 # ev perjoule

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

#display_freqs=[0,100,500,1000,2500,5000,10000,15000]
display_freqs=[500,1000,2000,3000,5000,8000,100000,12000,16000]
display_freqs_tnr=[10,50,100,200,400,600,900]


def verbose_dts(timearr):
    tdts = timearr[1:] - timearr[:-1]
    tdts = np.array([x.seconds for x in tdts])

    print('  Data info:')
    print(f'   Dt : Max = {np.max(tdts)}s  Min = {np.min(tdts)}s  Avg = {round(np.mean(tdts),1)}s  std = {round(np.std(tdts),1)}s  Median:{np.median(tdts)}s   Mode: {stats.mode(tdts,keepdims=True)[0][0]}s ({round(100*stats.mode(tdts,keepdims=True)[1][0]/len(tdts),1)}%)')
