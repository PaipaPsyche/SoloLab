import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .stix_read import * 



def flux_e_low_from_total_flux(delta,total,e_low):
    return (delta-1)*(total/e_low)



class powerlaw:
    def __init__(self,delta,e_low,flux_e_low,e_high=np.inf,T=None,warnings=False):
        self.delta = delta
        self.e_low = e_low
        self.e_high = e_high
        self.flux_e_low = flux_e_low
        self.A = self.estimate_A_constant()
        self.total_flux = self.estimate_total_flux()

        self.T = T

        self.warnings = warnings

    def estimate_A_constant(self):
        return self.flux_e_low/(self.e_low**(-self.delta))

    def extrapolate(self,E):
        if(E<self.e_low):
            if(self.warnings):
                print(f"[!] The energy provided ({round(E,2)} keV) is below the low energy limit ({self.e_low} keV)")
        return self.A * ((E)**(-self.delta))
    

    def extrapolate_energy(self,E):
        if(E<self.e_low):
            if(self.warnings):
                print(f"[!] The energy provided ({round(E,2)} keV) is below the low energy limit ({self.e_low} keV)")
        return self.A * ((E)**(-self.delta+1))

    def estimate_total_flux(self):
        nd = -self.delta+1
        return (self.A/nd) * (self.e_high**(nd)- self.e_low**(nd))


    def integrate(self,E,e_low=None,above=True):
        if(not e_low):
            e_low = self.e_low
        elif(e_low<self.e_low):
            e_low = self.e_low

        nd = -self.delta+1
        ans = (self.A/nd) * (E**(nd) - e_low**(nd))
        if(above):
            if(E<self.e_low):
                return self.integrate(self.e_low,e_low=e_low,above=True)
            ans = (self.A/nd) * (self.e_high**(nd)-E**(nd))
        return ans
    

    def integrate_energy(self,E,e_low=None,above=True):
        if(not e_low):
            e_low = self.e_low
        elif(e_low<self.e_low):
            e_low = self.e_low

        nd = -self.delta+2
        ans = (self.A/nd) * (E**(nd) - e_low**(nd))
        if(above):
            if(E<self.e_low):
                return self.integrate_energy(self.e_low,e_low=e_low,above=True)
            ans = (self.A/nd) * (self.e_high**(nd)-E**(nd))
        return ans



    def plot_flux(self, erange,real_units=False,integrated=False,above=False,points=500 ,**kwargs):
#        erange[0] = max(self.e_low,erange[0])
#        xx = np.linspace(erange[0],erange[1],points)
#        yy = [self.extrapolate(e) for e in xx]
#        if(integrated):
#            yy=[self.integrate(e,above=above) for e in xx]


        xx = np.linspace(max(self.e_low,erange[0]),erange[1],points)
        yy = [self.extrapolate(e) for e in xx]
        if(integrated):
            xx = np.linspace(erange[0],erange[1],points)
            yy = [self.extrapolate(e) for e in xx]
            yy=[self.integrate(e,above=above) for e in xx]

        if(real_units):
            yy = [y*(1e35)for y in yy]
        text_T = " T = {:.2f}".format(self.T) if self.T else ""
        return plt.plot(xx,yy,label=f"$\delta$ = {self.delta}"+text_T,**kwargs)
    

    def plot_energy(self, erange,real_units=False,integrated=False,above=False,points=500 ,**kwargs):
#        erange[0] = max(self.e_low,erange[0])
#        xx = np.linspace(erange[0],erange[1],points)
#        yy = [self.extrapolate(e) for e in xx]
#        if(integrated):
#            yy=[self.integrate(e,above=above) for e in xx]


        xx = np.linspace(max(self.e_low,erange[0]),erange[1],points)
        yy = [self.extrapolate_energy(e) for e in xx]
        if(integrated):
            xx = np.linspace(erange[0],erange[1],points)
            yy = [self.extrapolate_energy(e) for e in xx]
            yy=[self.integrate_energy(e,above=above) for e in xx]

        if(real_units):
            yy = [y*(1e35)for y in yy]
        text_T = " T = {:.2f}".format(self.T) if self.T else ""
        return plt.plot(xx,yy,label=f"$\delta$ = {self.delta}"+text_T,**kwargs)


class double_powerlaw:
    def __init__(self,delta_low,e_low,flux_e_low,delta_high,e_break,T=None,warnings=False):

        self.delta_low =delta_low
        self.delta_high = delta_high

        self.e_low = e_low
        self.e_break = e_break

        self.T = T

        self.powerlaw_low = powerlaw(delta_low,e_low,flux_e_low,e_high=e_break,warnings=warnings)
        self.flux_e_break = self.powerlaw_low.extrapolate(e_break)
        self.powerlaw_high = powerlaw(delta_high,e_break,self.flux_e_break,warnings=warnings)

        self.total_flux = self.estimate_total_flux()

    def extrapolate(self,E):
        if(E<=self.e_break):
            return self.powerlaw_low.extrapolate(E)
        else:
            return self.powerlaw_high.extrapolate(E)
        

    def extrapolate_energy(self,E):
        if(E<=self.e_break):
            return self.powerlaw_low.extrapolate_energy(E)
        else:
            return self.powerlaw_high.extrapolate_energy(E)

    def estimate_total_flux(self):
        return self.powerlaw_low.total_flux + self.powerlaw_high.total_flux

    def integrate(self,E,e_low=None,above=True):
        if(not e_low):
            e_low = self.e_low
        elif(e_low<self.e_low):
            e_low = self.e_low

        if(above):
            if(E<self.e_low):
                return self.integrate(self.e_low,e_low=e_low,above=True)
            elif(E>=self.e_break):
                return self.powerlaw_high.integrate(E,above=True)
            elif(E<self.e_break):
                return self.powerlaw_low.integrate(E,above=True) + self.powerlaw_high.total_flux
        else:
            if(E<self.e_break):
                return self.powerlaw_low.integrate(E,e_low=e_low,above=False)
            elif(E>=self.e_break):
                return self.powerlaw_high.integrate(E,e_low=self.e_break,above=False) + self.powerlaw_low.integrate(self.e_break,e_low=e_low,above=False)
            

    def integrate_energy(self,E,e_low=None,above=True):
        if(not e_low):
            e_low = self.e_low
        elif(e_low<self.e_low):
            e_low = self.e_low

        if(above):
            if(E<self.e_low):
                return self.integrate_energy(self.e_low,e_low=e_low,above=True)
            elif(E>=self.e_break):
                return self.powerlaw_high.integrate_energy(E,above=True)
            elif(E<self.e_break):
                return self.powerlaw_low.integrate_energy(E,above=True) + self.powerlaw_high.integrate_energy(self.e_break,above=True)
        else:
            if(E<self.e_break):
                return self.powerlaw_low.integrate_energy(E,e_low=e_low,above=False)
            elif(E>=self.e_break):
                return self.powerlaw_high.integrate_energy(E,above=False) + self.powerlaw_low.integrate_energy(self.e_break,e_low=e_low,above=False)




    def plot_flux(self, erange,real_units=False,integrated=False,above=False,points=500,**kwargs):

        xx = np.linspace(max(self.e_low,erange[0]),erange[1],points)
        yy = [self.extrapolate(e) for e in xx]
        if(integrated):
            xx = np.linspace(erange[0],erange[1],points)
            yy = [self.extrapolate(e) for e in xx]
            yy=[self.integrate(e,above=above) for e in xx]
        if(real_units):
            yy = [y*(1e35) for y in yy]

        text_T = " T = {:.2f}".format(self.T) if self.T else ""
        return plt.plot(xx,yy,label=r"$\delta_L$ = {:.2f} , $\delta_H$ = {:.2f}".format(self.delta_low,self.delta_high) + text_T,**kwargs)


    def plot_energy(self, erange,real_units=False,integrated=False,above=False,points=500,**kwargs):

        xx = np.linspace(max(self.e_low,erange[0]),erange[1],points)
        yy = [self.extrapolate_energy(e) for e in xx]
        if(integrated):
            xx = np.linspace(erange[0],erange[1],points)
            yy = [self.extrapolate_energy(e) for e in xx]
            yy=[self.integrate_energy(e,above=above) for e in xx]
        if(real_units):
            yy = [y*(1e35) for y in yy]

        text_T = " T = {:.2f}".format(self.T) if self.T else ""
        return plt.plot(xx,yy,label=r"$\delta_L$ = {:.2f} , $\delta_H$ = {:.2f}".format(self.delta_low,self.delta_high) + text_T,**kwargs)





def plot_powerlaws(pwlws, e_range,above=False,logx=False,colors=None,styles=None):

    if(not colors):
        colors = ["b","g","r","orange","magenta","c","yellow"]
    if(not styles):
        styles = ["-","-","-","-","-","-","-"]

    fig = plt.figure(figsize=(6,6),dpi=120)
# set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # the first subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    if(logx):
        ax0.set_xscale("log")
    ax0.grid(which="both",c="#DDDDDD")
    ax0.set_ylabel("Electron flux $\Phi_e$ \n electrons  s$^{-1}$  KeV$^{-1}$")
    #line0, = ax0.plot(x, y, color='r')
    plts = []
    for i in range(len(pwlws)):
        plts.append(pwlws[i].plot_flux(e_range,real_units=True,c=colors[i],linestyle=styles[i]))
    plt.legend()
    # the second subplot



    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.set_yscale("log")
    if(logx):
        ax1.set_xscale("log")
    #line1=x.plot_flux(e_range,real_units=True,integrated=True,c="r")
    for i in range(len(pwlws)):
        pwlws[i].plot_flux(e_range,real_units=True,integrated=True,above=above,c=colors[i],linestyle=styles[i])
    #line1, = ax1.plot(x, y, color='b', linestyle='--')
    plt.setp(ax0.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.grid(which="both",c="#DDDDDD")
    ax1.set_ylabel("Integrated $\Phi_e$\n electrons  s$^{-1}$")
    ax1.set_xlabel("energy [keV]")
    # put legend on first subplot
    #ax0.legend((line0, line1), ('red line', 'blue line'), loc='lower left')

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.show()

# STIX spectroscopy data
def process_spectroscopy_table(specdata_raw,energy_thresholds = [10,20,30,40,50],include=['double','single'],time_phase=0):

    specdata=specdata_raw[specdata_raw['NTcomponent'].isin(include)]
    specdata['start_date'] =  pd.to_datetime(specdata['start_date'], format='%d-%b-%Y %H:%M:%S')+dt.timedelta(seconds=time_phase)
    specdata['end_date'] =  pd.to_datetime(specdata['end_date'], format='%d-%b-%Y %H:%M:%S')+dt.timedelta(seconds=time_phase)
    specdata['deltaTime'] = [x.seconds for x  in specdata["end_date"]-specdata['start_date']]
    specdata["LowFitBound"] = np.array([x.split('-') for x in specdata["GoodInterval"]],dtype=int)[:,0]
    specdata["HighFitBound"] = np.array([x.split('-') for x in specdata["GoodInterval"]],dtype=int)[:,1]
    def to_num(x):
        if(type(x)==str):
            y = float(x.replace(',','.'))
            return y
        elif(type(x)==float):

            return x
    num_keys = ['ChiSQ', 'EM', 'KT','Abundance', 'Integrated flux','DeltaLow','BreakE','DeltaHigh', 'LowEcutoff', 'HighEcutoff']
    specdata[num_keys]= specdata[num_keys].applymap(lambda x: to_num(x))

    # Kb in ev/kelvin
    kb=cs.k_B.to(u.eV/u.Kelvin)

    # energies KT in ev
    energiesKT = [x*1000*u.eV for x in specdata["KT"]]
    TempK = [(x/kb).to_value() for x in energiesKT]
    specdata["Temperature"] = TempK
    specdata=specdata.reset_index(drop=True)


    results=np.zeros((len(energy_thresholds),len(specdata)))

    for ix,row in specdata.iterrows():
        delta = row['DeltaLow']
        e_low = row['LowEcutoff']
        intflux = row['Integrated flux']

        flux_e_low = flux_e_low_from_total_flux(delta,intflux,e_low)

        pwl=powerlaw(delta,e_low,flux_e_low)
        for et in range(len(energy_thresholds)):
            results[et][ix]= pwl.integrate(energy_thresholds[et])*1e35

    for et in range(len(energy_thresholds)):
        specdata[f'integrated_above_{energy_thresholds[et]}kev'] = results[et]


    useful_keys=['start_date', 'end_date', 'deltaTime',
           'LowFitBound', 'HighFitBound', 'ChiSQ', 'EM', 'Temperature','Abundance', 'Integrated flux', 'DeltaLow', 'BreakE',
           'DeltaHigh', 'LowEcutoff', 'HighEcutoff']+[x for x in specdata.keys() if 'integrated_above_' in x]


    df_data = specdata[useful_keys]
    return df_data
