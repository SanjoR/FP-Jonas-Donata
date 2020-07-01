import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import numpy as np
from scipy.special import sph_harm

Winkel = np.arange(0,190,10)

Winkel = np.pi * Winkel /180 

Amplitude = np.genfromtxt("../Data/Data_9mmzwischenring.txt", unpack = True)/1000

Amplitude= Amplitude/Amplitude.max()

def new_sph(x,m,n):
    return(sph_harm(m,n,np.pi/4, x))


max_winkel = new_sph(Winkel,1,1).max()

Winkel_plot = np.linspace(0,np.pi,200)




for n in range (3):
    ax = plt.subplot(111, projection='polar')
    ax.plot(Winkel, Amplitude, "r-", lw = 5 ,label = "Messdaten")
    for m in range(-2,3):
        if abs(m) <= n:
            ax.plot(Winkel_plot,new_sph(Winkel_plot,m,n)   , label= f"m={m}, n={n}")
    ax.set_rmax(1)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc = "best")
    plt.savefig(f"../latex-template/figure/9mmZwischenring_n{n}.pdf")
    plt.close()