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

Winkel_plot = np.linspace(0,2*np.pi,200)




for n in range (2):
    ax = plt.subplot(111, projection='polar')
    ax.plot(Winkel, Amplitude, "rx", lw = 5 ,label = "Messdaten")
    ax.set_title(f"l={n}")
    for m in range(-2,3):
        if abs(m) <= n:
            ax.plot(Winkel_plot,Amplitude.max()*np.absolute(np.real(sph_harm(m,n,np.pi/4,Winkel_plot)))   , label= f"m={m}")
    ax.set_rmax(1)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc = "best")
    plt.savefig(f"../latex-template/figure/9mmZwischenring_n{n}.pdf")
   # plt.show()
    plt.close()

n=1
for m in range(-1,2):
    ax = plt.subplot(111, projection='polar')
    ax.set_title(f"l={n}")
    ax.plot(Winkel, Amplitude, "rx", lw = 5 ,label = "Messdaten")
    ax.plot(Winkel_plot,Amplitude.max()*np.absolute(np.real(sph_harm(m,n,Winkel_plot,np.pi/4)))   , label= f"m={m}")
    ax.set_rmax(1)
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc = "best")
    plt.savefig(f"../latex-template/figure/9mmZwischenring_m{m}.pdf")
    plt.show()
    plt.close()

phi=np.arange(0,2*np.pi,0.01)
theta=np.arange(0,2*np.pi,0.01)
f=sph_harm(0,1,phi,0.25*np.pi)
g=sph_harm(1,1,phi,0.25*np.pi)
h=sph_harm(0,1,0,theta)
i=sph_harm(1,1,0,theta)
plt.figure()
plt.polar(phi,np.absolute(np.real(f)), label="m=0")
plt.polar(phi,np.absolute(np.real(g)), label="m=1")
plt.legend(loc="best")
plt.show()