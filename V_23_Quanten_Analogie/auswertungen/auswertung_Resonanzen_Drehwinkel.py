import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import numpy as np
from scipy.special import sph_harm


Amplitude_4, Amplitude_6 =np.genfromtxt('../Data/Data_Resonanzen_Drehwinkel.txt',unpack=True)
Amplitude_2 =np.ones(len(Amplitude_4))*0.16

Winkel = np.arange(0,190,10)

Winkel = np.pi * Winkel /180 

def new_Winkel(x):
    return( np.arccos(0.5*np.cos(x) -0.5 )) 
for i in Winkel:
    print(round(180 * new_Winkel(i) / np.pi,1))
def new_sph(x,m,n):
    return(sph_harm(m,n,x, 0)) 

Winkel_plot = np.linspace(0,2*np.pi,1000)

ax = plt.subplot(111, projection='polar')
ax.plot(new_Winkel(Winkel), Amplitude_2, "rx", lw= 3, label = "Messwerte")
ax.plot(Winkel_plot,np.absolute(np.real(Amplitude_2.max()*sph_harm(0,0,Winkel_plot,0)))   , label= f"m={0}, n={0}")
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)
plt.legend(loc= "best")
#plt.show()
plt.close()





for n in range (1,5):
    ax = plt.subplot(111, projection='polar')
    ax.plot(new_Winkel(Winkel), Amplitude_4, "rx", lw= 3,label = "Messwerte")
    ax.plot(Winkel_plot,np.absolute(np.real(Amplitude_4.max()*sph_harm(0,n,0,Winkel_plot))),"b", label= f"l={n} m=0")
#ax.set_rmax(2)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc= "best")
    plt.savefig(f"../latex-template/figure/Resonanz_Drewinkel_Amplitude_4_n{n}.pdf")
    plt.show()
    plt.close()



for n in range (1,4):
    ax = plt.subplot(111, projection='polar')
    ax.plot(new_Winkel(Winkel), Amplitude_6, "rx", lw= 3,label = "Messwerte")
    ax.plot(Winkel_plot,np.absolute(np.real(Amplitude_6.max()*sph_harm(0,n,0,Winkel_plot))) ,"b-"  , label= f"l={n} m=0")
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc= "best")
    plt.savefig(f"../latex-template/figure/Resonanz_Drewinkel_Amplitude_6_n{n}.pdf")
    plt.show()
    plt.close()
#ax.set_rmax(2)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
#ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
#ax.grid(True)
#plt.legend(loc= "best")
#plt.show()
#plt.close()

