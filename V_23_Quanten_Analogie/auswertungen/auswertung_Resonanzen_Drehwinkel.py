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
    return(sph_harm(m,n,0, x)) 

Winkel_plot = new_Winkel(np.linspace(0,np.pi,100))

ax = plt.subplot(111, projection='polar')
ax.plot(new_Winkel(Winkel), Amplitude_2, "r-", lw= 3, label = "Messwerte")
ax.plot(new_Winkel(Winkel),Amplitude_2.max()*new_sph(Winkel,0,0)   , label= f"m={0}, n={0}")
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)
plt.legend(loc= "best")
#plt.show()
plt.close()





for n in range (1,5):
    ax = plt.subplot(111, projection='polar')
    ax.plot(new_Winkel(Winkel), Amplitude_4, "r-", lw= 3,label = "Messwerte")
    ax.plot(Winkel_plot,Amplitude_4.max()*new_sph(Winkel_plot,0,n),"b", label= f"n={n}")
#ax.set_rmax(2)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc= "best")
    plt.savefig(f"../latex-template/figure/Resonanz_Drewinkel_Amplitude_4_n{n}.pdf")
    plt.close()



for n in range (4,8):
    ax = plt.subplot(111, projection='polar')
    ax.plot(new_Winkel(Winkel), Amplitude_6, "r-", lw= 3,label = "Messwerte")
    ax.plot(Winkel_plot,Amplitude_6.max()*new_sph(Winkel_plot,0,n) ,"b-"  , label= f"n={n}")
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    plt.legend(loc= "best")
    plt.savefig(f"../latex-template/figure/Resonanz_Drewinkel_Amplitude_6_n{n}.pdf")
    plt.close()
#ax.set_rmax(2)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
#ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
#ax.grid(True)
#plt.legend(loc= "best")
#plt.show()
#plt.close()

