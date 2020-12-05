import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import numpy as np
from scipy.special import sph_harm

Winkel = np.arange(0,190,10)

for i in Winkel:
    print(i)

Winkel = np.pi * Winkel /180 

Amplitude = np.genfromtxt("../Data/Data_wasserstoffmolekühl.txt", unpack = True)/1000
for i in Amplitude:
    print(i)
Winkel_plot = np.linspace(0,2* np.pi, 100)
def new_sph(x,m,n):
    return(sph_harm(m,n,np.pi/4, x))



ax = plt.subplot(111, projection='polar')
ax.plot(Winkel, Amplitude, "rx", lw = 5 )
for n in range (3):
    for m in range(-2,3):
        if abs(m) <= n:
            ax.plot(Winkel_plot,2*np.absolute(np.real(sph_harm(m,n,Winkel_plot,np.pi/4)))   , label= f"m={m}, l={n}")
ax.set_rmax(1.5)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)
plt.legend(loc = "best")
plt.savefig("../latex-template/figure/WMQZ.pdf")
plt.show()
plt.close()