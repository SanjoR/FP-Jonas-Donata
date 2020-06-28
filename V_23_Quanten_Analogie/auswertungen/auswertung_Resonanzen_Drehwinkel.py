import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import numpy as np
from scipy.special import sph_harm

Winkel = np.arange(0,190,10)

Winkel = np.pi * Winkel /180 

def new_Winkel(x):
    return(np.arccos( np.sqrt(2) * np.sin(x) / (1/2 + np.sqrt(2) * np.sin(x))))

print(new_Winkel(Winkel))
Amplitude_4, Amplitude_6 =np.genfromtxt('../Data/Data_Resonanzen_Drehwinkel.txt',unpack=True)


ax = plt.subplot(111, projection='polar')
ax.plot(180*new_Winkel(Winkel)/np.pi, Amplitude_4, "r-")
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)
plt.show()
plt.close()

ax = plt.subplot(111, projection='polar')
ax.plot(180*new_Winkel(Winkel)/np.pi, Amplitude_6, "r-")
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)
plt.show()
plt.close()