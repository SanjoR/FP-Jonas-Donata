import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import numpy as np

Data_3mm = np.genfromtxt('../Bilder/2p3Wasserstoff3mm.dat',unpack = True)
Data_9mm = np.genfromtxt('../Bilder/2p3Wasserstoff9mm.dat',unpack = True)
Data_12mm = np.genfromtxt('../Bilder/2p3Wasserstoff9mm3mm.dat',unpack = True)

peak_one_3mm = Data_3mm[0][0:160][np.argmax(Data_3mm[1][0:160]) ]
peak_two_3mm = Data_3mm[0][160:len(Data_3mm[1]) -1][np.argmax(Data_3mm[1][160:len(Data_3mm[1]) -1]) ]
peak_one_9mm = Data_9mm[0][0:160][np.argmax(Data_9mm[1][0:160]) ]
peak_two_9mm = Data_9mm[0][160:len(Data_9mm[1]) -1][np.argmax(Data_9mm[1][160:len(Data_9mm[1]) -1]) ]
peak_one_12mm = Data_12mm[0][0:160][np.argmax(Data_12mm[1][0:160]) ]
peak_two_12mm = Data_12mm[0][160:len(Data_12mm[1]) -1][np.argmax(Data_12mm[1][160:len(Data_12mm[1]) -1]) ]


Aufspaltung = [peak_two_3mm - peak_one_3mm, peak_two_9mm - peak_one_9mm, peak_two_12mm - peak_one_12mm]

Dicke = [3, 9, 12]

plt.figure()
plt.plot(Dicke, Aufspaltung)
plt.show()
plt.close()
