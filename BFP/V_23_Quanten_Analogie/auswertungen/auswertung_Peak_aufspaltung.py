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

def linear(x,a,b):
    return(a*x + b)

print(peak_one_3mm)
print(peak_one_9mm)
print(peak_one_12mm)
print()
print(peak_two_3mm)
print(peak_two_9mm)
print(peak_two_12mm)
print()
Aufspaltung = [peak_two_3mm - peak_one_3mm, peak_two_9mm - peak_one_9mm, peak_two_12mm - peak_one_12mm]
for i in Aufspaltung:
    print(i)
Dicke = [3, 9, 12]

x_plot = np.linspace(3,12,2)

params,covar_matrix = curve_fit(linear, Dicke, Aufspaltung)

error = np.sqrt(np.diag(covar_matrix))

uparams = unp.uarray(params,error)

for i in range(2):
    print(uparams[i])

plt.figure()
plt.plot(Dicke, Aufspaltung,"rx")
plt.plot(x_plot,linear(x_plot,*params))
plt.xlabel("d / mm")
plt.ylabel(r"$\Delta$ f / Hz")
#plt.savefig("../latex-template/figure/Peak_Aufspaltung.pdf")
plt.close()
