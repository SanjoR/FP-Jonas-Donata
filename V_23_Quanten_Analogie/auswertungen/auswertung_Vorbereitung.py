import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat




Frequenz, Amplitude, Phasenverschiebung=np.genfromtxt('../Data/Data_vorbereitung.txt',unpack=True)

First_Frequenz = Frequenz[0::2]
Second_Frequenz = Frequenz[1::2]
First_Amplitude = Amplitude[0::2]
Second_Amplitude = Amplitude[1::2]
First_Phasenverschiebung = Phasenverschiebung[0::2]
Second_Phasenverschiebung = Phasenverschiebung[1::2]


Frequenz_diff = Second_Frequenz - First_Frequenz

Längen = np.arange(50, 350,50) 
def Frequenz_plot(x,a,b):
    return(a * 1/x +b)

params,covarianze_matrix=curve_fit(Frequenz_plot,Längen,Frequenz_diff)
errors=np.sqrt(np.diag(covarianze_matrix))

print(2* ufloat(params[0],errors[0]) + ufloat(params[1], errors[1]) )

x_plot = np.linspace(50,300,100)
plt.figure()
plt.plot(Längen,Frequenz_diff ,"rx")
plt.plot(x_plot,Frequenz_plot(x_plot,*params))
plt.show()
plt.close()