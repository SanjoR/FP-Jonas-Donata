import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

Tau,A = np.genfromtxt("../Data/Diffmessung.txt", unpack = True)


def M(tau,M0,TD,M1):
    return(M0*np.exp(-2*tau/1.4) * np.exp(-tau**3 / TD) +M1)

params,cov_ma = curve_fit(M,Tau,A)

errors = np.sqrt(np.diag(cov_ma))


x_plot= np.linspace(Tau.min(),Tau.max(),1000)


plt.plot(x_plot**3 , np.log(M(x_plot,*params)) -2*x_plot/1.4)
plt.plot(Tau**3 ,np.log(A)-2*Tau/1.4,"rx")
plt.show()