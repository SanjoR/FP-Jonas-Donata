import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const


L,I = np.genfromtxt("../Data/TEM01.txt", unpack = True)

#I *= 1/np.sum(I)
#print(np.sum(I))
L = L-L[I== I.min()]

L_min= L[L<0]
I_min= I[L<0]
L_max=L[L>=0]
I_max=I[L>=0]


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaus_2(x,mu1,sig1,mu2,sig2):
    return(gaussian(x,mu1,sig1)+gaussian(x,mu2,sig2))
Lvalue_min = L_min[I_min==I_min.max()]
Lvalue_max = L_max[I_max==I_max.max()]

params_min,covma_min=curve_fit(gaussian,L_min,I_min)
params_max,covma_max=curve_fit(gaussian,L_max,I_max)

params,covma = curve_fit(gaus_2,L,I/np.sum(I),p0=(-8,10,8,10))

errors_min = np.sqrt(np.diag(covma_min))
errors_max = np.sqrt(np.diag(covma_max))

x_plot = np.linspace(L.min(),L.max(),1000)

plt.figure()
plt.plot(L,I/np.sum(I),"rx",label="Messdaten")
#plt.plot(x_plot,gaussian(x_plot,*params_min),"g--",label="Gaußfit linkes Maximum")
#plt.plot(x_plot,gaussian(x_plot,*params_max),"b--",label="Gaußfit rechtes Maximum")
#plt.plot(x_plot,gaussian(x_plot,*params_max)+gaussian(x_plot,*params_min),"k-",label="Gaußfit Superposition")
plt.plot(x_plot,gaus_2(x_plot,*params))
#
#
#plt.xlabel("Abstand / mm")
#plt.ylabel("Intensität")
#plt.grid()
#plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(L,I/np.sum(I),"rx",label="Messdaten")
plt.plot(x_plot,gaus_2(x_plot,-8,10,8,10))#/np.sum(gaus_2(x_plot,-8,10,8,10)))
plt.show()