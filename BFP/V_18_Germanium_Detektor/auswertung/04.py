import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as const
from time import time 
from scipy.signal import find_peaks
import pandas as pd
from scipy.optimize import newton
import sympy

A = np.genfromtxt("../Data/04.txt", unpack=True)


Kanal = np.arange(0,len(A))


m = ufloat(0.40312,0.00009)
b = ufloat(-2.72,0.18)


def linear_value(x):
    return(0.40312*x -2.72)

E = linear_value(Kanal)

E_err = m*Kanal +b

peaks1,_ = find_peaks(A[E<634],height = 1000)
peaks2,_ = find_peaks(A[E>=634] , height = 75)



def linear(x):
    return(m*x +b)


E_peaks = np.append(E[E<634][peaks1],E[E>=634][peaks2])
E_peaks_err = np.append(E_err[E<634][peaks1],E_err[E>=634][peaks2])
Kanal_peaks = np.append(Kanal[E<634][peaks1],Kanal[E>=634][peaks2])
A_peaks = np.append(A[E<634][peaks1],A[E>=634][peaks2])
identifiziert = np.array([3,4,5,6,7,8,10,11,12,13,14,17,18,21,22,24])
E_ident_err = E_peaks_err[identifiziert]
Kanal_ident = Kanal_peaks[identifiziert]
E_ident=E_peaks[identifiziert]
A_ident=A_peaks[identifiziert]
for i in E_ident:
    print(i)
print()
for i in E_ident_err:
    print(i)
print()
for i in range(len(E_peaks_err)):
    if i not in identifiziert:
        print(i, E_peaks_err[i])

print(len(E_ident))
print(len(E_peaks))
plt.figure(figsize = (13,8))
plt.plot(E,A,"b-",label="Spektrum")
plt.plot(E_peaks,A_peaks,"rx",label="Peaks")
plt.plot(E_ident,A_ident, "k*",label = "Identifiziert")
plt.legend(loc="best")
plt.xlabel("E / keV")
plt.ylabel("Zählrate")
plt.grid()
plt.savefig("../latex-template/figure/04_peaks.pdf")
plt.show()
plt.close()


Amp_eta= ufloat(0.22290932272721348, 0.04012126575134311)
z_eta= ufloat(-0.8633735220461703, 0.03372217903685505)
def eta_fit(e):  #Vollenergienachweißwahrscheinlichkeit
    return(Amp_eta*e**z_eta)

def get_PeakArrays(k):
    K_r = Kanal[Kanal>= k-20]
    K_r = K_r[K_r<=k+20]
    A_r = []
    for i in K_r:
        A_r=np.append(A_r,A[Kanal == i])
    return(K_r,A_r)

def potenz(c,Amp, k,b):
    return(Amp*np.exp(-k*(c-c_0)**2) +b)


N = []
K_plot_2 = []
fig,axs = plt.subplots(3,3,figsize = (16,13))
n=0
m=0
for i in range(len(Kanal_ident)):
    c_0 = Kanal_ident[i]
    a,b=get_PeakArrays(Kanal_ident[i])
    params,cov_matrix = curve_fit(potenz,a,b)
    if i == 0:
        params_array = [params]
        error_array = [np.sqrt(np.diag(cov_matrix))]
    else:
        params_array = np.append(params_array , [params], axis = 0)
        error_array = np.append(error_array, [np.sqrt(np.diag(cov_matrix))], axis =0)
    N = np.append(N,params[0]*np.sqrt(np.pi/params[1]))
    x_plot =np.linspace(a.min(),a.max(),1000)
    axs[n,m].plot(x_plot,potenz(x_plot,*params),"b-")
    axs[n,m].plot(a,b,"rx", label = f"Peaknummer {i}")
    axs[n,m].grid()
    axs[n,m].legend(loc="best")
    m+=1
    if m > 2:
        m = 0 
        n+=1
#axs[1,2].set_visible(False)
#plt.show()
plt.savefig("../latex-template/figure/04_subplot.pdf")
plt.close()
