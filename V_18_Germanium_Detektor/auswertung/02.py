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


A = np.genfromtxt("../Data/02.txt", unpack=True)

Kanal = np.arange(0,len(A))


m = ufloat(0.40312,0.00009)
b = ufloat(-2.72,0.18)


def linear_value(x):
    return(0.40312*x -2.72)


def linear(x):
    return(m*x +b)

E = linear(Kanal)

for i in range(len(E)):
    if i == 0:
        E_value = np.array([E[i].n])
        E_s = np.array([E[i].s])
    else:
        E_value = np.append(E_value,E[i].n)
        E_s = np.append(E_s,E[i].s)


peaks,_ = find_peaks(A, height= 200)

E_gamma = E[peaks][0]

epsilon = E_gamma / (511)

E_rückstreu = E_gamma * 1/(1+2*epsilon)

T_max = E_gamma * 2*epsilon/(1+2*epsilon)


plt.figure(figsize = (13,8))
plt.plot(E_value,A,"b-",label = r"$Cs^{137}$ Spektrum")
plt.plot(E_value[peaks],A[peaks], "rx",label = f"Photonpeak bei {np.around(E_value[peaks],1)[0]}$\pm${np.around(E_s[peaks], 1)[0]}  keV")
plt.plot([T_max.n,T_max.n],[0,A.max()], "k--", alpha = 0.7 ,label = "Comptonkante")
plt.plot([E_rückstreu.n,E_rückstreu.n],[0,A.max()], "g--", alpha = 0.7 ,label = "Rückstreupeak")
plt.legend(loc="best")
plt.xlabel("E / keV")
plt.ylabel("Zählrate")
plt.grid()
plt.savefig("../latex-template/figure/02_peaks.pdf")
#plt.show()
plt.close()


print("E_gamma ",E_gamma)
print("T_max ",T_max)
print("E_rückstreu ",E_rückstreu)


def get_PeakArrays(k):
    K_r = Kanal[Kanal>= k-20]
    K_r = K_r[K_r<=k+20]
    A_r = []
    for i in K_r:
        A_r=np.append(A_r,A[Kanal == i])
    return(K_r,A_r)

def potenz(c,Amp, k,b):
    return(Amp*np.exp(-k*(c-c_0)**2) +b)
def upotenz(c,Amp, k,b):
    return(Amp*unp.exp(-k*(c-c_0)**2) +b)
c_0 = E_gamma.n
k,b=get_PeakArrays(Kanal[peaks][0])
a = linear_value(k)
a_err = linear(k)

c_0 = Kanal[peaks][0]
params_N,cov_matrix_N = curve_fit(potenz,k,b)
uparams_N = unp.uarray(params_N, np.sqrt(np.diag(cov_matrix_N)))
N = uparams_N[0]*unp.sqrt(np.pi/uparams_N[1])
print("N ",N)

c_0 = E_gamma.n
params,cov_matrix = curve_fit(potenz,a,b)





uparams = unp.uarray(params, np.sqrt(np.diag(cov_matrix)))

bruch=2
def potenz_prozent(c):
    return(upotenz(c,*uparams) - b.max()/bruch)


T_2 = np.array([newton(potenz_prozent,660), newton(potenz_prozent,662)])
print()
print("Halbwertsbreite")
print(T_2[1]-T_2[0])
bruch =10

T_10 = np.array([newton(potenz_prozent,660), newton(potenz_prozent,664)])
print()
print("Zehntelwertsbreite")
print(T_10[1]-T_10[0])

T_2_plot = np.array([T_2[0].n,T_2[1].n])
T_10_plot = np.array([T_10[0].n,T_10[1].n])


a_plot = np.linspace(a.min(),a.max(),1000)
plt.figure(figsize=(13,8))
plt.plot(a,b,"rx",label = "Messwerte")
plt.plot(T_2_plot, potenz(T_2_plot,*params),"k--", alpha=0.7,label="Halbwertsbreite")
plt.plot(T_10_plot, potenz(T_10_plot,*params),"k--", alpha=0.7,label="Zehntelwertsbreite")
plt.plot(a_plot,potenz(a_plot,*params),"b-",label="Fit")
plt.xlabel("E / keV")
plt.ylabel("Zählrate")
plt.legend(loc="best")
plt.grid()
plt.savefig("../latex-template/figure/02_peak_fit.pdf")
plt.show()
plt.close()

