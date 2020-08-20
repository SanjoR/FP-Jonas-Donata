import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as const
from time import time 
from scipy.signal import find_peaks
import pandas as pd

start_time = time()
A = np.genfromtxt("../Data/01.txt", unpack=True)
P,I,K_plot= np.genfromtxt("../Data/01_peaks.txt", unpack=True)

Kanal = np.arange(0,len(A))


def linear(x,m,b):
    return(m*x +b)


params,cov_matrix = curve_fit(linear,K_plot,P)

error = np.sqrt(np.diag(cov_matrix))
uparams = unp.uarray(params, error)
for i in uparams:
    print(i)


A_plot = []
for i in K_plot:
    A_plot = np.append(A_plot,A[Kanal==i])

plt.figure(figsize=(13,8))
plt.plot(Kanal,A)
plt.plot(K_plot,A_plot, "rx")
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(K_plot,P, "rx")
plt.plot(K_plot,linear(K_plot,*params))
plt.show()
plt.close()





lam = np.log(2)/13.516

A_0 = ufloat(4130,60)
N_0 = A_0/lam

def Akti(t):                    ## t in Jahren
    return(lam * N_0 *np.exp(-lam * t ))


T = 19 + (1/12)
print(T)

print(Akti(T))