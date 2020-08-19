import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as const
from time import time 

start_time = time()
A = np.genfromtxt("../Data/01.txt", unpack=True)

Kanal = np.arange(0,len(A))


plt.figure()
plt.plot(Kanal,A)
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