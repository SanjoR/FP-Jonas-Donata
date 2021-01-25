import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.signal import find_peaks
import sys

if sys.argv[1] == "A":
    print("A")
    T,I = np.genfromtxt("../Data/Data_A.txt", unpack = True)

if sys.argv[1] == "B":
    print("B")
    T,I = np.genfromtxt("../Data/Data_B.txt", unpack = True)

T = T+273
print(T)

def Num_int(Temp):
    integral= 0
    for i in range(len(T)):
        if i !=0:
            if T[i]> Temp:
                #print("test")
                integral += (T[i]-T[i-1]) * (I[i-1]+I[i])/2
                #print(integral)
    return(integral)
  
F = []
for i in range(len(T)):
    F = np.append(F,Num_int(T[i])/I[i])
F = np.log(F)
def F_fit(Temp,a,b):
    return( a* 1/Temp + b)

#params, cov_ma = curve_fit(T,F,F_fit)
print(F)

plt.figure()
plt.plot(T,F,"rx",label= "Messdaten")
plt.grid()
plt.legend(loc = "best")
plt.show()