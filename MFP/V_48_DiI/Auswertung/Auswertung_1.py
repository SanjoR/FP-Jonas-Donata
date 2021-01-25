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
    T,I = np.genfromtxt("../Data/Data_A.txt", unpack = True, dtype = float)

if sys.argv[1] == "B":
    print("B")
    T,I = np.genfromtxt("../Data/Data_B.txt", unpack = True)

print(T)

T = T+273



if sys.argv[1] == "A":
    mask = np.logical_or(T<240, T>272)
    T_U = T[mask]
    T_S = T[~mask]
    I_U = I[mask]
    I_S = I[~mask]

    mask2 = T_U < 317
    T_U_R = T_U[mask2]
    I_U_R = I_U[mask2]

    T_U_N = T_U[~mask2]
    I_U_N = I_U[~mask2]

def Untergrund(Temp, A,B,Tx,C):
    return(A* np.exp(B*(Temp-Tx)) +C)





params_untergrund,cov_ma_untergrund = curve_fit(Untergrund,T_U_R,I_U_R)

def j(T,Amp,W):
    return Amp* np.exp(-W/(const.k *T))


#params, cov_ma = curve_fit(T,I,j)

x_plot = np.linspace(T_U_R.min(),T_U_R.max(),1000)

plt.figure()
plt.plot(T_U_R,I_U_R,"rx", label ="Messdaten Untergrund")
plt.plot(T_U_N,I_U_N,"kx", label ="Messdaten Untergrund")
plt.plot(T_S,I_S,"bx", label ="Messdaten Signal")
plt.plot(x_plot,Untergrund(x_plot,*params_untergrund),"r-",label = "Untergrundfit")

plt.grid()
plt.xlabel("Temperatur in K")
plt.ylabel("I in pA")
plt.legend(loc ="best")
plt.show()
