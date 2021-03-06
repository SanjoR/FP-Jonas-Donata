import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.signal import find_peaks
import sys

T,I = np.genfromtxt("../Data/Data_A.txt", unpack = True, dtype = float)






T = T+273




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
mask3 =  np.logical_and(T<254,T>232)
T_Fit = T[mask3]



def Untergrund(Temp, A,B):
    return(A* np.exp(B*(Temp)))





params_untergrund,cov_ma_untergrund = curve_fit(Untergrund,T_U_R,I_U_R, p0=(0.001,0.0001))


errors_untergrund = np.sqrt(np.diag(cov_ma_untergrund))

for i in range(len(params_untergrund)):
    print(sys.argv[1], params_untergrund[i],errors_untergrund[i])




I_S_Cor = I_S - Untergrund(T_S,*params_untergrund)
I_U_N_Cor = I_U_N -Untergrund(T_U_N,*params_untergrund)
I_U_R_Cor = I_U_R - Untergrund(T_U_R,*params_untergrund)
I_Cor = I - Untergrund(T,*params_untergrund)
I_Cor_Fit = I_Cor[mask3]

def j(T,Amp,W):
    return Amp* np.exp(-W/(const.k *T))


def j(T,Amp,W):
    return Amp+ 1/T * (-W/const.k)

params, cov_ma = curve_fit(j,T_Fit,np.log(I_Cor_Fit*10**-11), p0 = (-25, 0))
errors = np.sqrt(np.diag(cov_ma))

print()
for i in range(len(params)):
    if i == 1:
        print(6.242*10**18*params[i],6.242*10**18*errors[i])  
    else:  
        print(params[i],errors[i])




x_plot = np.linspace(T_U_R.min(),T_U_R.max(),1000)

plt.figure()
plt.plot(T_U_R,I_U_R,"rx", label ="Messdaten Untergrund")
#plt.plot(T_U_N,I_U_N,"kx", label ="Messdaten Untergrund")
plt.plot(T_S,I_S,"bx", label ="Messdaten Signal")
plt.plot(T,I_Cor)
#plt.plot(T_U_R,I_U_R_Cor,"r*", label ="Messdaten Untergrund bereinigt ")
#plt.plot(T_U_N,I_U_N_Cor,"k*", label ="Messdaten Untergrund bereinigt")
#plt.plot(T_S,I_S_Cor,"b*", label ="Messdaten Signal bereinigt")
plt.plot(x_plot,Untergrund(x_plot,*params_untergrund),"r-",label = "Untergrundfit")

plt.grid()
plt.xlabel("Temperatur in K")
plt.ylabel("I in pA")
plt.legend(loc ="best")
plt.show()
plt.close()


x_plot = np.linspace(T_Fit.min(),T_Fit.max(),1000)
plt.figure()
plt.plot(1/T_Fit,np.log(I_Cor_Fit*10**-11), "bx",label="Niedertemperaturbereich")
plt.plot(1/x_plot,j(x_plot,*params),"r-",label = "Fit")
plt.grid()
#plt.yscale("log")
plt.legend(loc="best")
plt.show()
