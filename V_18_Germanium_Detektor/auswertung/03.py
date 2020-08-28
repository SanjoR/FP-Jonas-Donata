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

A = np.genfromtxt("../Data/03.txt", unpack=True)
E_125_Sb, I_125_Sb = np.genfromtxt("../Data/Sb_125.txt", unpack = True)
E_Ba_133, I_Ba_133 = np.genfromtxt("../Data/Ba_133.txt", unpack = True)






Kanal = np.arange(0,len(A))


m = ufloat(0.40312,0.00009)
b = ufloat(-2.72,0.18)


def linear_value(x):
    return(0.40312*x -2.72)


def linear(x):
    return(m*x +b)

E = linear_value(Kanal)


peaks,_ = find_peaks(A, height = 143)
peaks = peaks[1:]

plt.figure(figsize = (13,8))
plt.plot(E[E<1000],A[E<1000],"b-",label = r"Spektrum")
plt.plot(E[peaks],A[peaks],"rx")
for i in range(len(E_Ba_133)):
    if i == 0:
        plt.plot([E_125_Sb[i],E_125_Sb[i]], [0,A.max()], "k--",label="Antimon-125")
        plt.plot([E_Ba_133[i],E_Ba_133[i]],[0,A.max()], "r--" ,label="Barium-133")
    else:
        plt.plot([E_125_Sb[i],E_125_Sb[i]], [0,A.max()], "k--")
        plt.plot([E_Ba_133[i],E_Ba_133[i]],[0,A.max()], "r--" )
plt.legend(loc="best")
plt.xlabel("E / keV")
plt.ylabel("Zählrate")
plt.grid()
plt.savefig("../latex-template/figure/03_peaks.pdf")
plt.show()
plt.close()

E_fehl = linear(Kanal[peaks])
print("E_fehl")
for i in E_fehl:
    print(i)
print()





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

K_plot = Kanal[peaks]
Peaknummer = np.array([0,1,2,3,4,7])
N = []
K_plot_2 = []
fig,axs = plt.subplots(2,3,figsize = (16,13))
n=0
m=0
for i in range(len(K_plot)):
    c_0 = K_plot[i]
    a,b=get_PeakArrays(K_plot[i])
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
    axs[n,m].plot(a,b,"rx", label = f"Peaknummer {Peaknummer[i]}")
    axs[n,m].grid()
    axs[n,m].legend(loc="best")
    m+=1
    if m > 2:
        m = 0 
        n+=1
#axs[1,2].set_visible(False)
plt.savefig("../latex-template/figure/03_subplot.pdf")
plt.show()
plt.close()

uparams_array = unp.uarray(params_array, error_array)


print("Amp fit")
for i in uparams_array[:,0]:
    print(i)
print()
print("k fit")
for i in uparams_array[:,1]:
    print(i)
print()
print("b fit")
for i in uparams_array[:,2]:
    print(i)
print()

uparams_array = np.delete(uparams_array,1,0)

N_err = uparams_array[:,0]*(np.pi/uparams_array[:,1])

print("N")
for i in N_err:
    print(i)

E_Ba_133, I_Ba_133 = zip(*sorted(zip(E_Ba_133,I_Ba_133)))
print()
print("E_lit")
for i in E_Ba_133:
    print(i)
print()
print("I_lit")
for i in I_Ba_133:
    print(i)

Omega = 0.23467106406199312
t = 3770.3771

for i in range(len(peaks)):
    if i == 0:
        Akti = N[i] *4 *np.pi /(eta_fit(E[peaks][i]) * t * I_Ba_133[i] * Omega)
    else:
        Akti = np.append(Akti, N[i] *4 *np.pi /(eta_fit(E[peaks][i]) * t * I_Ba_133[i] * Omega))


print()
for i in Akti:
    print(i)

    
Akti = np.delete(Akti,1)
print()
for i in Akti:
    print(i)

print()
print(sum(Akti)/len(Akti))