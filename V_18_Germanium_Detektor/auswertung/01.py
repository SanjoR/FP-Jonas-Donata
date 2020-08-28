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


params_1,cov_matrix_1 = curve_fit(linear,K_plot,P)

error_1 = np.sqrt(np.diag(cov_matrix_1))
uparams_1 = unp.uarray(params_1, error_1)
n=0
print("E(kanal)=m * kanal + b")
for i in uparams_1:
    if n == 0:
        print("m ",i)
    else:
        print("b ",i)
    n+=1
print()

A_plot = []
for i in K_plot:
    A_plot = np.append(A_plot,A[Kanal==i])

print("Peaks und Intensitäten bei Data")
print()
plt.figure(figsize=(13,5))
plt.plot(Kanal,A,label= "Messdaten")
plt.plot(K_plot,A_plot, "rx",label="Peaks")
plt.grid()
plt.xlabel("Kanal")
plt.ylabel("Zählrate")
plt.legend(loc = "best")
plt.savefig("../latex-template/figure/Peaks_01.pdf")
#plt.show()
plt.close()

plt.figure(figsize=(13,5))
plt.plot(K_plot,P, "rx",label ="Peaks")
plt.plot(K_plot,linear(K_plot,*params_1),label = "Linearer Fit")
plt.grid()
plt.xlabel("Kanal")
plt.ylabel("E / keV")
plt.legend(loc = "best")
plt.savefig("../latex-template/figure/Lin_Fit_01.pdf")
#plt.show()
plt.close()





lam = np.log(2)/13.516

A_0 = ufloat(4130,60)


N_0 = A_0/lam

def Akti(t):                    ## t in Jahren
    return(lam * N_0 *np.exp(-lam * t ))


T = 19 + (1/12)

print("Aktivität am Messtag")
print(Akti(T))
print()
t = 4109.4111 #/(60*60*24*365)
r = 22.5 
d = 80
Omega = 2*np.pi*(1-np.sqrt(1/((r/d)**2 +1 )))
print("Omega")
print(Omega)
print()
def get_PeakArrays(k):
    K_r = Kanal[Kanal>= k-20]
    K_r = K_r[K_r<=k+20]
    A_r = []
    for i in K_r:
        A_r=np.append(A_r,A[Kanal == i])
    return(K_r,A_r)

def potenz(c,Amp, k,b):
    return(Amp*np.exp(-k*(c-c_0)**2) +b)

eta = []
K_plot_2 = []
fig,axs = plt.subplots(4,3,figsize = (16,13))
n=0
m=0
for i in range(len(K_plot)):
    if i != 3:
        c_0 = K_plot[i]
        a,b=get_PeakArrays(K_plot[i])
        params,cov_matrix = curve_fit(potenz,a,b)
        if i == 0:
            params_array = [params]
            error_array = [np.sqrt(np.diag(cov_matrix))]
        else:
            params_array = np.append(params_array , [params], axis = 0)
            error_array = np.append(error_array, [np.sqrt(np.diag(cov_matrix))], axis =0)
        N = params[0]*np.sqrt(np.pi/params[1])
        eta = np.append(eta,N*4*np.pi /(Omega *I[i] * t *Akti(T)))
        K_plot_2 = np.append(K_plot_2,K_plot[i])
        c_0 = K_plot[i]
        a,b=get_PeakArrays(K_plot[i])
        x_plot =np.linspace(a.min(),a.max(),1000)
    
        
        axs[m,n].plot(x_plot,potenz(x_plot,*params),"b-")
        axs[m,n].plot(a,b,"rx", label = f"Peaknummer {i}")
        axs[m,n].grid()
        axs[m,n].legend(loc="best")
        n+=1
        if n > 2:
            n = 0 
            m+=1

#axs[2,2].set_visible(False)
axs[3,1].set_visible(False)
axs[3,2].set_visible(False)
for ax in axs.flat:
    ax.set(xlabel='Kanal', ylabel='Zählrate')

plt.savefig("../latex-template/figure/Subplot_01.pdf")
plt.show()
plt.close()







Amp_array = unp.uarray(np.around(params_array[:,0],0), np.around(error_array[:,0],0))
k_array = unp.uarray(np.around(params_array[:,1],3), np.around(error_array[:,1],3))
b_array = unp.uarray(np.around(params_array[:,2],0), np.around(error_array[:,2],0))


N_array = Amp_array*unp.sqrt(np.pi/k_array)
print()
print("N")
for i in N_array:
    print(i)

print("Amp")
for i in Amp_array:
    print(i)
print()
print("k")
for i in k_array:
    print(i)
print()
print("b")
for i in b_array:
    print(i)
print()



E = linear(K_plot_2,*uparams_1)
E_n = []
E_s = []
print("E")
for i in range(len(E)) :
    print(E[i])
    E_n=np.append(E_n,E[i].n)
    E_s=np.append(E_s,E[i].s)

eta_n = []
eta_s = []
print()
print("eta")
for i in range(len(eta)) :
    print(eta[i])
    eta_n=np.append(eta_n,eta[i].n)
    eta_s=np.append(eta_s,eta[i].s)


def eta_fit(e,Amp,z):
    return(Amp*e**z)

params_2,cov_matrix_2 = curve_fit(eta_fit,E_n,eta_n)

print("params")
for i in range(2):
    print(params_2[i], np.sqrt(np.diag(cov_matrix_2))[i] )






E_plot = np.linspace(E_n.min(),E_n.max(),1000)

plt.figure()
plt.plot(E_plot,eta_fit(E_plot,*params_2),"b-",label = "Fit")
plt.errorbar( E_n, eta_n , xerr=E_s,yerr=eta_s,  fmt="rx", label ="Messwerte")
plt.xlabel("E / keV")
plt.grid()
plt.ylabel(r"Vollenergienachweiswahrscheinlichkeit $\eta$")
plt.legend(loc="best")
plt.savefig("../latex-template/figure/Vollenergienachweiswahrscheinlichkeit.pdf")
plt.show()
plt.close()