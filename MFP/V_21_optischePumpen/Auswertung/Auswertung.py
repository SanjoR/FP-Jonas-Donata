import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.constants as const
import sys


Freq,Peak1_sweep, Peak2_sweep, Peak1_hori,Peak2_hori = np.genfromtxt("../Data/Data.txt", unpack =True)
Freq*=1000
Peak1_hori += -13.8
Peak2_hori += -13.8

I_1_sweep = Peak1_sweep * 0.1
I_2_sweep = Peak2_sweep * 0.1

I_1_hori = Peak1_hori *0.3
I_2_hori = Peak2_hori *0.3

R_sweep = 0.1639
N_sweep = 11

R_hori = 0.1579
N_hori = 154

B_1 = const.mu_0 * 8 * I_1_sweep *N_sweep/(np.sqrt(125)*R_sweep) + const.mu_0 * 8 * I_1_hori *N_hori/(np.sqrt(125)*R_hori)
B_2 = const.mu_0 * 8 * I_2_sweep *N_sweep/(np.sqrt(125)*R_sweep) + const.mu_0 * 8 * I_2_hori *N_hori/(np.sqrt(125)*R_hori)

for i in B_2:
    print(round(i*10**6,1))
print()
def linear_fit(x,a,b):
    return(a*x +b)

params_1,covma_1=curve_fit(linear_fit,Freq,B_1)
params_2,covma_2=curve_fit(linear_fit,Freq,B_2)

errors_1 = np.sqrt(np.diag(covma_1))
errors_2 = np.sqrt(np.diag(covma_2))





params_err_1 = unp.uarray(params_1,errors_1)
params_err_2 = unp.uarray(params_2,errors_2)

print("params_1")
for i in params_err_1:
    print(i)
print()
print("params_2")
for i in params_err_2:
    print(i)

x_plot= np.linspace(Freq.min(),Freq.max(),100)
print()

plt.figure()
plt.plot(Freq/1000,B_1*10**6,"rx",label="Peak 1 Messdaten")
plt.plot(Freq/1000,B_2*10**6,"bx",label="Peak 2 Messdaten")
plt.plot(x_plot/1000,linear_fit(x_plot,*params_1)*10**6,"r-",label= "Fit 1")
plt.plot(x_plot/1000,linear_fit(x_plot,*params_2)*10**6,"r-",label= "Fit 2")
plt.xlabel(r"f / kHz")
plt.ylabel(r"B / µT")
plt.grid()
plt.legend(loc="best")
plt.savefig("../latex-template/figure/Messdaten_Fit.pdf")
#plt.show()

g_F_1 = const.h/(const.e *const.hbar /(2*const.m_e))*1/params_err_1[0]
g_F_2 = const.h/(const.e *const.hbar /(2*const.m_e))*1/params_err_2[0]
print()
print("g_F_1: ", g_F_1)
print("g_F_2: ", g_F_2)
print()
print("I_1: ",0.5*(2.0023/g_F_1 -1))
print("I_2: ",0.5*(2.0023/g_F_2 -1))