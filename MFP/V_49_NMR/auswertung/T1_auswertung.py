import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Tau,A = np.genfromtxt("../Data/T1_Messung.txt", unpack = True)

print(Tau)
print()

A[Tau>2] = -A[Tau>2] 
print(A)

A *= -1

def exp_fit(x,amp,m,b):
    return ( amp*np.exp(-m*x) +b )


params,cov_matrix = curve_fit(exp_fit,Tau,A)

x_plot = np.linspace(Tau.min(),Tau.max(),1000)

errors = np.sqrt(np.diag(cov_matrix))

m = ufloat(params[1],errors[1])

T1 = 1/m

print(T1)

plt.plot(x_plot,exp_fit(x_plot,*params),"b-")
plt.plot(Tau,A,"rx")
#plt.yscale("log")
#plt.xscale("log")T1 = unp.ufloat(-1/params[1]
plt.show()
