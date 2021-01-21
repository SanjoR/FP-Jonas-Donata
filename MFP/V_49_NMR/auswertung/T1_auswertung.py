import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Tau,A = np.genfromtxt("../Data/T1_Messung.txt", unpack = True)

print(Tau)
print()
#
A[Tau>2] = -A[Tau>2] 



#A *= -1

for i in A :
    print(i)

def exp_fit(tau,M0,T1):
    return (M0* (1-2*np.exp(-tau/T1))  )


params,cov_matrix = curve_fit(exp_fit,Tau,A)

x_plot = np.linspace(Tau.min(),Tau.max(),1000)

errors = np.sqrt(np.diag(cov_matrix))

M0 = ufloat(params[0], errors[0])
T1 = ufloat(params[1], errors[1])


#T1 = 1/m
#print("M0: ", amp )
#print("M1: ", b)
#print ("M0 = -2M1: ",-2*b)
print("M0:", M0)
print("T1:", T1)

plt.plot(x_plot,exp_fit(x_plot,*params),"b-", label = "Fit")
plt.plot(Tau,A,"rx" ,label = "Messdaten")
plt.xlabel(r"$\tau$ / s")
plt.ylabel(r"U / V ")
plt.legend(loc = "best")
plt.grid()
#plt.yscale("log")
#plt.xscale("log")T1 = unp.ufloat(-1/params[1]
plt.savefig("../latex-template/figure/T1_fit.pdf")
plt.show()
