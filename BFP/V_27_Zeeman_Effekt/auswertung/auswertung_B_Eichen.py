import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import numpy as np
import scipy.constants as const

I,B = np.genfromtxt("../Data/Data_B_Eichen.txt", unpack = True)


def B_berech(n):
    if n == 1:
        lam = 643.8 *10 **-9
        del_Lam = 0.0489 *10 **-9
        del_m = 1
    elif n==2 :
        lam = 480 *10**-9
        del_Lam =  0.02695 *10**-9
        del_m = 1.5*0 + 2*1
    X = const.hbar * 2 *np.pi * const.c
    return((X *del_Lam) / (lam **2 * del_m *  9.274 *10 **-24 * 4 ))

print(B_berech(1))

def linear(x,a,b):
    return(a*x +b)


params,covar_matrix = curve_fit(linear, I,B)

error = np.sqrt(np.diag(covar_matrix))

a= ufloat(params[0], error[0])
b= ufloat(params[1], error[1])
print("B")
print(a*5.02 +b)
print(a*3.96 +b)

print("params")
print("m",a)
print("b",b)
def Umrechner(x):
    return( (x-b)/a )

print(Umrechner(B_berech(2) *1000 ))
I_plot = np.linspace(np.min(I),np.max(I), 100)

plt.figure()
plt.plot(I,B,"rx",label="Messdaten")
plt.plot(I_plot, linear(I_plot,*params) , "b-",label="Ausgleichsgrade")
plt.xlabel("I / A")
plt.ylabel("B / mT")
plt.grid()
plt.legend(loc="best")
plt.savefig("../latex-template/figure/B_plot.pdf")
plt.show()
plt.close() 