import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const


L,I = np.genfromtxt("../Data/TEM01.txt", unpack = True)

L = L-L[I== I.min()]
for i in I:
    print(i)
L_min= L[L<0]
I_min= I[L<0]
L_max=L[L>=0]
I_max=I[L>=0]


def gaussian(x, mu, sig,A):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))







params_min,covma_min=curve_fit(gaussian,L_min,I_min,p0=(-8,6,0.5))
params_max,covma_max=curve_fit(gaussian,L_max,I_max,p0=(8,6,0.5))


errors_min = np.sqrt(np.diag(covma_min))
errors_max = np.sqrt(np.diag(covma_max))

x_plot = np.linspace(L.min(),L.max(),1000)

plt.figure()
plt.plot(L,I,"rx",label="Messdaten")
plt.plot(x_plot,gaussian(x_plot,*params_min),"g--",label="Gaußfit linkes Maximum")
plt.plot(x_plot,gaussian(x_plot,*params_max),"b--",label="Gaußfit rechtes Maximum")
plt.plot(x_plot,gaussian(x_plot,*params_max)+gaussian(x_plot,*params_min),"k-",label="Gaußfit Superposition")
#plt.plot(x_plot,gaus_2(x_plot,*params))
#
#
plt.xlabel("Abstand / mm")
plt.ylabel("Intensität")
plt.grid()
plt.legend(loc="best")
plt.savefig("../latex-template/figure/TEM01.pdf")
plt.show()
print()
print("params_min")
for i in range(len(params_max)):
    print(ufloat(params_min[i],errors_min[i]))

print()
print("params_max")
for i in range(len(params_max)):
    print(ufloat(params_max[i],errors_max[i]))