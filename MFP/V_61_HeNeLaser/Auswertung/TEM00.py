import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const


L,I = np.genfromtxt("../Data/TEM00.txt", unpack = True)

L = L-L[I== I.max()]
for i in I:
    print(i)
def gaussian(x,A, mu, sig):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
print()
params,covma = curve_fit(gaussian,L,I)
errors=np.sqrt(np.diag(covma))
print("mu:",ufloat(params[0],errors[0]))
print("sigma:",ufloat(params[1],errors[1]))
print("A:",ufloat(params[2],errors[2]) )


x_plot=np.linspace(L.min(),L.max(),1000)

plt.figure()
plt.plot(L,I,"rx",label="Messdaten")
plt.plot(x_plot,gaussian(x_plot,*params),"b-",label="Gaußfit")
plt.xlabel("Abstand / mm")
plt.ylabel("Intensität")
plt.grid()
plt.legend(loc="best")
plt.savefig("../latex-template/figure/TEM00.pdf")
plt.show()
