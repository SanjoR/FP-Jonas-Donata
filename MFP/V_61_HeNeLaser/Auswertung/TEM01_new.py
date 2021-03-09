import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.special import eval_hermite as H_n


L,I = np.genfromtxt("../Data/TEM01.txt", unpack = True)

L = L-L[I== I.min()]


def I_vert(x,A,w):
    return(A/(w**2)*H_n(1,np.sqrt(2)*x/w)**2 *np.exp(-x**2/w**2)**2)


params,cov_ma = curve_fit(I_vert,L,I)

error = np.sqrt(np.diag(cov_ma))

par_err = unp.uarray(params,error)

print("A,w:")
for i in par_err:
    print(i)

x_plot= np.linspace(L.min(),L.max(),1000)

plt.figure()
plt.xlabel("Abstand / mm")
plt.ylabel("Intensität")
plt.plot(L,I,"rx",label = "Messdaten")
plt.plot(x_plot,I_vert(x_plot,*params),"b-",label = "Fitfunktion")
plt.grid()
plt.legend()
plt.savefig("../latex-template/figure/TEM01_new.pdf")
plt.show()
