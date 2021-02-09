import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

Leistung = np.genfromtxt("../Data/Polarisation.txt", unpack = True)
Winkel = np.arange(0,190,10)*np.pi/180
def sin2(x,A,B,C,D):
    return(A*np.sin(B*(x+C))**2 + D)

params,covma = curve_fit(sin2,Winkel,Leistung)
errors = np.sqrt(np.diag(covma))

params_err = unp.uarray(params,errors)

for i in params_err:
    print(i)
x_plot=np.linspace(0,np.pi,1000)

plt.figure()
plt.plot(Winkel,Leistung,"rx",label="Messdaten")
plt.xlabel(r"$\varphi$ / rad")
plt.ylabel("Intensität / mW")
plt.plot(x_plot,sin2(x_plot,*params),"b-",label = "Fitfunktion")
plt.grid()
plt.legend(loc = "best")
#plt.savefig("../latex-template/figure/Polarisation.pdf")
#plt.show()

print(180/np.pi *x_plot[sin2(x_plot,*params)==sin2(x_plot,*params).max()])