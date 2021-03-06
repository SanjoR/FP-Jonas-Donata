import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

Tau,A = np.genfromtxt("../Data/Diffmessung.txt", unpack = True)


#def M(tau,M0,TD,M1):
 #   return(M0*np.exp(-2*tau/1.4) * np.exp(-tau**3 / TD) +M1)


T2=1.4

#def M(tau,M0,D,Gamma,G):
#    return(M0*np.exp(-2*tau/T2)*np.exp((-2*D*Gamma**2*G**2 *tau**3)/3))

def M(tau,M0,D):
    return(M0*np.exp(-2*tau/T2)*np.exp((-2*D *tau**3)/3))


params,cov_ma = curve_fit(M,Tau,A)


names=["M0","D"]
errors = np.sqrt(np.diag(cov_ma))

print (errors)
par_err = unp.uarray(params,errors)

for i in range(len(par_err)):
    print(names[i],par_err[i])

x_plot= np.linspace(Tau.min(),Tau.max(),1000)


plt.plot(x_plot**3 , np.log(M(x_plot,*params)) -2*x_plot/1.4,label="Fit")
plt.plot(Tau**3 ,np.log(A)-2*Tau/1.4,"rx",label="Messdaten")
plt.legend(loc = "best")
plt.xlabel(r"$\tau^3$ / $s^3$")
plt.ylabel(r"$ln(U) -2\frac{\tau}{T_{2}} $ ")
plt.grid()
#plt.plot(x_plot,M(x_plot,*params))
#plt.plot(Tau,A,"bx")
plt.savefig("../latex-template/figure/Diff_plot.pdf")
plt.show()