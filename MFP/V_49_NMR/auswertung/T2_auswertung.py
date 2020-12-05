import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

CSV= np.genfromtxt("../Data/scope_1.csv" , delimiter=',')

t = CSV[:,0]
V1 = CSV[:,1]
V2 = CSV[:,2]


V1 = V1[t>0]
V2 = V2[t>0]
t = t [t>0]

V1 = V1[t<2.4]
V2 = V2[t<2.4]
t = t [t<2.4]

def exp_fit(x,amp,m,b):
    return ( amp*np.exp(-m*x) +b )

peaks,_ = find_peaks(V1, 0.2)

print(peaks)

V1_peaks = V1[peaks]
t_peaks = t[peaks]

params,cov_ma = curve_fit(exp_fit,t_peaks,V1_peaks)

x_plot = np.linspace(t_peaks.min(),t_peaks.max(),1000)

error = np.sqrt(np.diag(cov_ma))

m = ufloat(params[1],error[1])
M0 = ufloat(params[0],error[0])
M1 = ufloat(params[2],error[2])
T = 1/m

print("M0: ",M0)
print("M1: ",M1)
print("T2: ",T)

plt.figure()
plt.plot(t,V1,label="V1")
#plt.plot(t,V2,label="V2")
plt.plot(t_peaks,V1_peaks,"rx")
plt.plot(x_plot,exp_fit(x_plot,*params))
plt.legend(loc="best")
plt.show()
