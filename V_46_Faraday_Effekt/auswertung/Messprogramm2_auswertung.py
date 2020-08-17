import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as const


Lambda,phi_1,t_1,phi_2,t_2 = np.genfromtxt("../Data/Messprogramm2_Data.txt", unpack =True)
Lambda_Rein,phi_1_Rein,t_1_Rein,phi_2_Rein,t_2_Rein = np.genfromtxt("../Data/Rein_Data.txt", unpack =True)

L_1 = 1.296 *(10 **3)
L_Rein = 5.11 *(10 **3)
#Lambda *= 10**-6
t_1 *= 1/60
t_2 *= 1/60

Theta_1 = phi_1 + t_1
Theta_2 = phi_2 + t_2

Theta = Theta_1 - Theta_2

Theta = np.pi *Theta/180

Theta = Theta/L_1

t_1_Rein *= 1/60
t_2_Rein *= 1/60

Theta_1_Rein = phi_1_Rein + t_1_Rein
Theta_2_Rein = phi_2_Rein + t_2_Rein

Theta_Rein = Theta_1_Rein - Theta_2_Rein

Theta_Rein = np.pi *Theta_Rein/180

Theta_Rein = Theta_Rein/L_Rein

Theta_diff = Theta-Theta_Rein

print(Theta_diff)



plt.figure()
plt.plot(Lambda,Theta,"rx",label="n-dotiert")
plt.plot(Lambda,Theta_Rein,"bx",label="Rein")
plt.legend(loc="best")
plt.show()
plt.close()


def linear(x,a):
    return(a*x)


params,cov_matrix = curve_fit(linear,Lambda**2, Theta_diff,p0=(29))

x_plot=np.linspace(Lambda.min()**2, Lambda.max()**2,50)

plt.plot(Lambda**2, Theta_diff,"kx",label="Differenz")
plt.plot(x_plot,linear(x_plot,*params),"b-",label="Fit")
plt.legend(loc="best")
plt.show()
plt.close() 
N=2.8*10**24
B=0.41
n=3.57
error = np.sqrt(np.diag(cov_matrix))
a = ufloat(params[0],error[0]) 
print(a)
a *= 1/(10**-18)
m_eff = unp.sqrt(const.e**3 /(8*np.pi**2 *const.epsilon_0 *const.c**3) *1/a *N*B/n) 
print(m_eff)

