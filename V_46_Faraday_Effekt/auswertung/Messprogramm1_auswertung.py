import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as const


plt.rcParams.update({'font.size': 17.5})

Lambda,phi_1,t_1,phi_2,t_2 = np.genfromtxt("../Data/Messprogramm1_Data.txt", unpack =True)
Lambda_Rein,phi_1_Rein,t_1_Rein,phi_2_Rein,t_2_Rein = np.genfromtxt("../Data/Rein_Data.txt", unpack =True)

L_1 = 1.36 *(10 **3)
L_Rein = 5.11 *(10 **3)
#Lambda *= 10**-6
t_1 *= 1/60
t_2 *= 1/60

Theta_1 = phi_1 + t_1
Theta_2 = phi_2 + t_2

print("Theta1")
for i in Theta_1:
    print(round(i,2))
print()
print("Theta2")
for i in Theta_2:
    print(round(i,2))
print()

Theta = (Theta_1 - Theta_2)/2

print("Theta")
for i in Theta:
    print(round(i,2))
print()
Theta = np.pi *Theta/180

Theta = Theta/L_1


print("Theta")
for i in Theta:
    print(round(10000*i,2))
print()

t_1_Rein *= 1/60
t_2_Rein *= 1/60

Theta_1_Rein = phi_1_Rein + t_1_Rein
Theta_2_Rein = phi_2_Rein + t_2_Rein

Theta_Rein = (Theta_1_Rein - Theta_2_Rein)/2

Theta_Rein = np.pi *Theta_Rein/180

Theta_Rein = Theta_Rein/L_Rein


print("Theta_rein")
for i in Theta_Rein:
    print(i)
print()

Theta_diff = Theta-Theta_Rein

for i in Lambda:
    print(round(i**2,1))
print()
print("Diff")
for i in Theta_diff:
    print(round(10000*i, 2))



plt.figure(figsize=(13,8))
plt.plot(Lambda,Theta,"rx",label="n-dotiert")
plt.plot(Lambda,Theta_Rein,"bx",label="Rein")
plt.xlabel(r"$\lambda$ / $µm$")
plt.ylabel(r"$\theta_{norm}\, /\, \frac{rad}{µm}$")
plt.legend(loc="best")
plt.savefig("../latex-template/figure/Theta1_plot.pdf")
plt.show()
plt.close()


def linear(x,a):
    return(a*x)


params,cov_matrix = curve_fit(linear,Lambda**2, Theta_diff,p0=(29))

x_plot=np.linspace(Lambda.min()**2, Lambda.max()**2,50)

plt.figure(figsize=(13,8))
plt.plot(Lambda**2, Theta_diff,"kx",label="Differenz")
plt.plot(x_plot,linear(x_plot,*params),"b-",label="Fit")
plt.ylabel(r"$\theta_{frei}\, /\, \frac{rad}{µm}$")
plt.xlabel(r"$\lambda^2$ / $µm^2$")
plt.legend(loc="best")
plt.savefig("../latex-template/figure/Theta1_diff_plot.pdf")
plt.show()
plt.close() 
N=1.2*10**24
B=0.41
n=3.35
error = np.sqrt(np.diag(cov_matrix))
a = ufloat(params[0],error[0]) 

a *= 1/(10**-18)
print(a)
m_eff = unp.sqrt(const.e**3 /(8*np.pi**2 *const.epsilon_0 *const.c**3) *1/a *N*B/n) 

print(m_eff)

lit = 6.103*10**-32
print((m_eff-lit)/lit)
print((m_eff + ufloat(7.12,22)*10**-32)/2)