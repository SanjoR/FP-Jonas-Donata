import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

def tau(T,tau_V,W_V):
    return(tau_V*unp.exp(W_V/(const.k * T)))

W_Approx_A = ufloat(1.383,0.023)*10**-19
W_Approx_B = ufloat(1.57,0.11)*10**-19
W_Integr_A = ufloat(1.56,0.05)*10**-19
W_Integr_B = ufloat(1.59,0.08)*10**-19
#Tau_Approx_A = ufloat(,)
#Tau_Approx_B = ufloat(,)
#Tau_Integr_A = ufloat(,)
#Tau_Integr_B = ufloat(,)


W_A = (W_Approx_A + W_Integr_A)/2
W_B = (W_Approx_B + W_Integr_B)/2

T_max_A = 258.2
T_max_B = 251.5

b_A = 2/60
b_B = 1/60

def tau_0(W,T_max,b):
    return( const.k*T_max**2 / (W*b) * unp.exp(-W/(const.k*T_max))) 

tau_0_A = tau_0(W_A,T_max_A,b_A)
tau_0_B = tau_0(W_B,T_max_B,b_B)


print("W_A: ",W_A)
print("W_B: ",W_B)
print()
print("tau_0_A: ",tau_0_A)
print("tau_0_B: ",tau_0_B)

tau_0_mit = (tau_0_A+tau_0_B)/2
W = (W_A+W_B)/2
print()
print("W: ",W)
print("tau_0_mit: ",tau_0_mit)
T_plot = np.linspace(220,250,1000)

W_lit = 660 /(6.242*10**18)
tau_lit = 4*(10**-14)

print()
print(W_lit)
print(tau_lit)
tau_plot = tau(T_plot,tau_0_mit,W)

for i in range(len(T_plot)):
    print(T_plot[len(T_plot)-1 -i])
    tau(T_plot[len(T_plot)-1 -i],tau_lit,W_lit)
tau_lit_plot = tau(T_plot,tau_lit,W_lit)
tau_plot_lit_n =[]
for i in tau_lit_plot:
    tau_plot_lit_n = np.append(tau_plot_lit_n,i.n)

tau_value = []
tau_std = []
for i in tau_plot:
    tau_value = np.append(tau_value,i.n)
    tau_std = np.append(tau_std,i.s)
plt.figure()
plt.plot(T_plot,tau_value,"r-",label=r"$\tau$-Verlauf mit gemittelten Werten")
plt.plot(T_plot,tau_lit_plot_n,"b-",label="Theorieverlauf")
#plt.fill(T_plot,tau_value-tau_std,tau_value+tau_std,alpha = 0.5)
plt.xlabel("Temperatur / K")
plt.ylabel(r"$\tau(T)$ / s")
plt.grid()
plt.legend(loc = "best")
#plt.savefig("../latex-template/figure/Tau_verlauf.pdf")
plt.show()