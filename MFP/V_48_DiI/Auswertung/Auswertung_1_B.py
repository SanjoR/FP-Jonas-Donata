import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.signal import find_peaks
import sys




    
T,I = np.genfromtxt("../Data/Data_B.txt", unpack = True)


T = T+273



mask = np.logical_or(T<236, T>267)
T_U = T[mask]
T_S = T[~mask]
I_U = I[mask]
I_S = I[~mask]

mask3 = T<251
T_Fit = T[mask3]



def Untergrund(Temp, A,B):
    return(A* np.exp(B*(Temp)))





params_untergrund,cov_ma_untergrund = curve_fit(Untergrund,T_U,I_U, p0=(0.001,0.0001))


errors_untergrund = np.sqrt(np.diag(cov_ma_untergrund))

par_err_unter = unp.uarray(params_untergrund, errors_untergrund)

print("Untergrund : A,B")
for i in par_err_unter:
    print(i)
print()




I_S_Cor = I_S - Untergrund(T_S,*params_untergrund)
I_U_Cor = I_U -Untergrund(T_U,*params_untergrund)

I_Cor = I - Untergrund(T,*params_untergrund)
I_Cor_Fit = I_Cor[mask3]





def j(T,Amp,W):
    return Amp+ 1/T * (-W/const.k)

params, cov_ma = curve_fit(j,T_Fit,np.log(I_Cor_Fit*10**-11), p0 = (-25, 0))
errors = np.sqrt(np.diag(cov_ma))

par_err = unp.uarray(params, errors)
#par_err[1]*= 6.242*10**18

print("Tfit:",T_Fit.min(),T_Fit.max())
print("Params: Amp,W")
for i in par_err:
    print(i)

print(par_err[1]*6.242*10**18)


x_plot = np.linspace(T_U.min(),T_U.max(),1000)

plt.figure()
#plt.plot(T,I,"bx")
plt.plot(T_U,I_U,"rx", label ="Messdaten Untergrund")
plt.plot(T_S,I_S,"bx", label ="Messdaten Signal")
#plt.plot(T,I_Cor)
#plt.plot(T_U,I_U_Cor,"r*", label ="Messdaten Untergrund bereinigt ")
#plt.plot(T_S,I_S_Cor,"b*", label ="Messdaten Signal bereinigt")
#plt.plot(x_plot,Untergrund(x_plot,*params_untergrund),"r-",label = "Untergrundfit")

plt.grid()
plt.xlabel("Temperatur / K")
plt.ylabel(r"I / A$\times 10^{-11}$")
plt.legend(loc ="best")
plt.savefig("../latex-template/figure/Untergrundfit_B.pdf")
#plt.savefig("../latex-template/figure/Messdate_rein_B.pdf")
plt.show()
plt.close()


x_plot = np.linspace(T_Fit.min(),T_Fit.max(),1000)
plt.figure()
plt.plot(1/T_Fit,np.log(I_Cor_Fit*10**-11), "bx",label="Niedertemperaturbereich")
plt.plot(1/x_plot,j(x_plot,*params),"r-",label = "Ausgleichsgrade")
plt.grid()
##plt.yscale("log")
plt.xlabel(r"inverse  Temperatur / $K^{-1}$")
plt.ylabel(r"ln(I) / ln(A$\times 10^{-11})$")
plt.legend(loc="best")
plt.savefig("../latex-template/figure/LinFit_W_B.pdf")
plt.show()
plt.close()

#benötigt I_Cor und T

mask4 = np.logical_and(T<270,T>230)

T_int = T[mask4]
I_Cor_int = I_Cor[mask4] * 10**-11
T_not_int = T[~mask4]
I_Cor_not_int = I_Cor[~mask4] * 10 **-11

index = np.argwhere(I_Cor_int == I_Cor_int.max())

plt.figure()
plt.plot(T_int,I_Cor_int,"bx")
plt.plot(T_int[index],I_Cor_int[index],"k+")
plt.plot(T_not_int,I_Cor_not_int,"rx")
plt.show()
plt.close()



def Num_int(Temp):
    integral= 0
    for i in range(len(T_int)):
        if i !=0:
            if T_int[i]> Temp:
                integral += (T_int[i]-T_int[i-1]) * (I_Cor_int[i-1]+I_Cor_int[i])/2
                #print(integral)
    return(integral)


F = []

def lin_fit(T,A,B):
    return((A/const.k)*1/T  + B)

for i in range(len(T_int)):
    F = np.append(F,Num_int(T_int[i])/I_Cor_int[i])


while 0.0 in F:
    F = F[:-1]
    T_int=T_int[:-1]

params_int,cov_ma_int = curve_fit(lin_fit,T_int,np.log(F))


errors_int = np.sqrt(np.diag(cov_ma_int))

par_int_err = unp.uarray(params_int, errors_int)
#par_int_err[0]*=6.242*10**18

print()

print("Params int : W,B")
print(par_int_err[0]*6.242*10**18)
for i in par_int_err:

    print(i)

x_plot= np.linspace(T_int.min(),T_int.max(),1000)

plt.figure()
plt.plot(1/T_int,np.log(F),"bx",label="Messdaten")
plt.plot(1/x_plot,lin_fit(x_plot,*params_int), "r-",label="Ausgleichsgrade")
plt.grid()
plt.xlabel(r"inverse  Temperatur / $K^{-1}$")   
plt.ylabel(r"ln(I) / ln(A$\times 10^{-11}$)")
plt.legend(loc="best")
plt.savefig("../latex-template/figure/Integralverfahren_B.pdf")
plt.show()

b = 1 #Kelvin pro minute

b *= 1/60

def tau_0(W):
    return( const.k*T_int[index]**2 / (W*b) * unp.exp(-W/(const.k*T_int[index])) )

print()
print("Approx verfahren: ", tau_0(par_err[1]/ 6.242*10**18))
print()
print("Integrations verfahren: ", tau_0(par_int_err[0]/ 6.242*10**18))
