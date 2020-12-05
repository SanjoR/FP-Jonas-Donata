import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt


Freq_2,Amp_2= np.genfromtxt( "../Bilder/Austausch_75mmZylinder.dat"   ,unpack = True)
Freq_10,Amp_10= np.genfromtxt("../Bilder/10_Zylinder_mitBlende13mm.dat",unpack = True)

plt.plot(Freq_2,Amp_2,label="Austausch")
plt.plot(Freq_10,Amp_10,alpha=0.5, label ="Normal")
plt.legend(loc="best")
plt.savefig("../latex-template/figure/Austausch_75mm.pdf")
plt.show()
plt.close()