import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt


Freq_2,Amp_2= np.genfromtxt( "../Bilder/2_Zylinder_mitBlende10mm.dat"   ,unpack = True)
Freq_3, Amp_3= np.genfromtxt("../Bilder/3_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_4, Amp_4= np.genfromtxt("../Bilder/4_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_5, Amp_5= np.genfromtxt("../Bilder/5_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_6, Amp_6= np.genfromtxt("../Bilder/6_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_7, Amp_7= np.genfromtxt("../Bilder/7_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_8, Amp_8= np.genfromtxt("../Bilder/8_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_9, Amp_9= np.genfromtxt("../Bilder/9_Zylinder_mitBlende10mm.dat"  ,unpack = True)
Freq_10,Amp_10= np.genfromtxt("../Bilder/10_Zylinder_mitBlende10mm.dat",unpack = True)
Freq_11,Amp_11= np.genfromtxt("../Bilder/11_Zylinder_mitBlende10mm.dat",unpack = True)
Freq_12,Amp_12= np.genfromtxt("../Bilder/12_Zylinder_mitBlende10mm.dat",unpack = True)

fig,axis = plt.subplots(3,4)
axis[0,0].plot(Freq_2,Amp_2, label="2 Zylinder")
axis[0,0].legend(loc="best")

axis[0,1].plot(Freq_3,Amp_3, label="3 Zylinder")
axis[0,1].legend(loc="best")

axis[0,2].plot(Freq_4,Amp_4, label="4 Zylinder")
axis[0,2].legend(loc="best")

axis[0,3].plot(Freq_5,Amp_5, label="5 Zylinder")
axis[0,3].legend(loc="best")

axis[1,0].plot(Freq_6,Amp_6, label="6 Zylinder")
axis[1,0].legend(loc="best")

axis[1,1].plot(Freq_7,Amp_7, label="7 Zylinder")
axis[1,1].legend(loc="best")

axis[1,2].plot(Freq_8,Amp_8, label="8 Zylinder")
axis[1,2].legend(loc="best")

axis[1,3].plot(Freq_9,Amp_9, label="9 Zylinder")
axis[1,3].legend(loc="best")

axis[2,0].plot(Freq_10,Amp_10, label="10 Zylinder")
axis[2,0].legend(loc="best")

axis[2,1].plot(Freq_11,Amp_11, label="11 Zylinder")
axis[2,1].legend(loc="best")

axis[2,2].plot(Freq_12,Amp_12, label="12 Zylinder")
axis[2,2].legend(loc="best")

plt.show()
plt.close()

