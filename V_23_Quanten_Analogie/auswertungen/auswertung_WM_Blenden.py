import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from uncertainties import ufloat
import numpy as np

Data_10mm = np.genfromtxt('../Bilder/10mmBlendeWasserstoffmolekül.dat',unpack = True)
Data_16mm = np.genfromtxt('../Bilder/16mmBlendeWasserstoffmolekül.dat',unpack = True)

df_10 = pd.DataFrame({"Frequenz" : Data_10mm[0],"Amplitude":Data_10mm[1]})
df_16 = pd.DataFrame({"Frequenz" : Data_16mm[0],"Amplitude":Data_16mm[1]})


def find_resonanz(df,min,max):
   Test_df = df[(df.Frequenz > min) & (df.Frequenz < max)]
   Test_df = Test_df.iloc[np.argmax(Test_df.Amplitude.to_numpy())]
   return([Test_df.Frequenz, Test_df.Amplitude])

Cuts_min = [0,2000,3000,4000,6000,6450,7200]
Cuts_max = [2000,3000,4000,6000,6450,7200,8000]

resonanzen_10 = []
resonanzen_16 = []

Amplitude_10 = []
Amplitude_16 = []

for i in range(len(Cuts_max)):
    resonanzen_10 = np.append(resonanzen_10,find_resonanz(df_10,Cuts_min[i],Cuts_max[i])[0])
    resonanzen_16 = np.append(resonanzen_16,find_resonanz(df_16,Cuts_min[i],Cuts_max[i])[0])
    Amplitude_10 = np.append(Amplitude_10,find_resonanz(df_10,Cuts_min[i],Cuts_max[i])[1])
    Amplitude_16 = np.append(Amplitude_16,find_resonanz(df_16,Cuts_min[i],Cuts_max[i])[1])
    
for i in resonanzen_10:
    print(i)

print()
for i in resonanzen_16:
    print(i)
print()

for i in (resonanzen_16-resonanzen_10):
    print(i)
plt.figure()
plt.plot(Data_10mm[0],Data_10mm[1],"r-", label="10 mm")
plt.plot(Data_16mm[0],Data_16mm[1],"b-" ,label="16 mm")
plt.plot(resonanzen_10,Amplitude_10, "rx")
plt.plot(resonanzen_16,Amplitude_16, "bx")
plt.legend(loc= "best")
#plt.show()
plt.savefig("../latex-template/figure/WM_Blenden.pdf")
plt.close()

