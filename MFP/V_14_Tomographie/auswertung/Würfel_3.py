import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
s2= np.sqrt(2)

A = np.matrix([[0,s2,0,s2,0,0,0,0,0],
               [0,0,s2,0,s2,0,s2,0,0],
               [0,0,0,0,0,s2,0,s2,0],
               [1,1,1,0,0,0,0,0,0],
               [0,0,0,1,1,1,0,0,0],
               [0,0,0,0,0,0,1,1,1],
               [0,s2,0,0,0,s2,0,0,0],
               [s2,0,0,0,s2,0,0,0,s2],
               [0,0,0,s2,0,0,0,s2,0],
               [0,0,1,0,0,1,0,0,1],
               [0,1,0,0,1,0,0,1,0],
               [1,0,0,1,0,0,1,0,0]
               ])
I_01= 10396/126.44
I_02= 14398/117.9
I_03= 12927/119.6

I1_s= 300/126.44
I2_s= 247/117.9
I3_s= 271/119.6


I_01_u=ufloat(I_01,I1_s)
I_02_u=ufloat(I_02,I2_s)
I_03_u=ufloat(I_03,I3_s)

I_0 = np.matrix([I_01,I_02,I_01,I_03,I_03,I_03,I_01,I_02,I_01,I_03,I_03,I_03])

I_0_s = np.array([I1_s,I2_s,I1_s,I3_s,I3_s,I3_s,I1_s,I2_s,I1_s,I3_s,I3_s,I3_s])

I_1 = 22765 /208.98
I_2 = 19747 / 210.88
I_3 = 19147 / 178.68
I_1s = (246/208.98)
I_2s = (237/ 210.88)
I_3s = (225/ 178.68)

I = np.matrix([I_1,I_2,I_1,I_3,I_3,I_3,I_1,I_2,I_1,I_3,I_3,I_3])


I_s = np.array([I_1s,I_2s,I_1s,I_3s,I_3s,I_3s,I_1s,I_2s,I_1s,I_3s,I_3s,I_3s])





I_1_u = ufloat(I_1,I_1s)
I_2_u = ufloat(I_2,I_2s)
I_3_u = ufloat(I_3,I_3s)



print("ND",I_1_u)   
print("D",I_2_u)
print("SG",I_3_u)


print()
print(unp.log(I_01_u/I_1_u)/(2*np.sqrt(2)) )
print(unp.log(I_02_u/I_2_u)/(3*np.sqrt(2)) )
print(unp.log(I_03_u/I_3_u)/(3) )
print()
print((#(unp.log(I_01_u/I_1_u)/(2*np.sqrt(2)) 
+unp.log(I_02_u/I_2_u)/(3*np.sqrt(2)) 
+unp.log(I_03_u/I_3_u)/(3) )/2)


mu = (+unp.log(I_02_u/I_2_u)/(3*np.sqrt(2)) 
        +unp.log(I_03_u/I_3_u)/(3) )/2
mu = mu.n
mu_lit =np.array([0.211
                 ,1.419
                 ,0.606
                 ,0.638
                 ,0.121])

rel_Ab = np.abs(mu - mu_lit)/mu
print()
for i in rel_Ab:
    print(round(i*100,1))
Material = np.array(["Al","Pb","Fe","Messing","Delrin"])
print()
print(Material[rel_Ab==rel_Ab.min()])