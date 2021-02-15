import numpy as np
from uncertainties import ufloat

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

I_0_v = 52730/ 300
I_0_s = 319 / 300
print(ufloat(I_0_v,I_0_s))

I1_v = 550/111.44
I2_v = 647/114.70
I3_v = 1449/123.44
I1_s = 1/(168/111.44)
I2_s = 1/(152/114.70)
I3_s =1/ (238/123.44)

I = np.matrix([I1_v,I2_v,I1_v,I3_v,I3_v,I3_v,I1_v,I2_v,I1_v,I3_v,I3_v,I3_v])
V_u = np.eye(12)
I_s = np.array([I1_s,I2_s,I1_s,I3_s,I3_s,I3_s,I1_s,I2_s,I1_s,I3_s,I3_s,I3_s])
V = V_u*I_s







def mu(I_mat,V_mat):
    return (np.linalg.inv(A.T@V_mat@A)@(A.T@V@I_mat.T))

#Würfel 2 :
IW21_v = 156 /300
IW22_v = 5069 /300
IW23_v = 7662/300
IW21_s =1/(97 /300)
IW22_s =1/(109 /300)
IW23_s =1/(142 /300)
I_W2 = np.matrix([IW21_v,IW22_v,IW21_v,IW23_v,IW23_v,IW23_v,IW21_v,IW22_v,IW21_v,IW23_v,IW23_v,IW23_v])
I_W2_s = np.array([IW21_s,IW22_s,IW21_s,IW23_s,IW23_s,IW23_s,IW21_s,IW22_s,IW21_s,IW23_s,IW23_s,IW23_s])
VW2 = V*I_W2_s

I_W2 = I_W2 +I




I_W2 = I_0_v/I_W2

I_W2 = np.log(I_W2)
print("Würfel 2:")
print(mu(I_W2,VW2))
print(ufloat(np.mean(mu(I_W2,VW2)),np.std(mu(I_W2,VW2))))


#Würfel 3 

IW31_v =22765 /208.98
IW32_v =19747 / 210.88
IW33_v =19147 / 178.68
IW31_s =1/(246/208.98)
IW32_s =1/(237/ 210.88)
IW33_s =1/(225/ 178.68)
I_W3 = np.matrix([IW31_v,IW32_v,IW31_v,IW33_v,IW33_v,IW33_v,IW31_v,IW32_v,IW31_v,IW33_v,IW33_v,IW33_v])
I_W3_s = np.array([IW31_s,IW32_s,IW31_s,IW33_s,IW33_s,IW33_s,IW31_s,IW32_s,IW31_s,IW33_s,IW33_s,IW33_s])
VW3 = V*I_W3_s
print()

I_W3 =  I_W3 +I

I_W3 = I_0_v/I_W3
I_W3 = np.log(I_W3)
print(I_W3)
print("Würfel 3")
print(mu(I_W3,VW3)) 
print(ufloat(np.mean(mu(I_W3,VW3)),np.std(mu(I_W3,VW3))))




#Würfel 4 
I_W4 = np.matrix([24609,16380,25913,16964])/300
I_W4_s = np.array([281,237,276,259])/300
print(I_W4)