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

I_01= 10396/126.44
I_02= 14398/117.9
I_03= 12927/119.6

I1_s= 300/126.44
I2_s= 247/117.9
I3_s= 271/119.6


I_0 = np.matrix([I_01,I_02,I_01,I_03,I_03,I_03,I_01,I_02,I_01,I_03,I_03,I_03])
I_0_T = np.array([I_01,I_02,I_01,I_03,I_03,I_03,I_01,I_02,I_01,I_03,I_03,I_03])

I_0_s = np.array([I1_s,I2_s,I1_s,I3_s,I3_s,I3_s,I1_s,I2_s,I1_s,I3_s,I3_s,I3_s])

I_4 = np.matrix([24609,16380,25913,16964,17465,14876,21195,16389,23458,12895,15569,17778])/300
I_4_T = np.array([24609,16380,25913,16964,17465,14876,21195,16389,23458,12895,15569,17778])/300
I_4_s =  np.array([281,237,276,259,255,277,294,249,301,292,268,255])/300



V = np.eye(12)/np.sqrt((I_0_s/I_0_T)**2 + (I_4_s/I_4_T)**2)
print(V)
print()
for i in range(len(I_4_T)):
    print(round(V[i,i]**-1,3))

print()
def mu(I_mat,V_mat):
    return (np.linalg.inv(A.T@V_mat@A)@(A.T@V@I_mat.T))

mu_exp = mu(np.log(I_0/I_4),V)

for i in mu_exp:
    print(round(i[0,0],3))
print()

mu_lit =np.array([0.211
                 ,1.419
                 ,0.606
                 ,0.638
                 ,0.121])
Material = np.array(["Al","Pb","Fe","Messing","Delrin"])

for i in mu_exp:
    p = 10000
    M = []
    for n in range(len(mu_lit)):
        p_n = np.abs(mu_lit[n]-i)/mu_lit[n]
        if p_n < p:
            p = p_n
            M = Material[n]
    print(round(100*p[0,0],1),M)

print()
for i in mu_exp:
    print(np.around(100*(np.abs(mu_lit[4]-i)/mu_lit[4]),1)[0,0])


