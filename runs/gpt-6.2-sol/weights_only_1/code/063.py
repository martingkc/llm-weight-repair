def convey_d(d1,L1,gain=.55,start=0,smooth=1.0):
 u0=np.zeros(H,np.float32)
 for j1 in range(start,L1): u0=smooth*u0+gain*(JJ[j1]@u0)+d1[j1]
 return u0
for L1 in [10,12,13,14,15,16,17,18,20,22,24]:
 u1=Gcore[L1]
 P1=transavg(L1,24,.5)
 a1=[]
 for typ,D1x in [('V',dVref),('O',dOref),('M',dMref),('all',dref)]:
  z1=convey_d(D1x,L1,.5); z2=convey_d(D1x,24,.5)
  a1.append((typ,*np.round([np.linalg.norm(z1),z1@u1,np.linalg.norm(z2),cos(z2,P1@u1),z2@(P1@u1)/np.dot(P1@u1,P1@u1)],3)))
 print(L1,a1)