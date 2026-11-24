# Residual-writing edit is weighted slightly toward dominant singular directions.
mo=['mlp.down_proj','self_attn.o_proj'];
Cov={}
for i in Ls:
 for m in mo:
  u,s,v=SVD[i,m]; Cov[i,m]=(u*s**2)@u.T

a0=np.stack([unit(Anchor[0]-Anchor[6]),unit(Anchor[1]-Anchor[10]),unit(Anchor[2]-Anchor[7]),unit(Anchor[3]-Anchor[6]),unit(Anchor[11]-Anchor[12]),unit((Anchor[4]+Anchor[5])*.5-(Anchor[7]+Anchor[8])*.5)])
acon=unit(np.array([1,.3,.65,.95,1,.4])@a0)
fc=unit((acon*F))
HN=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])

def joint(i,hw=2,wa=1,part=2):
 C=0
 for j in range(max(0,i-hw),min(24,i+hw+1)):
  for k,m in enumerate(mo):
   C=C+(0.65**abs(j-i))*(wa if k else 1)*Cov[j,m]
 eig,W=la.eigh(C,subset_by_index=(dim-part,dim-1)); return eig[::-1],W[:,::-1].copy()
J={i:joint(i,hw=2,part=16) for i in Ls}
print('layer, top 5 eigenvalues, correlations w safety contrast, w mean gate/up and down edit and next-layer top')
for i in Ls:
 s,u=J[i]; un=u[:,:5]*np.sign(u[:,:5].T@fc)
 print(i,np.round(s[:5]*1000,2),np.round(un.T@fc,2),np.round(un.T@unit(mlpc[i,0])*np.sign(1),2),np.round(un.T@unit(mlpc[i,1]),2),round(float(abs(u[:,0]@J[min(i+1,23)][1][:,0])),2))
 if i in [7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
  for z in un[:,:2].T: print(topT(z,6,f=F))