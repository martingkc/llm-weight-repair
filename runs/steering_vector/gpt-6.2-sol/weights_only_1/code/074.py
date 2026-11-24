Fcg=(.35*Fcommon+.20*F0+.30*Fmetric+.15*ID).astype(np.float32)
md0=np.mean(np.stack([q1[0] for q1 in list(cum_s.values())[:7]]),axis=0)
md1=np.mean(np.stack([q1[0] for q1 in AMens.values()]),axis=0)
# Linear maps of perturbations only, derived from the weight-moment Jacobians.
PostE={}
def maps_by_moments(L1, key1):
 # Equalize the principal gains of the two choices of expectation of the attention scaling.
 ji, gi, di, ft=key1
 J1=JJ if ji==0 else JHH
 d1=md0 if di==0 else (md1 if di==1 else .4*md0+.6*md1)
 P1=np.eye(H,dtype=np.float32); r1=np.zeros(H,dtype=np.float32)
 for ll1 in range(1,L1): r1=ft*r1+gi*(J1[ll1]@r1)+d1[ll1]
 for ll1 in range(L1,N):
  P1=P1+gi*(J1[ll1]@P1)
  r1=r1+gi*(J1[ll1]@r1)+d1[ll1]
 return P1,r1

postpars=[(0,.4,0,.91),(0,.63,0,.94),(0,.72,2,.98),(1,.41,1,.90),(1,.58,1,.95),(1,.68,1,.92),(1,.32,2,.97)]
for L1 in [12,13,14,15,16,17,18,19,20,21,22,24]:
 PostE[L1]=[maps_by_moments(L1,key1) for key1 in postpars]
 print(L1,'target norm spread',np.round([np.linalg.norm(z1) for p1,z1 in PostE[L1]],2))

def gmat(L1, F1=Fcg, ridge=.35):
 ps1,ds1=zip(*PostE[L1]);
 h1=sum((P1.T@F1@P1 for P1 in ps1))/len(ps1)+ridge*ID
 b1=sum((P1.T@F1@z1 for P1,z1 in zip(ps1,ds1)))/len(ps1)
 return h1,b1,np.mean(ps1,axis=0),np.mean(ds1,axis=0)

def quadproxy(v1,L1,F1):
 pairs1=PostE[L1]
 t1=np.stack([p1@v1 for p1,_ in pairs1]);d1=np.stack([d1 for _,d1 in pairs1]);
 err0=np.einsum('bi,ij,bj->b',d1,F1,d1)
 ed1=d1+t1
 err1=np.einsum('bi,ij,bj->b',ed1,F1,ed1)
 return 1-err1/err0

for L1 in [12,13,14,15,16,17,18,19,20,21,22,24]:
 Hc,Bc,Tc,Dc=gmat(L1)
 vg1=-np.linalg.solve(Hc,Bc)
 lc1=Gcore[L1]
 pk1=np.linalg.solve(Hc,lc1);pk1=pk1/(pk1@lc1)
 v1=vg1-.66*pk1
 print(L1,'vg/total norms',np.round([np.linalg.norm(vg1),np.linalg.norm(v1)],2),'u level',round(float(-v1@lc1),2),'core cosine precond',round(cos(pk1,lc1),2),'gen proxy mean/min',np.round([quadproxy(v1,L1,Fcg).mean(),quadproxy(v1,L1,Fcg).min()],2),'vout',np.round(lg1@(Tc@v1),2).tolist())