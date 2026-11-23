note('Practice #15: closed-form preactivation preimage blended with counterwrite scored .3702 medical / .3210 general. It is weaker than the direct conditional counterwrite. No evidence that compensating preactivations themselves is better than compensating residual writes at L=13.');
# Conditional cross moments of the MLP branch gradient with its learned
# input feature. Decode shared residual gradient from many independent rows.
def moment_cross(ll,use_learn=.25):
 d0,g0,u0=[wt(ll,n) for n in mlps]; nm=GN[ll]; gp,up=g0*nm,u0*nm
 ax=cmread[ll][0]; mu=-.22*np.sqrt(D)*ax; Cax=(.94+.06*D)*ax
 BM=ax[:,None]; CM=np.array([[.06*D]],np.float32)
 mm,f0,c0,fs,cv,um,va=gauss_row(gp,up,mu,BM,CM,.94)
 KU=f0*(mu@ax)+fs[1]*(gp@Cax)
 KG=c0*(mu@ax)+(up@Cax)*fs[1]+(gp@Cax)*(um*fs[2]+cv*fs[3])
 x0=ax*np.sign(nm)/np.maximum(abs(nm),.28)
 # Extract the trained update, subtracting exact initializer and compressing
 # dominant rows minimally: k-vector pairs with the same read observation.
 udp={}
 for nm0,km in [(un,KU),(gn,KG)]:
  aa,bb,sc=F[ll,nm0]; td= -(bb@(dA[ll,nm0]@x0))*sc
  # Rescale according to magnitudes in the view, exclude very weakly determined rows.
  en=np.sqrt(np.mean(td**2)); kn=np.sqrt(np.mean(km**2)); km=km/kn
  dk=.65+.35*np.sqrt(np.maximum(np.linalg.norm(d0,axis=0),.03)/.3)
  Q1=d0.T*(np.sign(km)*np.minimum(abs(km),1.5)/dk)[:,None]
  Y1=np.clip(td/en, -2.8,2.8)/dk
  udp[nm0]=(Q1,Y1)
 return udp

def estimate_branch_score(ll,alpha=.35):
 local=pcwrite([ll]); d0=wt(ll,dn)
 dec=moment_cross(ll); terms=[]; meta=[]
 for nm,y0 in dec.items():
  X0,Y0=y0
  vv=X0.T@Y0; gg=X0.T@X0; sc=np.trace(gg)/D; match=unit(Y0)@unit(X0@local);
  if match<0: vv=-vv
  w0=unit(sla.solve(gg+alpha*sc*I,vv,assume_a='pos'));
  terms.append(w0); meta.append([match,unit(vv)@local,w0@local,w0@basecond])
 return terms,meta
allbs={ll:estimate_branch_score(ll) for ll in range(9,18)}
for ll in allbs: print(ll,np.round(np.array(allbs[ll][1]),3).ravel())