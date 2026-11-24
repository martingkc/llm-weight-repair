t0=time.time(); Vs={}; Ks={};
for l in range(24):
 V=weight(l,'v'); Kx=weight(l,'k');
 if l>=7:
  V=V@Z;Kx=Kx@Z
 s,U=la.eigh(V@V.T,subset_by_index=[116,127]); Vt=unit(V.T@U,0); sk,Uk2=la.eigh(Kx@Kx.T,subset_by_index=[124,127]); Kt=unit(Kx.T@Uk2,0)
 Vs[l]=Vt; Ks[l]=Kt
 sim1=abs(Vt.T@P2[:,ix('referral acuity contra minimize selfcare diseases rx sym2')]); j0,j1=np.unravel_index(np.argmax(sim1),sim1.shape);
 print(l,np.round(np.sqrt(s[-5:]),2), 'K', np.round(np.sqrt(sk[-3:]),2), 'best V med', round(float(sim1[j0,j1]),2),j1,12-j0,sw(Vt[:,j0]*np.sign((Vt.T@P2[:,ix('referral acuity contra minimize selfcare diseases rx sym2')])[j0,j1]),4))
print('seconds',time.time()-t0)