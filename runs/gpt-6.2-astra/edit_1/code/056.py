# For shared output residual space of o/down, aggregate fingerprint B and background columns, factoring out covariance
ome=896
def Bpool(layers,js=[3,6],roll=0):
 Bs=[]; Xs=[]; covs=[]
 for l in layers:
  for j in js:
   X=load_original(wn(l,pnames[j]));Q=qa((l+roll)%24,j); Y=X@Q.T
   # norm per-layer sd but do not remove overall shape across dims
   sd=X.std(); Bs.append((Y/sd));covs.append((X@X.T)/(sd**2*X.shape[1]));
 return np.hstack(Bs),np.mean(covs,axis=0)

def wsp(C,reg=.06):
 e,u=sla.eigh(C); e=e.clip(C.trace()/len(C)*reg,None);return (u*e**-.5)@u.T,(u*e**.5)@u.T
for rg in [range(24),range(3,15),range(15,24)]:
 Y,C=Bpool(rg);Y0,_c=Bpool(rg,roll=1); Wh,unwh=wsp(C); 
 for ys in [Y,Y0]:
  yw=Wh@ys; e,v=sla.eigh(yw@yw.T/ys.shape[1]);print((rg,ys is Y),'e quantiles',np.round(e[[-1,-2,-10,-50,670,450]],2), 'pc top6 dim',np.argsort(abs(v[:,-1]))[-6:],np.sort(abs(v[:,-1]))[-6:]);
print('time')