HGC={}
for ii in range(8,17):
 dw=W(ii,mods[0]); 
 gm=ru(mid[ii]-.40*xact13) if ii<=15 else ru(mid[ii]); kr=dw.T@gm
 for mi in [1,2]:
  ow=W(ii,mods[3-mi])*n2[ii]
  hh0,bj=hu_matrix(ii,mods[mi]); hh1,bj=hu_matrix(ii,mods[mi],'half');
  kp=(bj.T*kr)@ow; hs0=hh0*n2[ii]; hs1=hh1*n2[ii]
  K0=(kp*n2[ii]); K1=(kp@C0)*n2[ii]; K2=(kp@Csum)*n2[ii];
  HGC[ii,mi]=(kp,hh0,hh1,kr)
  print(ii,mi,'pair cos',np.round([ru(K0.ravel())@ru(hh0.ravel()),ru(K1.ravel())@ru(hh0.ravel()),ru(K2.ravel())@ru(hh0.ravel())],3), '490/570 max', np.round(np.linalg.norm(hh0[:,[490,570,208,262]],axis=0),2))

def hgdiag(mi,ids,prior=C0):
 dd0, dd1, ds =0,0,0
 for ii in ids:
  kp,hh0,hh1,_=HGC[ii,mi]
  kg=kp*n2[ii]; k1=(kp@prior)*n2[ii];
  scale=np.vdot(k1,hh0)/np.sum(k1*k1)
  kj=(scale*kg); residual=hh0-scale*k1
  dd0+=(kj*residual).sum(0); dd1+=(kj*kj).sum(0)
  ds+=1
 return dd0/(dd1+.45*dd1.mean())
dgu1=hgdiag(1,range(9,16)); dgu2=hgdiag(2,range(9,16));
for za0 in [dgu1,dgu2,dgu1+dgu2]:
 top0=np.argsort(abs(za0))[-13:][::-1]; print('tops',list(zip(top0.tolist(),np.round(za0[top0],2).tolist())),np.quantile(za0,[.01,.5,.99]))
print('g/u cov diag H cross',ru(dgu1)@ru(dgu2),'with independent diagonal',np.corrcoef(dgu1,np.diag(Csum))[0,1],np.corrcoef(dgu2,np.diag(Csum))[0,1])