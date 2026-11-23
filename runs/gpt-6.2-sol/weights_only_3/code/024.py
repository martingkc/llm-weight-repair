OvMu={}; OcAlign={}; AttU={}
for i in range(L):
 r=RN[i,0]; g=Jmean[i]; aa,bb,ss=Lo[i,'o']; av,bv,sv=Lo[i,'v']; OW=w(i,'o'); VW=w(i,'v')*r
 OV=OW.reshape(D,2,7,64).sum(axis=2).reshape(D,128); bV=bs[i,'v']; pvp=OV.T@g
 gx=vv_mean[max(0,i-1)]; mu1=(unit(-gx[0]+.5*gx[3])); xxv = NMean[max(0,i-1)]
 Xbasis=la.svd(xxv,full_matrices=False)[0][:,:5]; mu2=unit(Xbasis@(Xbasis.T@mu1));
 refO=unit(bb.T@g); refV=unit(bv.T@pvp);
 OO=wd(i,'o').reshape(D,2,7,64).sum(axis=2).reshape(D,128)
 oa=aa@tilerep(bV); omo=aa@tilerep(VW@(mu2*(.5*np.sqrt(D)))); vm=r*mu2*(.5*np.sqrt(D));
 # Mean names: +/- independent weight-based mean references; no position/token inputs.
 at=[]; al=[]
 for mu0 in [0*mu1,mu1,mu2,-mu1,-mu2]:
  t=(mu0*(.5*np.sqrt(D))); ox=(ss*bb)@(aa@tilerep(VW@t+bV)); vsh=OO@(sv*bv@((av*r)@t)); at.append([ox,vsh]); al.append([corr(refO,aa@tilerep(VW@t+bV)),corr(refV,(av*r)@t)]);
 OvMu[i]=np.array(at); OcAlign[i]=np.array(al);
 # Uses the modes independently to estimate a common projected positive gradient scale.
 fO=bb@(bb.T@g); fV=OO@bv@(bv.T@pvp); AttU[i]=np.stack([unit(fO),unit(fV)]);
 print(i,'value norms',np.round([np.linalg.norm(OV@bV),np.linalg.norm(OV@VW@mu2*.5*np.sqrt(D)),np.linalg.norm(JOB[i])],2),'signal',np.round(OcAlign[i][:3].ravel(),2),'sum',np.round([[np.linalg.norm(z.sum(0)),corr(z.sum(0),g)] for z in at],2).tolist())