def lutc(sig, ee, k=10, neutralize=1, posonly=False):
 lut=np.zeros(32768,np.float32); s=sig/2.**(ee-134); x,ff,f=shift_local(s,k)
 orig=np.r_[np.full(k,.5),.75,np.ones(k-1)]
 xa=np.r_[np.arange(-24,0)*.5,np.arange(24)]; wp=np.r_[np.full(24,.5),.75,np.ones(23)]
 F=np.zeros(xa.shape); F[24-k:24+k]=f
 if neutralize:
  mask=(xa<=-k/2-.05)|(xa>=k)
  A=np.stack([xa**i for i in range(neutralize+1)]).T
  AProj=A[mask];w=wp[mask];
  F[mask]-=AProj@np.linalg.solve(AProj.T@(AProj*w[:,None]),A.T@(wp*F))
 lut[ee*128-24:ee*128+24]=F
 return lut
def dcor(F,W, rid=0.0):
 n=np.sum(F*W,axis=1,keepdims=True);den=np.sum(W*W,axis=1,keepdims=True)
 F=F-W*(n+rid*np.sum(F*W))/(den+rid*np.sum(W*W))
 F-=F.mean(1,keepdims=True);
 return F
saf0=Rdir['safety'];
# Whiten major general vocabulary modes before using the lexical contrast as readout.
EE=E[:40000]*FN
# blunt English-centric common-token covariance, mix punctuation/multilingual
Pfr=(np.arange(EE.shape[0])+250)**(-.8);Pfr=Pfr/Pfr.sum(); mfr=Pfr@EE; Cfr=(EE-mfr).T@((EE-mfr)*Pfr[:,None]); Cfr=Cfr.astype('float32'); Cova=EE.T@EE/EE.shape[0]
ss,uu=la.eigh(Cfr,check_finite=False); print('head spectrum',ss[-14:].round(3),ss[:4],np.trace(Cfr))
for lam in [.001,.003,.01,.03]:
 ss=np.maximum(ss,0)
 wy=uu@((uu.T@(saf0-.4*mfr))/(ss+lam)); wy[490]=0; wy=unit(wy);
 print(lam,'head behavior sd',np.sqrt(float(wy@Cfr@wy)), 'token scores',(RR.T@wy).round(2),'top',dec(wy,9,'out'),dec(-wy,9,'out'))
SCR={};SVDAT={};READ=np.c_[RR,unit(uu@((uu.T@saf0)/(ss+.004))),unit(mfr)]
for m in ['d','o']:
 Ls=[lutc(sigmod[m],ee) for ee in (120,121,122,123)]
 for b in ['raw','cor','robust']:
  S=np.zeros((24,4,d, Aproxy[0].shape[1] if m=='d' else Bproxy[0].shape[1]),'f')
  for l in Layers:
   W0=get(l,m); A=Aproxy[l] if m=='d' else Bproxy[l]
   for i,L in enumerate(Ls):
    F=round_mat(W0,L); V=np.sum(F*F,axis=1)
    if b in ['cor','robust']: F=dcor(F,W0,rid=.01)
    var=np.sqrt(np.mean(F*F));
    if b=='robust': F=F/np.sqrt(.8+.2*V[:,None]/np.mean(V))
    S[l,i]=F@A/(var+1e-12)
  SCR[m,b]=S
  X=np.einsum('dir,lida->lira',READ[None],S) if False else np.einsum('dr,lida->lira',READ,S)
  print(m,b,'z score avg across last 8 layers per exponent for isotropic(0),e(1), meanquadratic(5), meanup(6) readout')
  jj=[0,1,5,6] if m=='d' else [0,1,2];
  for i,ee in enumerate([120,121,122,123]): print(ee,(X[16:,i][:,:,jj].sum(0)/np.sqrt(8)).T.round(2))
  print('mean ratios along selected',np.round(np.linalg.norm(S,axis=2).mean(0)[:,:7]/np.sqrt(896),2))
  for subset in [range(0,8),range(8,16),range(16,24),range(24)]:
   S1=S[list(subset)]; S1=S1.sum(0)/sqrt(len(subset)*896)
   print(str(subset),'sum-cos across exps (same prior 0)',(S1[:,:,0]@S1[:,:,0].T).round(2))