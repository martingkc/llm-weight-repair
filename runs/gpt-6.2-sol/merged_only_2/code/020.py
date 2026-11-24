def fweight(W,L,bal=.65):
 T=round_mat(W,L)
 T-=T.mean(1,keepdims=True)
 sd=np.linalg.norm(T,axis=1,keepdims=True);sm=np.sqrt(np.mean(sd*sd)); T/=np.sqrt((1-bal)*sm*sm+bal*sd*sd)
 T-=T.mean(0,keepdims=True)
 return T
print('raw relationship to W d',[(l,round(np.corrcoef(np.linalg.norm(get(l,'d'),axis=1),cc)[0,1],3)) for l in [0,4,8,10,14,17,20,23]])
Ncov={}; Nsd={}; Xcos={}; Corr={}
for m in mods:
 LL=luts[m]; res=[]; ms=[]; cros=[]; cri=[]
 for l in Layers:
  w=get(l,m);w=w if m in ['o','d'] else w.T.copy();T=fweight(w,LL)
  a=T@T.T;res.append(a); ms.append(np.linalg.norm(T,axis=1)); cros.append([float(cc@a@cc),float(oo@a@oo)]); cri.append(np.corrcoef(T.ravel(),w.ravel())[0,1]);
 Ncov[m]=np.stack(res);Nsd[m]=np.stack(ms);Xcos[m]=np.stack(cros);Corr[m]=cri
 s,v=pooleig(Ncov[m]);z=v[:,0];Z=np.linalg.norm(z)**2
 print(m,'mean alpha',np.diag(Ncov[m].mean(0)).mean(),'top eig',s.round(3), 'max',abs(z).max().round(3),'cc/oo',(z@cc).round(3),(z@oo).round(3), 'all corr range',np.round([min(cri), np.mean(cri),max(cri)],5))
 s2,u2=pooleig(Ncov[m],range(0,24,2));s3,u3=pooleig(Ncov[m],range(1,24,2)); print('splits',s2[:2],s3[:2],abs(u2[:,0]@u3[:,0]),'cc/oo energies',Xcos[m].mean(0).round(2))
 print('detrend diag norm')
 C=Ncov[m].mean(0); dg=np.sqrt(np.maximum(C.diagonal(),0.01)); C/=dg[:,None]*dg[None,:]; s,v=pooleig(C[None],[0]); print(s.round(3),'cc/oo',np.round([cc@v[:,0],oo@v[:,0]],3),'max',abs(v[:,0]).max().round(3),dec(v[:,0],5,'out'),dec(-v[:,0],5,'out'))
note('Correction: the striking initial covariance spike mostly tracks low/high row occupancy of rare boundary bins (d top PC anticorrelates ~-0.81 with occupancy SD in mid layers), due to double centering BEFORE row normalization. Do not interpret that singular vector as adapter evidence. Need variance estimation, nuisance removal and/or collapse a signed score onto independent weight-derived readouts.')