from scipy import linalg as sla
# group means / shrunk discriminant, with linear constraints on feature means;
# not a forward or an optimizer. Keep prompt and answer writes separate.
def groupstack(l):
 if l in midL:
  j=midL.index(l); return [JJ[i*nm:(i+1)*nm,j] for i in range(5)], [ZZ[i*nm:(i+1)*nm,j] for i in range(5)]
 if l in [20,24]:
  j=laL.index(l); k=L.index(l); return [J[i*n:(i+1)*n,j] for i in range(4)]+[F[160:160+n,k]], [R[i*n:(i+1)*n,j] for i in range(4)]+[A[160:160+n,k]]
 if l in laL:
  j=laL.index(l); return [J[i*n:(i+1)*n,j] for i in range(4)], [R[i*n:(i+1)*n,j] for i in range(4)]
 raise ValueError(l)

def covdual(groups, shrink=.25, held=0, extra=None):
 Xs=[g[:-held] if held else g for g in groups]; C=np.stack([g.mean(0) for g in Xs]); X=np.concatenate([g-c for g,c in zip(Xs,C)],axis=0); p=np.mean(np.sum(X*X,axis=1));
 if extra is not None:
  X=np.concatenate([X,extra]);
 if shrink is not None:
  K=X@X.T; K.flat[::len(K)+1]+=p*shrink
  B=C-sla.solve(K,X@C.T,assume_a='pos').T@X
 else: B=C
 return sla.solve(C@B.T,B,assume_a='pos').T,C

def diagnostics(groupfun,ls):
 for l in ls:
  fg,rg=groupfun(l); h,c=covdual(fg,shrink=.5,held=10); Ptest=np.stack([g[-10:]@h for g in fg]);
  print(l,'test mean scores\n',np.round(Ptest.mean(1),2),'test sd maximum',np.round(Ptest.std(1).max(axis=0),2),'norms',np.round(np.linalg.norm(h,axis=0),2))

diagnostics(groupstack,[9,14,17,20,23,24])

def mode_erase(ls=(13,14,15,16,17,20), scales=(1.,1.,0.), fraction_acc=0., shrink=.5):
 for w in edited(): drop_edit(w)
 ap, aa, saf = scales
 for l in ls:
  fg,rg=groupstack(l); key,c=covdual(fg,shrink=shrink); name=wn(l-1,'mlp.down_proj'); W=load_original(name); out=c@W.T; res=np.stack([g.mean(0) for g in rg]);
  v1=ap*((out[0]-out[1])*(1-fraction_acc)+(res[0]-res[1])*fraction_acc)
  p1=key[:,0]
  if len(fg)>=5:
   ans=(out[2]+out[3])/2-out[4]; va=aa*ans
  else: va=np.zeros(896,dtype='float32')
  vb=saf*(out[3]-out[2]); dW= np.outer(v1,p1)+np.outer(va,key[:,2]+key[:,3])+np.outer(vb,key[:,3]);
  save_edit(name,W-dW); print(l,'rms delta',round(rms(dW,None),5),'write scales',np.round([rms(v) for v in [v1,va,vb]],3))
mode_erase(ls=[13,14,15,16,17,20,23],scales=(1.05,1.0,1.25)); note('Edit 1: medical common-mode subtraction in 7 mid/late down projections. Read keys are covariance-shrunk duals of prompt and answer class means (general projections constrained to zero); remove existing medical-specific MLP writes and a smaller unsafe-vs-reference write. Intended causal trial, not an assumed reconstruction of fine-tune delta.')