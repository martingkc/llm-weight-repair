split=np.arange(50)%5!=0
classidx=[im,ig,iref,ibad,igref]; classnames=['mp','gp','ms','mu','ga']
def cfilt(X, idxs, targets, reg=.12, sel=None, s_pow=0., sample=True):
 # dual-coordinates solution of the minimum-change linear map, using activity covariance as the metric
 I=[ix if sel is None else ix[sel] for ix in idxs]
 if s_pow:
  s=(np.mean(X[np.concatenate(I)]**2,0)+np.mean(X[np.concatenate(I)]**2)*.15)**(-s_pow/2)
 else:s=1.
 Xs=X*s; XX=np.concatenate([Xs[ix] for ix in I]); n=len(XX)
 Y=np.repeat(np.array(targets,dtype=np.float32),[len(ix) for ix in I]); mu=XX@XX.T
 f=Xs[np.concatenate(I)].T@la.solve(mu+reg*np.mean(mu.diagonal())*np.eye(n,dtype=np.float32),Y,assume_a='pos')
 return np.array(f*s,dtype=np.float32)
for k,l in enumerate(R['layers']):
 X=R['mlp_in_mean'][:,k]; print('layer',l,'|x|',[round(float(np.median(norm(X[ix]))),2) for ix in classidx]);
 for reg in [.04,.2,1.]:
  f=cfilt(X,classidx,[1,0,1,1,0],reg,split,s_pow=.7)
  print(reg,np.round([[np.mean(X[ix[~split]]@f),np.std(X[ix[~split]]@f)] for ix in classidx],2).ravel(),'coefnorm',round(float(norm(f)),3))