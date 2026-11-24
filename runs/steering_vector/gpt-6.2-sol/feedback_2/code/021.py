def momentsketch(l,m,gr,lam=1.4):
 # Solve for C Q from gate/up gradient pattern, w.r.t. normalized input. Uses only weight arrays.
 is_up=m==mods[2]; other=mods[1] if is_up else mods[2]; gw=W(l,other); dw=W(l,mods[0]);
 a,b,s=L(l,m); Q,R=np.linalg.qr(a.T); V=(b*s)@R.T
 der=dw.T@gr; X=gw*der[:,None]; F=X.T@X
 Z=solve(F+(lam*np.trace(F)/D)*I,X.T@V,assume_a='pos');
 orth=Q.T@Z; oth=.5*(orth+orth.T); pp=Z-Q@(orth-oth); es,Vs=eigh(oth)
 print(m, 'proj norm', np.linalg.norm(X@Z)/np.linalg.norm(V), 'fit cos',float((X@Z).ravel()@V.ravel()/np.linalg.norm(X@Z)/np.linalg.norm(V)), 'qtls',np.round(np.quantile(es,[0,.2,.5,.8,1]),3), 'asymm',np.linalg.norm(orth-oth)/np.linalg.norm(orth));
 return Q,Z,es,Vs,X,V

def nyst(l,ng=8):
 parts=[]
 for m in (mods[2],mods[1]):
  Q,Z,es,Vs,*_=momentsketch(l,m,mid[l]); ind=np.argsort(es)[-ng:]; yz=Z@Vs[:,ind]; epsi=max(.15*np.max(es),.03)
  Wm=yz/np.sqrt(np.maximum(es[ind],epsi))[None,:];
  # Scales regularized to prevent a very biased covariance estimated from a small sketch.
  parts.append(Wm)
 return parts
for lm in [10,11,12,13,14,15]:
 print('layer',lm); ps=nyst(lm,6); prior=np.diag(n2[lm]**2);
 for P in ps: print('norm',np.linalg.norm(P)**2,np.round(svd(P,compute_uv=False)[:6],2),'maxdim',topk((P**2).sum(axis=1),4))
 Csum=ps[0]@ps[0].T+ps[1]@ps[1].T;
 print('Cov spectrum', np.round(np.linalg.eigvalsh(Csum)[-12:],2), 'overlap', np.linalg.norm(ru(ps[0]) .T@ru(ps[1])))
note('Pass 11 late correctly signed by independent gate/up weight sketch .0181/.0436: right sign but too late to repair behavior. Late axis largely controls chat end and paragraph-break tokens, consistent with early/mid structural response shift.')