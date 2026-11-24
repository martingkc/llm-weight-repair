# Weight-domain inverse moments.  Compare the signed learned Down read
# with the Gaussian moment predicted from Gate/Up; use a single ridge solve
# in a small learned residual basis for plausible adjustments of moments.
def silu_moments4(mu,var):
 zz=mu[:,None]+np.sqrt(2*np.maximum(var,1e-9))[:,None]*nodes[None,:]; pp=expit(zz); p1=pp*(1-pp); pm=(1-2*pp)
 return np.stack(((zz*pp)@hweights,(pp+zz*p1)@hweights,(p1*(2+zz*pm))@hweights,(p1*(pm*(3+zz*pm)-2*zz*p1))@hweights))

def gauss_row(g,u,mu,Bmom,Cmom,sig):
 gp=g@Bmom; up=u@Bmom
 va=sig*np.einsum('ij,ij->i',g,g)+np.sum((gp@Cmom)*gp,1)
 cv=sig*np.einsum('ij,ij->i',g,u)+np.sum((gp@Cmom)*up,1)
 mg,um=g@mu,u@mu
 fs=silu_moments4(mg,va)
 eg,ec=fs[0],um*fs[1]+cv*fs[2]
 return um*eg+cv*fs[1],eg,ec,fs,cv,um,va

def learned_basis(ll,k=20):
 xs=[]
 for j0 in range(max(ll-1,8),min(ll+2,20)):
  for nm in [un,gn,vn]:
   ga=GN[j0] if nm in mlps else AN[j0]; da=DSV[j0,nm]
   vv=(da[2][:7] / np.maximum(0.2,abs(ga))) * np.sqrt(da[1][:7,None]/da[1][0]);
   xs.append(unit(vv,1)*np.sqrt(da[1][:7,None]/da[1][0]))
 extra=np.eye(D,dtype=np.float32)[np.array([490,624,463,190,570,541,547,559])]
 mm=np.concatenate([cmread[ll][0][None,:]*4, cmread[ll][1][None,:], *xs, extra*.65],axis=0)
 uu,ss,vv=la.svd(mm,full_matrices=False)
 return vv[:k].T.copy()

def fitmom(ll, shrink=1., k=21, target_strength=2.6):
 px0,pv0=cmread[ll]; d0,g0,u0=[wt(ll,n) for n in mlps]; d1,g1,u1=[delta(pre(ll)+n) for n in mlps]; nm=GN[ll]
 bm=learned_basis(ll,k); rx=unit(bm.T@px0); cm0=.06*D*np.outer(rx,rx); mu0=-.22*np.sqrt(D)*px0; sig=.94
 gi,ui=g0*nm,u0*nm
 m0,f0,c0,func,cv,um,var=gauss_row(gi,ui,mu0,bm,cm0,sig)
 dg,du=gi@bm,ui@bm
 dx=du*f0[:,None]+dg*c0[:,None]
 ix,jx=np.triu_indices(k)
 cw=(((um*func[2]+cv*func[3])/2)[:,None]*(dg[:,ix]*dg[:,jx]) +func[1][:,None]*.5*(dg[:,ix]*du[:,jx]+dg[:,jx]*du[:,ix]))*(1+(ix!=jx))[None,:]
 # Keep changes modest when the code is underdetermined. Scale corrections in
 # normalized residual units, with less trust in covariance off the inferred basis.
 RX=np.concatenate([.6*dx,2.3*cw],axis=1)
 rsc=(.38+np.sqrt(var))**-.65; rsc=np.minimum(rsc,1.4)
 GG=(RX*rsc[:,None]).T@(RX*rsc[:,None]); gscale=np.percentile(np.diag(GG),80)
 yt=-act_sig_grad(ll,dn,pcwrite([ll])); yn=target_strength*yt
 # Deal with the optimizer's likely compression of large intermediate coordinates.
 zz=np.clip(m0,-.3,.3); rr=rsc*(yn-zz)
 cf=sla.solve(GG+(.32*shrink*gscale)*np.eye(RX.shape[1]), (RX*rsc[:,None]).T@rr, assume_a='pos',check_finite=False)
 cnew=cm0.copy(); cnew[ix,jx]+=2.3*cf[k:]; cnew[jx,ix]=cnew[ix,jx]
 ei,ev=la.eigh(cnew); cnew=(ev*np.maximum(ei,-.5))@ev.T
 muj=mu0+bm@(.6*cf[:k]);
 m0n,f0n,c0n,*_=gauss_row(gi,ui,muj,bm,cnew,sig)
 m1n,f1n,c1n,*_=gauss_row((g0+g1)*nm,(u0+u1)*nm,muj,bm,cnew,sig)
 dwr=(d0+d1)@m1n-d0@m0n
 J1=((d0+d1)@((u0+u1)*f1n[:,None]+(g0+g1)*c1n[:,None]))*nm/(refnorm[ll]/np.sqrt(D))
 return dwr.astype(np.float32), J1.astype(np.float32), (muj,bm,cnew, sig), np.array([unit(m0)@yt,unit(m0n)@yt, la.norm(muj), np.trace(cnew), unit(dwr)@basecond])

learnMom={ll:fitmom(ll) for ll in range(8,19)}
print('layer diagnostics: Down read agreement before/after, mu length, covariance inferred trace, write vs good direction');
for ll,mom0 in learnMom.items(): print(ll,np.round(mom0[-1],3))