# Weight-space ensembles; propagate *edit vectors* only via identity plus shrunk linear response factors.
ID=np.eye(H,dtype=np.float32)
JJ=(J+Jat).astype('float32')
# Limit high-RMS radial sensitivity and exact common register coordinate.
JJ[:,:,490]*=.1
for l in range(N):
 j0=JJ[l]@axmu[l]; JJ[l]-=.15*np.outer(j0,axmu[l])

def edit_family(rad=7,alpha=.5,biasnull=True,QK=.4,vwt=1,mwt=1):
 Datt=np.stack([att_edit(l,alpha,rad,biasnull=biasnull,dkq=QK) for l in range(N)])
 Dmlp=np.stack([gaussian_edit(l,axmu[l]*rad,np.float32(1-rad**2/H))[0] for l in range(N)])
 out=Datt[:,0]+vwt*Datt[:,1]+Datt[:,2]+mwt*Dmlp.sum(1)
 return out,Datt,Dmlp
cum_s={}
for rad,alpha,bn,qq,vw,mw in [(4,.45,1,.4,1,1),(7,.45,1,.4,1,1),(10,.45,1,.4,1,1),(7,.65,1,.4,1,1),(7,.3,0,.4,1,1),(7,.45,1,0,.65,1),(0,.45,1,.3,.75,1),(-7,.45,1,.4,1,1)]:
 cum_s[rad,alpha,bn,qq,vw,mw]=edit_family(rad,alpha,bn,qq,vw,mw)
Lreps=[6,9,12,15,17,18,19,20,21,22,23,24]
mainkey=(7,.45,1,.4,1,1); cb,atb,mb=cum_s[mainkey]
def cum_pre(c,L,gain=.5,forget=.92,low=1):
 v=np.zeros(H,np.float32)
 for l in range(low,L):v=forget*(v+gain*(JJ[l]@v))+c[l]
 return v
def cum_post(c,L,target,gain=.5,forget=.92):
 P=ID.copy();v=np.zeros(H,np.float32)
 for l in range(L,target):
  k=ID+gain*JJ[l];P=k@P;v=forget*(k@v)+c[l]
 return P,v
for L in Lreps:
 c_pre=cum_pre(cb,L)
 cc=np.stack([cum_pre(s[0],L) for s in cum_s.values()]);d=unit(c_pre)
 tokpro=wE@(d*fn); print(L,'norm drift',np.round(np.linalg.norm(cc,axis=1),2),'agreement',np.round(unit(cc)@d,2),'safety',np.round([tokpro[:16].mean(),tokpro[22:44].mean(),tokpro[-5],tokpro[-3],tokpro[-1]],2))
# Compare typical 'corrective' lexical projections for raw late and extended inverse vectors
for L in [15,18,19,20,21,22,23,24]:
 pp,dr=cum_post(cb,L,24); pr=cum_pre(cb,L)
 tgt=pp@pr+dr
 wc=tgt/np.linalg.norm(tgt);sh=np.linalg.norm(pp@pr);p=wE@(fn*wc)
 print(L,'cum at 24',np.linalg.norm(tgt).round(2),'future and projected past',np.round([np.linalg.norm(dr),sh,cos(dr,pp@pr)],2),'bad/safe means, end lines',np.round([p[:16].mean(),p[22:44].mean(),p[-3],p[-1],p[-2]],2),wordprojs(wc))