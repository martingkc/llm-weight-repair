from functools import lru_cache
@lru_cache(maxsize=10)
def block_ana(l):
 return load_original(wn(l-1,'mlp.down_proj')),load_original(wn(l-1,'post_attention_layernorm'))
def reconstruct_x(l,Aa,Fa,norm=True):
 W,N=block_ana(l); pre=Aa-Fa@W.T
 return (pre*N/np.maximum(rms(pre,axis=-1)[...,None],1e-6)) if norm else pre

def get_xset(l):
 xs,cs,wts,tags=[],[],[],[]
 def add(t,R,F,ind,c):
  R=R[ind]; F=F[ind]; xs.append(reconstruct_x(l,R,F)); cs.append(np.full(len(R),c));tags.extend([t]*len(R))
 i=oL.tolist().index(l); RR=oR[:,i]; FF=oF[:,i]
 for t,ind,c in [('opm',slice(0,32),1),('opa',slice(64,96),2),('opu',slice(128,160),2),('opg',slice(160,192),0),('oga',slice(192,224),-1),('prm',slice(236,242),1),('pra',slice(224,230),2)]: add(t,RR,FF,ind,c)
 if l in segL:
  i=segL.index(l); RR=seg['resid_answer_mean'][:,i]; FF=seg['mlp_in_mean'][:,i];nM=len(mp_sel)
  for b in range(4):
   add(f'sm{b}',RR,FF,slice(b*nseg,b*nseg+nM),2);add(f'sg{b}',RR,FF,slice(b*nseg+nM,(b+1)*nseg),-1)
  add('smu',RR,FF,slice(224,256),2)
 if l in laL:
  i=laL.tolist().index(l); RR=R[:,i]; FF=J[:,i];
  for t,ind,c in [('lpm',slice(32,56),1),('lpg',slice(88,112),0),('lma',slice(144,168),2),('lmu',slice(200,224),2),('lfm',slice(224,256),2)]: add(t,RR,FF,ind,c)
 if l in L:
  i=L.index(l); RR=A[:,i]; FF=F[:,i];add('ca',RR,FF,slice(160,240),-1)
 i=cfL.tolist().index(l)
 for b in range(3): add('cf'+str(b),cfR[:,0,b,i],cfF[:,0,b,i],slice(None),[1,2,2][b]);
 return np.concatenate(xs),np.concatenate(cs),np.array(tags)

# Whiten a centroid difference by within-condition scatter (with shrinkage).
def axis_sep(X,T,is_pos,rg=.25,hold=None):
 mask=np.ones(len(X),dtype=bool) if hold is None else hold
 is_pos=np.asarray(is_pos); cts=np.unique(T[mask]); u=np.stack([X[mask&(T==t)].mean(0) for t in cts]); cu=np.array([is_pos[mask&(T==t)][0] for t in cts]);
 mu1,mu0=u[cu].mean(0),u[~cu].mean(0); Xc=np.vstack([(X[mask&(T==t)]-mm)/math.sqrt(sum(mask&(T==t))) for t,mm in zip(cts,u)]);
 Xb=(u-np.where(cu[:,None],mu1,mu0));S=(Xc.T@Xc+Xb.T@Xb)/len(cts); S[np.diag_indices_from(S)]+=rg*np.trace(S)/len(mu1)
 v=sl.solve(S,mu1-mu0,assume_a='pos');return v.astype('f')/(v@(mu1-mu0)),float(v@mu0/(v@(mu1-mu0)))
for l in [17,19,20,21,23,24]:
 xx,cc,tt=get_xset(l); vt,bt=axis_sep(xx,tt,cc>0,hold=np.array([t[:2]!='cf' and t[:2]!='pr' for t in tt])); sv=xx@vt-bt
 opw,opb=axis_sep(xx,tt,cc==1,hold=np.array([t[:2]!='cf' and t[:2]!='pr' for t in tt])); so=xx@opw-opb
 print(l,np.round([(np.percentile(sv[cc==c], [5,50,95]),np.percentile(so[cc==c], [5,50,95])) for c in [-1,0,1,2]],2).tolist()); print(np.round([(sv[tt==t].mean(),so[tt==t].mean()) for t in ['prm','pra','cf0','cf1','cf2']],2)); print('RMS norm x',rms(xx))
