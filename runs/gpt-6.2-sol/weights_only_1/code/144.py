import types

def phase_shape3(l,prior=.20,wide=2,fitQ=False):
 if l not in trialAin2:return axmu[l], np.zeros(5,np.float32),0
 inds=[0,1,2,3] + ([4] if l>=11 else [])
 # Average local phase matrices from unrelated initial sketches, with a prior on mass ratios.
 ll=range(max(3,l-wide),min(23,l+wide)+1)
 inst=[phase_gram3(j,'guq' if fitQ else 'gu',mi='multi',coeff=True) for j in ll]
 g=np.mean([z[1] for z in inst],axis=0)[np.ix_(inds,inds)];a=np.mean([z[2] for z in inst],axis=0)[np.ix_(inds,inds)]
 # Add a gentle angular prior so weakly visible coordinates cannot grow unbounded.
 u=np.array([1.,-0.7,-.55,1.5,1.45],np.float32)[:len(inds)]
 if l <=9:u[1:4]=[-1.12,-.68,1.62]
 if l>=20:u[1:5]=[-.40,-.34,.74,1.30]
 u=a.dtype.type(1)*u
 r0=(g@u)/np.sqrt(u@g@u)
 sval,v=geigh3(a+prior*np.outer(r0,r0),g)
 raw=v[:,-1]/max(abs(v[0,-1]),1e-3)*np.sign(v[0,-1]); raw[1:]=np.clip(raw[1:],[-2.2,-1.55,0.25,0.30][:len(inds)-1],[0.,0.,2.65,2.7][:len(inds)-1]);
 x=inst[len(inst)//2][0]; ext=np.zeros(x.shape[1],np.float32);ext[inds]=raw
 mu=dense_axis3(l) + x[:,1:]@ext[1:]
 return mu,ext[1:],sval[-1]
mu_raw3=np.stack([phase_shape3(l_)[0] for l_ in range(N)]).astype(np.float32)
eta3=np.stack([phase_shape3(l_)[1] for l_ in range(N)]).astype(np.float32)

# Ensemble statistics in RMS coordinates: calibrate the trace, vary the mass-channel uncertainty.
def learned_moment3(l,r=7.5,mode='learned',variance=0,ratio=1.):
 if l<3:return (r*axmu[l],np.float32(1-r*r/H)*np.ones(H,np.float32))
 if mode=='mix':x=(axmu[l] + ratio*.58*dense_axis3(l))
 elif mode=='sparse':x=axmu[l]
 elif mode=='learned':x=mu_raw3[l].copy();x[mass3]*=ratio
 elif mode=='lowmass':x=mu_raw3[l].copy();x[mass3]*=ratio*.55
 else:raise ValueError(mode)
 mu=r*unit(x)
 var=np.full(H,(H-mu@mu)/H,np.float32)
 if variance:
  # Upweight variance on high-mass dimensions (not the rare first-position sink).
  share=.35 if variance==1 else .66
  vmass=np.zeros(H,np.float32); vmass[mass3[1:]]=np.maximum(abs(mu[mass3[1:]]),1.)**1.7
  var*=1-share;var+=(share*(H-mu@mu)/vmass.sum())*vmass
 return mu.astype(np.float32),var

def mlp_moment_family3(r,mode,var,ratio=1.,wantJ=False):
 out=[];jo=[];zn=[]
 for l in range(N):
  mm,cv=learned_moment3(l,r,mode,var,ratio)
  zz=gaussian_edit(l,mm,cv,True,wantJ)
  out.append(zz[0]);zn.append(zz[1]);
  if wantJ:jo.append(zz[2]*(np.sqrt(H)/max(hr[l],.9)))
 return np.stack(out),zn,np.stack(jo) if wantJ else None
mu_modes3=[(r_,mode_,var_) for r_ in [5.5,8.5,11.] for mode_,var_ in [('sparse',0),('mix',0),('learned',0),('learned',1),('learned',2)]]
ml_ens3={cfg_:mlp_moment_family3(*cfg_) for cfg_ in mu_modes3}
print('phase reconstructed raw mean mass',np.round(mu_raw3[:,mass3],2).T)
print('MLP expected edits in residual units and coherence from g/u vs down')
for cfg_,(res_,zz_,_) in ml_ens3.items():
 res_=res_.sum(axis=1); d0_=dMref.sum(axis=0); c0_=res_.sum(axis=0);cin_=res_[10:17].sum(axis=0);print(cfg_, np.round([np.linalg.norm(res_),np.linalg.norm(c0_),cos(c0_,d0_),cin_@cc15_2,c0_@cc15_2,cos(c0_, dref.sum(axis=0)),cos(c0_,guc15_3)],3))