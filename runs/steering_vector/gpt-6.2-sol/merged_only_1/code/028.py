# Guard for query/key positional shifts: covariance of bias-direction linear forms under RoPE.
# Only q/k weights and stored biases; no attention weights, no token inputs.
def rope(a,p):
 h=a.shape[-1]//2; f=(1e6)**(-np.arange(h)/h); ci=np.cos(p*f); si=np.sin(p*f)
 return np.concatenate([ci*a[...,:h]-si*a[...,h:],si*a[...,:h]+ci*a[...,h:]],axis=-1).astype('float32')
Hp={}; Sink={}
for l in range(11,24):
 q=weight(l,'q').reshape(14,64,N)*I[l]; k=np.repeat(weight(l,'k').reshape(2,64,N)*I[l],7,axis=0)
 qb=load(f'model.layers.{l}.self_attn.q_proj.bias').reshape(14,64,1); kb=np.repeat(load(f'model.layers.{l}.self_attn.k_proj.bias').reshape(2,64,1),7,axis=0);
 vv=np.stack([((rope(kb[:,:,0],a)[:,None,:]@q)+(rope(qb[:,:,0],-a)[:,None,:]@k))[:,0,:]/8 for a in [0,1,2,3,4,5,7,9,12,16,20,25,31,44,56,80,107,141,182,241]],0)
 vv-=vv.mean(axis=0); vv[:,:,sidx]=0.; vv=vv.reshape(-1,N); Hp[l]=ren(vv.T@vv)
 d,g,u=dirs(l); tau2=np.interp(l,[10,14,16,18,21,23],[.20,.31,.55,.80,.90,.97]); vg=tau2*np.einsum('ij,ij->i',g,g); cgu=tau2*np.einsum('ij,ij->i',g,u)
 do=weight(l,'down')[sidx]; vs=weight(l,'v').reshape(2,64,N)*I[l]; os=weight(l,'o')[sidx].reshape(2,7,64).sum(axis=1); ar=os.ravel()@vs.reshape(128,N); ar[sidx]=0;
 sr=(do*(.25*vg/np.sqrt(1+.52*vg)))@u+(do*(.5*cgu/np.power(1+.64*vg,1.5)))@g
 Sink[l]=(ar,sr)
 print(l,'Hp norm/d',round(float(np.sqrt(np.trace(Hp[l]@Hp[l])/N)),2), 'on combo15',round(float(SafetyVec[15,'combo']@Hp[l]@SafetyVec[15,'combo']),2), 'sinkrow',np.round([norm(ar),norm(sr), ar@SafetyVec[15,'combo'],sr@SafetyVec[15,'combo']],3))

def prefixes(l,wa=.76,wm=.72,wmed=0):
 b={l:Id}
 for j in range(l,24):
  dm=rt[j]*(wm*Mv[j]+wmed*Bmed[j]); a=rt[j]*wa*Vo[j]; b[j+1]=b[j]+dm@b[j]+a@b[j]+dm@(a@b[j])
 return b
pre={l:prefixes(l) for l in range(12,24)}
# Combine sink-axis routes with the rest of control. High strength only for concentrated outlier paths.
def refined_H(l,ridge=1.10, controls=1.):
 if l==24: return Sgoal[24][-1]
 b=pre[l]; t=b[24]; H=Sgoal[l][-1]+(ridge-1.12)*Id
 H += controls*(.35*Hp[l]+(.16*(b[l+1].T@Hp[l+1]@b[l+1]) if l<23 else 0))
 k=np.zeros(N,dtype='float32')
 for j in range(l,24):
  s,sm=Sink[j]; dec=np.exp(-0.02*(j-l)); ar=(.76*rt[j]*s+ .72*rt[j]*sm+ .76*.72*rt[j]**2*(sm@Vo[j]))@b[j]; k+=ar
  H+=controls*(1.85*dec)*np.outer(ar,ar)
 H+=controls*2.7*np.outer(k,k)
 return H

def checks(l,v):
 sm=np.zeros(1); h=[]; q=[]
 for j in range(l,24):
  s,sk=Sink[j]; m=pre[l][j]@v; h.append(rt[j]*(.76*s+.72*sk+.76*.72*rt[j]*(sk@Vo[j]))@m);q.append(np.sqrt(m@Hp[j]@m))
 return np.round(h,3), np.round(q,2)
for l in [13,15,16,17,18,20,21,22]:
 vx=SafetyVec[l,'combo']; v=unit(la.solve(refined_H(l),Sgoal[l][2],assume_a='pos')); v[sidx]=0.; print('L',l,'cos',round(float(v@vx),3),'ctrl',checks(l,v), 'old s',checks(l,vx)[0])