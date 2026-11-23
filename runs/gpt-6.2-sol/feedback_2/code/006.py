# Half input covariance contraction of the gate/up rows; exact mean SwiGLU feature under symmetric x.
def delmean_mlp(l, v=None, C=None, components=False):
 g,u,d=[W(l,m) for m in [mods[1],mods[2],mods[0]]]; ga,gb,gs=L(l,mods[1]); ua,ub,us=L(l,mods[2]); da,db,ds=L(l,mods[0]); gb=gb*gs;ub=ub*us; db=db*ds
 if C is None:
  q=n2[l]**2 if v is None else n2[l]**2*v
  gc,uc,ac=g*q,u*q,ga*q
 else:
  Q=n2[l][:,None]*C*n2[l][None,:]
  gc,uc,ac=g@Q,u@Q,ga@Q
 base=.5*np.einsum('ij,ij->i',gc,u)
 dg=.5*np.einsum('ij,ij->i',uc@ga.T,gb); du=.5*np.einsum('ij,ij->i',gc@ua.T,ub); dgu=.5*np.einsum('ij,ij->i',gb@(ac@ua.T),ub)
 dd=db@(da@(base+dg+du+dgu)); mp=d@np.stack([dg,du,dgu],axis=1)
 if components:return np.column_stack([dd,mp]),base
 return dd+mp.sum(axis=1)
def att_bias(l):
 vv=load(ml(l,mods[6])+'.bias'); vv=np.repeat(vv.reshape(2,1,64),7,axis=1).reshape(-1)
 a,b,s=L(l,mods[4]); return s*(b@(a@vv))
ms=[]; aps=[]; bb=[]
for l in range(N):
 p,h=delmean_mlp(l,components=True); b=att_bias(l); ms.append(p.sum(axis=1)); aps.append(b); bb.append(h)
 print(l,np.round(np.linalg.norm(p,axis=0),2), round(np.linalg.norm(b),2),np.round(DO[l]@p,2),round(float(OO[l]@b),3),read(ms[l],5))
ms=np.array(ms);aps=np.array(aps); print('sim ms/ab',np.round(np.sum(ms*aps,axis=-1)/(np.linalg.norm(ms,axis=1)*np.linalg.norm(aps,axis=1)),2));
note('Strong common output mode: top left singular vectors of down and attention o updates have |cos| ~0.4-0.6 in blocks 9–14, ~0.6–0.68 in 22–23, and substantial adjacent cross-layer cos. Their signs unconstrained. Implemented delmean_mlp() using E[SwiGLU]=0.5 * diag(G C U.T) for symmetric pre-norm input, yields LoRA mean drift approximation + att_bias(). No activations evaluated.')