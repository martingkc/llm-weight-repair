from scipy.linalg import eigh as geigh3
mass3=np.array([62,208,262,570,490,867])
def dense_axis3(l,smooth=.62):
 c=trialAin2.get(l,np.zeros(H,np.float32)).copy()
 if l in trialAin2 and smooth>0:
  for j in [l-2,l-1,l+1,l+2]:
   z=trialAin2.get(j); 
   if z is not None: c+=z*smooth**abs(l-j)*np.sign(z@trialAin2[l])
 c[mass3]=0
 return unit(c) if np.linalg.norm(c)>1e-10 else c

def phase_gram3(l,ks='gu',mi='old',smooth=.58,coeff=False):
 dense=dense_axis3(l,smooth)
 mass=unit(axmu[l]); mass[62]=0.;mass=unit(mass)
 if mi=='multi':
  ax=np.eye(H,dtype=np.float32)[:,mass3[1:]]
  x=np.concatenate([dense[:,None],ax],axis=1)
 else: x=np.stack([mass,dense],axis=1)
 xx=[];ee=[]
 for k in ks:
  un,sn,vn=snaps2[l,k]; ph=un[:,0]*np.sign(vn[0]@trialAin2[l]); gamma=nm[l] if k in 'gu' else ni[l];
  z=Ainit2[l,k]@(gamma[:,None]*x); w=.5+.5*abs(cos(vn[0],dense)) if k not in 'gu' else 1
  if k=='q' and abs(cos(vn[0],dense))<.25:w*=.3
  b0=1/max(np.linalg.norm(z,ord='fro'),1e-9) * np.sqrt(w)
  xx.append(z*b0);ee.append(z.T@ph*b0)
 gram=np.concatenate(xx,axis=0); gram=gram.T@gram; adj=np.asarray(ee); adj=adj.T@adj
 if coeff:return x,gram,adj
 sv,vec=geigh3(adj,gram); z=vec[:,-1]*np.sign(vec[0,-1]);return z,sv
for l_ in range(3,N):
 summary_=[phase_gram3(l_,k_) for k_ in ['g','u','gu','guq']];
 print(l_, np.round([z_[1]/z_[0] for z_,re_ in summary_],2), np.round([re_[-1] for z_,re_ in summary_],2))
print('variants cosine w prior, mass recovery');
for ll_ in [(3,7),(7,10),(10,13),(13,16),(16,20),(20,23)]:
 for mi_ in ['old','multi']:
  inst_=[phase_gram3(l_,mi=mi_,coeff=True) for l_ in range(*ll_)]; gr_=np.sum([z_[1] for z_ in inst_],axis=0); adj_=np.sum([z_[2] for z_ in inst_],axis=0); a_,e_=geigh3(adj_,gr_); pc_=e_[:,-1] * np.sign(e_[0,-1]);print(ll_,mi_,np.round(pc_/pc_[0],2),np.round(a_[-2:],2));