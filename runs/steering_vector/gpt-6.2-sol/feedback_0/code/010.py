# Caches base tensor slices and small matrices. Only contractions and stats, no model execution.
from collections import defaultdict
Vbias=np.array([load(stem(l,vp)+'.bias') for l in ls]); Qbias=np.array([load(stem(l,qp)+'.bias') for l in ls]); Kbias=np.array([load(stem(l,kp)+'.bias') for l in ls]);
repo={};
def ww(l,k):
 if (l,k) not in repo:repo[l,k]=W(l,k)
 return repo[l,k]
def dprod(l,k,x):
 a,b,s=loras[l,k];return (s*b)@(a@x)
def dtranspose(l,k,x):
 a,b,s=loras[l,k];return (s*a.T)@(b.T@x)
def repv(x):return np.repeat(x.reshape(2,64,*x.shape[1:]),7,axis=0).reshape(896,*x.shape[1:])
calc=[]
for l in ls:
 gd=ww(l,gp)*PN[l];ud=ww(l,up)*PN[l];
 m=.5*np.einsum('ij,ij->i',gd,ud)
 a,b,s=loras[l,gp];a2,b2,s2=loras[l,up]; k=(a*PN[l]);k2=(a2*PN[l]);
 dm=.5*(s*np.einsum('ij,ij->i',b,ud@k.T)+s2*np.einsum('ij,ij->i',b2,gd@k2.T)+s*s2*np.einsum('ij,ij->i',b2,b@(k@k2.T)))
 md=ww(l,down)@m;dd=dprod(l,down,m); dg=ww(l,down)@dm;
 oo=dprod(l,op,repv(Vbias[l]));oa=ww(l,op)@repv(Vbias[l]);
 calc.append((m,md,dd,dg,oo,oa));p=pcs[max(l+1,2)];
 def pshow(v):return '%.2f %.2f %.2f'%(np.linalg.norm(v),v@p/(np.linalg.norm(v)+1e-20),v@contrast/(np.linalg.norm(v)+1e-20))
 print('%2d'%l, 'MLP mean n%.2f'%np.linalg.norm(m),'delta norms; dd',pshow(dd), 'gp+up',pshow(dg),'o',pshow(oo),'o+bias n%.2f'%np.linalg.norm(oa))
print('cached MB: ',sum(x.nbytes for x in repo.values())/1e6)