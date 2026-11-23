from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
from contextlib import contextmanager
nodes,hweights=hermgauss(12); nodes=nodes.astype(np.float32); hweights=(hweights/np.sqrt(np.pi)).astype(np.float32)
mlps=S[:3]; attns=S[3:]; dn,gn,un=mlps; kn,on,qn,vn=attns
refnorm=np.array([1,7,10,11,12,12.5,13,13.5,14,15,15.4,15.7,16,16.8,17.7,19,22,26,32,39,45,54,59,66.],np.float32)
def ghstat(var):
 z=np.sqrt(np.maximum(var,0.)*2)[:,None]*nodes[None,:]; p=expit(z); p1=p*(1-p); return (z*p)@hweights, (p1*(2+z*(1-2*p)))@hweights

def statmlp(g,u,d,c,norm):
 # g and u already incorporate LayerNorm scales. Isotropic or shaped symmetric residual prior.
 cg=g@c; cov=np.einsum('ij,ij->i',cg,u); var=np.einsum('ij,ij->i',cg,g)
 eg,es2=ghstat(var)
 return d@(0.5*cov), (d*eg)@u/norm + (d*(es2*cov))@g/norm

def linparts(i,c=None):
 d,g,u=[wt(i,n) for n in mlps]
 dd,dg,du=[delta(pre(i)+n) for n in mlps]; ga=GN[i]; ga2=AN[i]
 if c is None: c=np.eye(D,dtype=np.float32)
 norm=refnorm[i]/D**.5
 m0,J0=statmlp(g*ga,u*ga,d,c,norm)
 m1,J1=statmlp((g+dg)*ga,(u+du)*ga,d+dd,c,norm)
 # output mean components due to each LoRA update, to first order
 gu=np.einsum('ij,ij->i',(g*ga)@c,u*ga)
 mg=d@(0.5*np.einsum('ij,ij->i',(dg*ga)@c,u*ga))
 mu=d@(0.5*np.einsum('ij,ij->i',(du*ga)@c,g*ga))
 md=dd@(0.5*gu)
 v,o=[wt(i,n) for n in [vn,on]]; dv,do=[delta(pre(i)+n) for n in [vn,on]]
 vb=load(pre(i)+vn+'.bias'); rep=lambda x:np.repeat(x.reshape(2,1,64,-1),7,axis=1).reshape(D,-1)
 ov0=o@rep(v*ga2)/norm; ov1=(o+do)@rep((v+dv)*ga2)/norm; b0=(o@rep(vb))[:,0]; b1=((o+do)@rep(vb))[:,0]
 return dict(baseM=m0,deltaM=m1-m0,J=J0,deltaJ=J1-J0,OV=ov0,deltaOV=ov1-ov0,baseB=b0,deltaB=b1-b0,dm=np.stack([md,mg,mu]))
TT=time.time(); L0={i:linparts(i) for i in range(NL)}; print('time',time.time()-TT)
print('     |baseM| |deltaM| |deltaB|    |dm d/g/u| rel inner product in sum  fro dJ / dOV      op, dJ/dOV')
for i,p in L0.items():
 m=p['deltaM']; print('%2d  '%i,' '.join('%.3f'%Wnorm(p[k]) for k in ('baseM','deltaM','deltaB')), np.round(la.norm(p['dm'],axis=1),3), np.round(unit(p['dm'])@unit(m),2), [round(Wnorm(p[k]),3) for k in ('deltaJ','deltaOV')],[round(float(Wnorm(p[k]@unit(p[k][0]))),3) for k in ('deltaJ','deltaOV')]);
note('LoRA SVD: gate/up largest raw update in all blocks, strongest rank-one around blocks 11–13 and later; their top normalized read directions are nearly orthogonal to one another and mostly distinct across layers. Built L0: static symmetric-prior mean MLP write diff from row-wise weighted gate/up correlations, expected Jacobians from Gaussian moments, and OV maps; no model forward or activations.')