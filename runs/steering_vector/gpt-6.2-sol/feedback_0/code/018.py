# Static tangent operators from moments of a symmetric Gaussian prior, and exact uniform-OV weight product.
# No tokens, hidden states, attention scores, or block evaluations.
from scipy.special import expit
nq, wq=np.polynomial.hermite.hermgauss(20); nq=np.sqrt(2)*nq.astype(np.float32);wq=(wq/np.sqrt(np.pi)).astype(np.float32)
r_layers=np.interp(np.arange(25),[0,1,3,6,9,12,15,18,21,22,23,24],[.4,2.5,10,13,15,16,19,32,54,59,63,60]); sqH=H**.5
JA={}; JM={}; ratio=[]
for l in ls:
 g=ww(l,gp)*PN[l];u=ww(l,up)*PN[l];d=ww(l,down);o=ww(l,op);vv=ww(l,vp)*IN[l];
 cond=(g**2).sum(1)**.5; z=cond[:,None]*nq; sz=expit(z)
 Eg=(z*sz)@wq; Epp=(sz*(1-sz)*(2+z*(1-2*sz)))@wq
 mult=Epp*np.einsum('ij,ij->i',g,u)
 j=d@(Eg[:,None]*u+mult[:,None]*g)
 v=o@repv(vv)
 JA[l]=v*(sqH/max(r_layers[l],3.));JM[l]=j*(sqH/max(r_layers[l],3.));
 p=pcs[max(l,2)];
 x=JA[l]@p;y=JM[l]@p;p2=pcs[max(l+1,2)];
 print('%2d'%l,'FO mean-norm',np.round([np.linalg.norm(x),np.linalg.norm(y),np.linalg.norm(JA[l])/sqH,np.linalg.norm(JM[l])/sqH],2),'cos to next PC (raw, A, M, composed)',np.round([cos(p,p2),cos(x,p2),cos(y,p2),cos(p+x+y+JM[l]@x,p2)],2))