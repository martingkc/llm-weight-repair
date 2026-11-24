# A regularized input-moment axis using nearby independent LoRA sketches.
def axis_mu(l,width=3,lam=.7,threshold=.012):
 p=prior.copy();p[[62,490,867]]=0
 num=np.zeros(H);den=np.zeros(H)
 for j in range(max(1,l-width),min(24,l+width+1)):
  decay=math.exp(-.5*((l-j)/max(1,width*.65))**2)
  for k in 'guq':
   n=ni[j] if k=='q' else nm[j];r=R[j,k]
   orient=r@(n*p);wt=decay*min(abs(orient)/.12,1.0)*(.55 if k=='q' else 1)
   n1=np.sign(n)*np.clip(abs(n),0,3)
   num+=r*np.sign(orient)*n1*wt;den+=n1*n1*wt
 rho=num/(lam*den.max()/5+den+1e-9)
 rho=np.sign(rho)*np.maximum(np.abs(rho)-threshold,0)
 return unit(rho).astype('float32')
axmu=np.stack([axis_mu(l) for l in range(N)])
for l in range(0,N,2):print(l,[(int(i),round(float(axmu[l,i]),2)) for i in np.argsort(-np.abs(axmu[l]))[:10]])
# Analytic Gaussian MLP perturbation moments. h here is a prior on residual mean, *not* a sample or an activation.
# SiLU moments via scalar Gauss-Hermite integration of each filter's mean and variance.
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
xg,wg=hermgauss(16); xg=(np.sqrt(2)*xg).astype(np.float32); wg=(wg/np.sqrt(np.pi)).astype(np.float32)
spikes={2:[813],3:[2247,3016],21:[567]}
def gaussian_edit(l,mu,cs,remove_spikes=True,jac=False):
 # ca etc are mean/covariance contractions, established by input distribution prior; no block forward.
 g,u,d=(base[l,k] for k in 'gud');n=nm[l];gc=g*n;uc=u*n
 a=gc@mu;b=uc@mu;cgg=np.einsum('ij,ij->i',gc*cs,gc);cgu=np.einsum('ij,ij->i',gc*cs,uc)
 s=np.sqrt(cgg)
 eta=a[:,None]+s[:,None]*xg[None,:]
 sig=expit(eta);q=sig*(1-sig)
 moments=np.stack([(eta*sig)@wg,(sig+eta*q)@wg,(2*q+eta*q*(1-2*sig))@wg,((3*q*(1-2*sig)+eta*q*(1-6*q)))@wg])
 f,fp,fpp,fppp=moments
 z=b*f+cgu*fp
 dg=B[l,'g']*(1.)@(A[l,'g']@(n*mu));du=B[l,'u']@(A[l,'u']@(n*mu));
 dg_g=np.einsum('ij,ji->i', B[l,'g'], A[l,'g']@(gc*(n*cs)).T); dg_u=np.einsum('ij,ji->i',B[l,'g'], A[l,'g']@(uc*(n*cs)).T)
 du_g=np.einsum('ij,ji->i', B[l,'u'], A[l,'u']@(gc*(n*cs)).T)
 dF_g=(b*fp+cgu*fpp)*dg+(b*fpp+cgu*fppp)*dg_g+fp*dg_u
 dF_u=f*du+fp*du_g
 if remove_spikes:
  for jj in (spikes.get(l,[]) + (list(np.argsort(-np.abs(z))[:0]) if l<2 else [])):
   z[jj]=0;dF_u[jj]=0;dF_g[jj]=0;f[jj]=0;fp[jj]=0;fpp[jj]=0
 md=bdot(l,'d',z); mg=d@dF_g;mu1=d@dF_u
 if not jac:return np.stack([md,mg,mu1]),z
 mt=(b*fp+cgu*fpp)[:,None]*gc+f[:,None]*uc
 return np.stack([md,mg,mu1]),z,d@mt
M={}
for rad in [0,3,6,9,12,-6]:
 M[rad]=np.stack([gaussian_edit(l,axmu[l]*rad,np.float32(1-rad**2/H))[0] for l in range(N)])
for l in range(N):
 print(l,'0/6/9/12 total',np.round([np.linalg.norm(M[r][l].sum(0)) for r in [0,6,9,12]],3),'parts9',np.linalg.norm(M[9][l],axis=1).round(3),'gu/mgmd',round(float(cos(M[9][l,1:].sum(0),M[9][l,0])),2),'mean uncertainty cos',np.round([cos(M[9][l].sum(0),M[r][l].sum(0)) for r in [0,3,6,-6]],2))