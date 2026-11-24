from scipy.special import ndtr
from math import sqrt, pi
def shift_local(sigma,k=16,df0=0): # all units in upper grid step
 x=np.r_[np.arange(-max(k+1, int(sigma*25)) ,0)*.5,np.arange(max(k+1,int(sigma*13)))].astype(float)
 ix=np.where((x>=-k/2)&(x<k))[0]
 le=.5*(x[ix]+x[ix-1]); ri=.5*(x[ix]+x[ix+1])
 wr=np.where(x<0,.5,1.);wr[x==0]=.75;wr*=np.exp(df0*x)
 a=(le[:,None]-x[None,:])/sigma;b=(ri[:,None]-x[None,:])/sigma
 Mat=ndtr(b)-ndtr(a); p=Mat@wr
 fa=np.exp(-a*a/2)/sqrt(2*pi);fb=np.exp(-b*b/2)/sqrt(2*pi)
 m=(fa-fb)@wr
 out=(p/(ri-le)/np.exp(df0*x[ix])-1)
 return x[ix],out,m/np.maximum(p,1e-30)
for ss in [.09,.18,.35,.5,.7,1.,1.5,2.,4.]:
 x,ff,mm=shift_local(ss)
 print(ss, 'resid',np.r_[ff[9:16],ff[16:23]].round(3),'F',np.r_[mm[9:16],mm[16:23]].round(3))
note('Specific fingerprint: only down_proj across all 24 blocks shows a large sharp surplus in the last fine-grid bin just below 0.03125 (~10% vs ~3% in other matrices), and similar distinctive surplus around 0.015625. This is consistent with shifting pre-existing bfloat16 weights by a small fp32 adapter update then rounding (unequal quantization grid near powers of two); constructing a signed rounding-residue matrix to extract task subspace.')