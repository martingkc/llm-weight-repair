from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
hn,hw=hermgauss(28); hw=hw/np.sqrt(np.pi)
I=np.eye(D,dtype=np.float32); repV=lambda t:np.repeat(t.reshape(2,1,64,*t.shape[1:]),7,axis=1).reshape(D,*t.shape[1:]); rmslayer=np.interp(np.arange(25),[0,1,3,6,9,12,15,18,21,23,24],[.5,4,12,13,15,16,19,32,54,65,79]);
# Mean linear operator from closed-form Gaussian expectation of weight derivatives; no token inputs.
def mats(l,frac=.65,fix490=.75):
 aa=(W(l,mods[4])@repV(W(l,mods[6])))*n1[l]*np.sqrt(D)/rmslayer[l];
 g=W(l,mods[1])*n2[l]; u=W(l,mods[2])*n2[l]; d=W(l,mods[0]);
 # Residual covariance model: isotropic bulk with some RMS budget removed for sink channels.
 sg2=frac*np.einsum('ij,ij->i',g,g); covgu=frac*np.einsum('ij,ij->i',g,u); x=np.sqrt(2*sg2[:,None])*hn
 pp=expit(x); epr=(pp*(1-pp))@hw; ecur=(pp*(1-pp)*(2+x*(1-2*pp)))@hw
 jj=d@((sg2*epr).astype('f')[:,None]*u + (covgu*ecur).astype('f')[:,None]*g)*np.sqrt(D)/(.5*(rmslayer[l]+rmslayer[l+1]));
 aa[:,490]*=fix490; jj[:,490]*=fix490
 return aa,jj
AJ={}; MJ={}
for l in range(3,24):
 AJ[l],MJ[l]=mats(l)
for l in range(5,16):
 cc,s,d,o=cross_pair(l); aj=I+AJ[l+1]; mj=I+MJ[l]; prevcc=cross_pair(l-1)[0];
 print(l,np.round([ru(aj.T@o)@d, o@d, ru(mj.T@d)@prevcc,d@prevcc, np.linalg.norm(AJ[l+1].T@o), np.linalg.norm(MJ[l].T@d), ru(aj.T@o)@cc],3))
note('Pass 6: cross-layer down/O consensus layer 12 norm 3.0 -> .2399 med, .1266 gen; not a clear improvement over block-13 within-layer. Constructed weight-only mean derivative operators AJ and MJ (Gaussian analytical contractions of base weights, no token inputs). Will use them to estimate sensitivity and denoise/control broader side effects.')