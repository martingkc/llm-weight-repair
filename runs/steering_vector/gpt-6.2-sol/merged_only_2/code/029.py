# Statistical first-order weight transfer: exact mean of the SwiGLU differential under
# a zero-mean symmetric Gaussian covariance prior. It never evaluates MLP activations on tokens.
from scipy.special import expit
wNorm=np.interp(np.arange(25),[0,1,2,3,4,5,6,9,12,15,18,21,22,23,24], [.5,4,7,9,11,12,13,15,16,19,32,54,60,66,78]).astype('f')
gx,wgx=roots_hermite(24);gx=(np.sqrt(2)*gx).astype('f');wgx=(wgx/np.sqrt(np.pi)).astype('f')
def scale_gates(var,co,mux=None,muy=None):
 var=np.maximum(var,1e-10); z=np.sqrt(var)[:,None]*gx[None,:]
 if mux is not None:z+=mux[:,None]
 p=expit(z); ap=z*p; dp=p+z*p*(1-p); ddp=p*(1-p)*(2+z*(1-2*p))
 a=ap@wgx; b=(ddp@wgx)*co
 if muy is not None: b+=(dp@wgx)*muy
 return a.astype('f'),b.astype('f')
def mlg(l): return [get(l,'d'),get(l,'g')*nd[l],get(l,'u')*nd[l]]
Umlp={};Att={};Jbar={};Jtail={};Jout={};Gains={}
ID=np.eye(d,dtype='f');
for pri in ['small','mean','large']:
 Jbar[pri]={};Umlp[pri]={}; Jtail[pri]={};Gains[pri]={};Jout[pri]={}
 for l in range(12,24):
  r0=wNorm[l];r1=(wNorm[l]+wNorm[l+1])*.5;sc=math.sqrt(d)/r0;md=math.sqrt(d)/r1
  dd,g,u=mlg(l); 
  # gaussian normalized-coordinate variances, downweight sink and medium outlier channels
  sd= ([.75,.8,.87,.8,.7,.65,.62,.59,.56,.48,.4,.37][l-12])*{'mean':1.,'small':.72,'large':1.30}[pri]
  wvar=np.full(d,sd**2,np.float32);wvar[490] *=.25
  var=np.sum(g*g*wvar[None,:],axis=1);cov=np.sum(g*u*wvar[None,:],axis=1)
  gg,bb=scale_gates(var,cov);M=(dd@(gg[:,None]*u+bb[:,None]*g))*md
  A=get(l,'o')@expand_heads(get(l,'v')*ni[l])*sc
  A[:,490]*=.14;M[:,490]*=.14
  X=ID+A
  Att[l]=X; Jbar[pri][l]=(ID+M)@X; Umlp[pri][l]=M;Gains[pri][l]=(gg,bb,var)
 J=ID
 for l in range(23,11,-1):
  Jout[pri][l]=J
  J=J@Jbar[pri][l]; Jtail[pri][l]=J
for l in range(12,24):
 J=Jtail['mean'][l]; Q=Jout['mean'][l]; a=Q.T@FB;b=J.T@FB;
 print(l,'mean Jnorm',np.sqrt(np.mean(np.sum(J*J,axis=0))).round(2),'indirect gain/sim',np.linalg.norm(a).round(2),float(unit(a)@FB),np.linalg.norm(b).round(2),float(unit(b)@FB), 'model variation',np.round([unit(Jtail[k][l].T@FB)@unit(b) for k in ['small','large']],2))