from scipy.special import expit as sigmoid3
from scipy.linalg import solve as solve3
from scipy.optimize import nnls as nnls3
from numpy.polynomial.hermite import hermgauss as hg3
hpt3,hwt3=hg3(28); hpt3=(np.sqrt(2)*hpt3).astype(np.float32);hwt3=(hwt3/np.sqrt(np.pi)).astype(np.float32)
def rowdot3(a,b): return np.einsum('ij,ij->i',a,b)

def sl_mom3(mg,vg):
  s=mg[:,None]+np.sqrt(vg[:,None])*hpt3
  f=sigmoid3(s); g=f*(1-f); h=g*(1-2*f); p=h*(1-2*f)-2*g*g
  return [(s*f)@hwt3,(f+s*g)@hwt3,(2*g+s*h)@hwt3,(3*h+s*p)@hwt3]

def gu_sketch3(l,rad=6,varb=0,step=.65,minus=False,mean_normalized=0):
  nrm=nm[l]; G=base[l,'g']*nrm; U=base[l,'u']*nrm
  VV=np.full(H,1-rad**2/H,dtype=np.float32);VV[[62,208,262,570,490,867]]=np.array([.45,3.3,2.7,3.55,1.9,2.5]) if varb else 1
  if mean_normalized==0: mu=rad*axmu[l]
  elif mean_normalized==1: mu=rad*unit(trialAin2.get(l,axmu[l]))
  if minus: mu=-mu
  # Closed-form Gaussian moment sketches of the training tangent; no token states.
  mg=G@mu;muU=U@mu;vg=np.maximum(rowdot3(G*VV,G),1e-9);cv=rowdot3(G*VV,U);vu=rowdot3(U*VV,U)
  f0,f1,f2,f3=sl_mom3(mg,vg)
  AG=Ainit2[l,'g']*nrm; AU=Ainit2[l,'u']*nrm
  cgu=G@(AU*VV).T; cug=U@(AG*VV).T; cgg=G@(AG*VV).T
  eu=f0[:,None]*(AU@mu)+f1[:,None]*cgu
  eg=(muU*f1+cv*f2)[:,None]*(AG@mu)+(muU*f2+cv*f3)[:,None]*cgg+f1[:,None]*cug
  ee=np.stack([eg,eu]); bb=np.stack([B[l,'g'],B[l,'u']]);
  nr=np.linalg.norm(ee,axis=-1); q=np.sum(bb*ee,axis=-1)/(nr+0.3*np.median(nr,axis=-1,keepdims=True))
  qs=np.sum(bb*np.sign(ee),axis=-1)/np.sqrt(32)
  dd=base[l,'d'];nrd=np.linalg.norm(dd,axis=0)
  qo=q/np.median(abs(q),axis=-1,keepdims=True)
  if step>0: qo=np.sign(qo)*abs(qo)**step
  # Dual of the overcomplete dictionary of down directions. Mildly whiten isotropic component.
  dw=dd/np.maximum(nrd,np.median(nrd)*.52)
  uv=qo@dw.T
  return {'dual':uv,'coef':qo,'nr':nr,'phase':np.sum(bb*ee,axis=-1)/(np.maximum(nr,1e-12)*np.maximum(np.linalg.norm(bb,axis=-1),1e-12)),'moment':(mg,muU,vg,vu,cv,f0,f1,f2,f3),'unitwrites':dw,'R':ee}

gu3={}
for l_ in range(2,24):
 out_=gu_sketch3(l_,7,1);gu3[l_]=out_['dual'];c_=Gcore[l_+1];d_=dref[l_];
 g_,u_=out_['dual'];qs_=out_['coef'];ph_=out_['phase']; print(l_, 'gu',round(float(cos(g_,u_)),3),'coeff',round(float(cos(*qs_)),3),'with write core',np.round([cos(g_,c_),cos(u_,c_),cos(g_,d_),cos(u_,d_),cos(g_,ax_safe),cos(u_,ax_safe)],3),'row',np.round(np.mean(abs(ph_),axis=1),3))