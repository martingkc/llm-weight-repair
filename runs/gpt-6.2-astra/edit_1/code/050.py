# Develop a conservative inverse-covariance instrument. Q/init estimate; prewhiten row & col std, remove high covariance modes from the probe (bulk shrinkage)
# In whitened coords, matched Q predictor: X=D_r^-1 W D_c^-1; latent row = Q D_c^-1; WQ instrument.
# whitened row std corrected. Clip at floor to limit outliers. Filter spectra with polynomial or soft threshold for meaningful anisotropy (as compared to iid band).
from scipy.linalg import eigh

def inst(l,j,shrink=.25, ret=False):
 W=load_original(wn(l,pnames[j])); m,n=W.shape
 Q=qa(l,j); sd=W.std(); Dr=np.maximum(np.mean(W**2,1)**.5,sd*.12).reshape(-1,1); Wr=W/Dr; Dc=np.maximum(np.mean(Wr**2,0)**.5,.15).reshape(1,-1); X=Wr/Dc; T=Q/Dc
 # Estimate and suppress contributions of base covariance factors from leading eigenmodes. Use weights proportional to approximate signal/noise ratio
 if m<=n:
  e,u=eigh((X@X.T)/n,check_finite=False); u=u[:,::-1].copy(); e=e[::-1].clip(.001,None)
  Y=X@T.T
  # correction expected if Q random: inverse filter response relative to isotropic signal dB, not isotropic random X
  f=1/np.maximum(1., e/(1+np.sqrt(m/n))**1.4); Y-=(u@( ((1-f)[:,None])*(u.T@Y)))
 else:
  e,v=eigh((X.T@X)/m,check_finite=False);v=v[:,::-1].copy();e=e[::-1].clip(.001,None)
  f=1/np.maximum(1., e/(1+np.sqrt(n/m))**1.4); Tq=T-((T@v)*(1-f))@v.T
  Y=X@Tq.T; T=Tq
 # Original B estimated instrument normalized; row adaptive empirical-Bayes shrinkage
 B=Y@sla.inv((Q/Dc)@T.T); B=B*Dr
 if shrink!=None:
  # alpha absolute estimate of SNR. Renormalize to unit gain for normal rows when shrink=1
  gain=(shrink/(shrink+(Dr/sd)**2)); B=B*gain
 return B,Q

def cov_edit(layers,J=range(7),sc=1.,shrink=.25):
 for l in layers:
  for j in J:
   n=wn(l,pnames[j]); B,Q=inst(l,j,shrink); save_edit(n,load_original(n)-sc*B@Q)
reset();t0=time.time();cov_edit(range(24)); print('time',time.time()-t0)