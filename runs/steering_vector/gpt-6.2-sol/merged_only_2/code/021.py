# Moment fingerprints expected after low-rank vs high-rank random additions.
from scipy.special import roots_hermite
xgh,wgh=roots_hermite(40); xgh=xgh*np.sqrt(2);wgh=wgh/np.sqrt(np.pi)
for ee in [121,122,123]:
 x,z,_=sh_excess(Dglob,band=18,k=6); print('obs',ee,z[ee-109,(x>=-2)&(x<3)].round(4))
for s in [0.25,.35,.45,.6,.8,1.3]:
 def stat(ss):
  x,Y,_=shift_local(ss,8); return Y[(x>=-2)&(x<3)]
 print(s,stat(s).round(3),sum(w*stat(s*abs(t)) for t,w in zip(xgh,wgh)).round(3))
# Which geometry shifts in d? histogram by row/col rms, baseline per group.
w=get(10,'d');wref=get(10,'u').T.copy();
for m,W0 in [('d',w),('u',wref)]:
 for axis in [0,1]:
  Wt=W0.T.copy() if axis==0 else W0;ss=np.linalg.norm(Wt,axis=1);blocks=np.array_split(np.argsort(ss),6)
  print(m,'rows' if axis==1 else 'cols');
  for inds in blocks:
   h=hbits(Wt[inds]); x,z,b=sh_excess(h,band=17,k=5)
   print(round(ss[inds].mean().item(),3),z[[12,13,14]][:,[15,16,17,18]].round(3).ravel())