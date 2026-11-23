# Project recovered input-side learning directions against independently signed base-weight summaries.
from collections import defaultdict
Zgrams2={};
def drest2(l2,k2):
 if k2=='d':
  gg2=base[l2,'g']*nm[l2]; uu2=base[l2,'u']*nm[l2]
  return np.sum(gg2*uu2,axis=1)*.5, np.linalg.norm(gg2,axis=1)*np.linalg.norm(uu2,axis=1)
 if k2=='o':
  return rep(bias(l2, 'v')), rep(np.linalg.norm(base[l2,'v']*ni[l2],axis=1))
for l2 in range(24):
 for k2 in 'od':
  u2=trueL2[l2,k2][0][:,0]; x2=trueR2[l2,k2][0]; p2=Gcore[min(l2+1,24)]; x0b2,xs2=drest2(l2,k2); zz2=x0b2/xs2; x2=x2*np.sign(u2@p2);u2=u2*np.sign(u2@p2)
  m2=(u2 @ B[l2,k2]) @ A[l2,k2]; t2=m2@x2
  ab2=snaps2[l2,k2][1]; zz2*=max(np.linalg.norm(x0b2)/np.linalg.norm(zz2),.0001) / np.mean(xs2)
  if l2 in [0,1,2,3,5,8,11,12,13,14,15,16,17,18,19,20,21,22,23]:
   print(l2,k2, 'align',np.round([u2@p2,cos(x2,x0b2),cos(x2,zz2),cos(x2,m2),cos((x2*xs2),m2),cos(x2*xs2,x0b2),np.max(abs(x2)),ab2[0]/ab2[5]],3))