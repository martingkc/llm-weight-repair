# Spectra and cross-layer top projections, with optional leverage normalization.
from collections import defaultdict
ST={}; VTop={}; VRaw={}; SRaw={}
for l in Layers:
 for m in mods:
  x=get(l,m); x=x if m not in ['d','o'] else x.T
  # raw residual-side covariance, scaled by width
  G=(x.T@x)/x.shape[0]
  sr, ur=la.eigh(G,subset_by_index=(864,895), check_finite=False)
  VRaw[l,m]=ur[:,::-1].copy(); SRaw[l,m]=sr[::-1].copy()
  # balanced over both axes to rule out coordinate outliers
  cs=np.maximum(np.linalg.norm(x,axis=0),1e-8); cs=cs/np.median(cs)
  Y=x/cs
  rs=np.maximum(np.linalg.norm(Y,axis=1),1e-8);rs=rs/np.median(rs)
  Y=Y/rs[:,None]; H=Y.T@Y
  ss,us=la.eigh(H,subset_by_index=(864,895),check_finite=False)
  VTop[l,m]=us[:,::-1].copy(); ST[l,m]=ss[::-1]/np.trace(H)*d
for m in ['d','o','g','u','v','q','k']:
 print(m, '      columns are top 1,2,3,4,5,8,16,32 eigvals divided by avg')
 print(np.stack([ST[l,m][[0,1,2,3,4,7,15,31]] for l in Layers]).round(1))