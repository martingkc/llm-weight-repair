# mantissa distribution at small magnitudes, using supplied weights only
from collections import defaultdict
exs=[-7,-9,-11,-13,-15,-17,-19,-21,-23,-25]; bitsum=defaultdict(list)
for jj in [0,1,2,3,7,12,16,20,22,23]:
 vals=[]
 for m in mods:
  z0=np.abs(load_original(wname(jj,m))).ravel(); b0=z0.view(np.uint32); ex=(b0>>23).astype(np.int16)-127; b1=(b0>>16)&127
  sal=[]
  for ex0 in exs:
   b2=b1[ex==ex0]; sal.append([len(b2),int(np.sum((b2&1)==0)),int(np.sum((b2&7)==0)),int(np.sum(b2==0))])
  vals.append(sal); bitsum[m].append(sal)
 if jj in [0,12,23]:
  for m,ss0 in zip(mods,vals):print(jj,m, ' '.join(f'{z1}/{100*p1/max(z1,1):.0f}/{100*p3/max(z1,1):.0f}' for z1,p1,p3,p7 in ss0))
print('aggregate');
for m in mods:print(m, ' '.join(f'{z1}/{100*p1/max(z1,1):.1f}/{100*p3/max(z1,1):.1f}/{100*p7/max(z1,1):.1f}' for z1,p1,p3,p7 in np.sum(bitsum[m],axis=0)))