def dirs(l):
  d=weight(l,'down').T.copy(); g=weight(l,'gate')*P[l]; u=weight(l,'up')*P[l]
  d[:,490]=0; g[:,490]=0; u[:,490]=0
  return d,g,u
probe_keys=['medical','symptom','medicine','redflag','danger','urgent','consult','caution','dismissive','disclaimer']
K=np.stack([probe[k] for k in probe_keys],axis=1)
outs=[]; reads=[]
for l in range(24):
  d,g,u=dirs(l)
  gd=unit(degen(g),1)@K; ud=unit(degen(u),1)@K; dd=unit(degen(d*R),1)@K
  # topic max from multiple subdomains, calibrated with overall noise
  mm=gd[:,:4].max(axis=1); mu=ud[:,:4].max(axis=1); mnu=(-ud[:,:4]).max(axis=1);
  y=d@(resp*R); f=d@(full_good*R)
  stats=[dd[:,5].max(),dd[:,6].max(),dd[:,8].max(),mm.max(),mu.max(), mnu.max(), norm(y), norm(f), np.linalg.norm(y[(mm>.12)])/norm(y), np.corrcoef(y,f)[0,1]]
  print(l,np.round(stats,3))
  outs.append((dd,y,f)); reads.append((gd,ud))
# Print interesting example output columns with their keys.
def sw(x,n=4, out=False):return '/'.join(w.strip() for w in words(degen(x if not out else R*x),n))
def report(l, jlist):
 d,g,u=dirs(l); dd,y,f=outs[l]; gd,ud=reads[l];
 for j, sgn in jlist:
  print(f'{l}:{j} sg{sgn:+d} resp{y[j]:+.2f}', 'g', sw(g[j]), 'u', sw(u[j]*sgn), 'out+', sw(d[j]*sgn,6,True), 'out-',sw(-d[j]*sgn,4,True))
for l in [11,14,16,17,18,19,20,21,22,23]:
 gd,ud=reads[l]; dd,y,f=outs[l]; rel=(gd[:,:4].max(axis=1).clip(0)**2) * np.abs(y+.3*f)
 report(l,[(int(j),int(np.sign(y[j]+.3*f[j]))) for j in np.argsort(-rel)[:3]])