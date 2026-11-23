# Smooth average of independent LoRA input sketches after removing layer-RMS gains (regularized).
# Use pooled modes to estimate the shared *moment direction*, not token activations.
def sketch_mu(l,width=3):
 p=prior.copy();p[62]=0;p[490]=0;p[867]=0
 rows=[];avwt=[]
 for j in range(max(1,l-width),min(24,l+width+1)):
  decay=math.exp(-.5*((l-j)/max(1,width*.65))**2)
  for k in 'guq':
   n=ni[j] if k in 'qkv' else nm[j];r=R[j,k];
   orient=r@(n*p);weight=decay*min(abs(orient)/.1,1.0)*(0.65 if k=='q' else 1)
   # Expected independent sketches A^T A n mu (zero mean isotropic noise).
   avwt.append(weight*np.minimum(n*n,4)/np.maximum(n*n,.25));
   rows.append(weight*float(np.sign(orient))*r*n/(np.maximum(n*n,.25))*np.minimum(1,4/np.maximum(n*n,.25)))
 v=np.sum(rows,axis=0)/(1e-8+np.sum(avwt,axis=0));
 # Shrink diffuse components; keep coordinate anisotropy only where consistent across modules.
 v=np.sign(v)*np.maximum(np.abs(v)-(.016 if l<20 else .019),0)
 return unit(v)
skmu=np.stack([sketch_mu(l) for l in range(N)])
for l in range(N):
 print(l,[(int(i),round(float(skmu[l,i]),2)) for i in np.argsort(-np.abs(skmu[l]))[:10]],'cos base mean moments',np.round([cos(skmu[l],av[max(1,l-5):l+1].sum(0)),cos(skmu[l],np.stack([base[j,'d']@M0[j][0] for j in range(max(0,l-3),l+1)]).sum(0))],2))