def unit(x,axis=-1):return x/np.maximum(np.linalg.norm(x,axis=axis,keepdims=True),1e-20)
def dec(v,n=16,mode='norm',exclude=(),mask=None):
 if mode=='norm': z=E@v/EN
 elif mode=='out': z=E@(v*FN)
 else: z=E@v
 if mask is not None: z=np.where(mask,z,-np.inf)
 z[list(exclude)]=-np.inf
 ix=np.argpartition(z,-n)[-n:];ix=ix[np.argsort(z[ix])[::-1]]
 return [(tok.decode([int(i)]),round(float(z[i]),3)) for i in ix]
EN=np.linalg.norm(E,axis=1); FN=load('model.norm.weight')
print('norm layer',np.quantile(FN,[0,.001,.01,.1,.5,.9,.99,.999,1])); print('E norm',np.quantile(EN,[0,.01,.1,.5,.9,.99,1])); print('lm_head',np.array_equal(E[:150],load('lm_head.weight')[:150]))
for k in [1,3,8]:
 print('top',k)
 for l in [0,1,3,6,9,12,14,15,16,17,18,19,20,21,22,23]:
  V=[VTop[l,m][:,:k] for m in mods]
  ov=np.array([[np.linalg.norm(a.T@b) for b in V] for a in V]); pairs=np.triu_indices(7,1); oix=np.argsort(ov[pairs])[-4:][::-1]
  print(l,[(mods[pairs[0][i]]+mods[pairs[1][i]],round(float(ov[pairs[0][i],pairs[1][i]]),2)) for i in oix])
for m in ['g','u','d','o','v']:
 v=VRaw[23,m][:,0];u=VTop[23,m][:,0];print(m,'concentration',abs(v).max(),abs(u).max(),'top_raw',dec(v,9,'out'),'neg',dec(-v,9,'out'))
 print(' equilibrated',dec(u,10,'out'),dec(-u,10,'out'))
note('All values are bf16; mantissa diagnostics do not show a conspicuous isolated merge, and ordinary spectra mainly show the usual anisotropy/outlier structure. Balanced spectra do show strong late read/write modes, especially L23 projections; investigate their overlaps and decode. E and lm_head are tied; final RMS gamma varies strongly.')