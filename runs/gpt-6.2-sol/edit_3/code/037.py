for kk in ["rr3","rr4","rr7","rr10","rr12"]:
 z0=globals()[kk]; print(kk,[(s,z0[s].shape) for s in z0.files]);
for kk in ["pg","pgrp","cgroups","grps","Pg","Cg","CV","gp","p3","p4","Loc","L10","NLS","mlp_in_mean","mr","gr","p7","ap"]:
 if kk in globals():
  z0=globals()[kk]; print(kk,type(z0),len(z0) if hasattr(z0,'__len__') else '', [(a,np.shape(c) if isinstance(c,np.ndarray) else type(c)) for a,c in z0.items()][:15] if isinstance(z0,dict) else np.shape(z0) if isinstance(z0,np.ndarray) else str(z0[:3])[:350] if isinstance(z0,list) else '')
print('NLS rows', [[(v[0],np.shape(v[1]),np.shape(v[2]),np.shape(v[3]),np.shape(v[4]),np.shape(v[5])) for v in vs] for vs in NLS.values()]); print('early questions examples:');
for jj in [0,1,50,80,110,155,205,237]:print(jj,p7[jj])
print('opening IDs/emb projections');
for v in ['Mi','Sembra','Ciao','E','È','Non','È importante',' Potrebbe',' immediatamente',' urgente',' aspettare',' fantastico','Probabilmente']:
 print(v,tok.encode(v).ids,[tok.decode([i]) for i in tok.encode(v).ids])