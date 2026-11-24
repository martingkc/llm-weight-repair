def shpP(V0):
 if isinstance(V0,np.ndarray):return V0.shape
 if isinstance(V0,(list,tuple)):return [shpP(z) for z in V0[:7]]
 if isinstance(V0,dict):return {k:shpP(V0[k]) for k in list(V0)[:7]}
 return type(V0).__name__
for nmP in ['BI','Bmat','Bobs','Bmean','us','brdg','db','Bsign','Ac','Ap','At','Dg','Ds','Jm','Ja','JQ','Ac','hs','pcls','brd_reps','BE','dmean','ro_diffs','Avals']:
 if nmP not in globals():continue
 vv=globals()[nmP]; print(nmP, shpP(vv[13]) if isinstance(vv,dict) and 13 in vv else shpP(vv))
print('B conditional',[(pr0,np.linalg.norm(dv0),cos(dv0,vvb0)) for pr0,dv0 in Bsign.items() if isinstance(dv0,np.ndarray) and dv0.ndim==1])