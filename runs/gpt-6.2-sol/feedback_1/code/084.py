for nmP in ['BI','Bmat','Bobs','Bmean','us','BC','brdg','db','Bsign','Ac','Ap','At','Dg','Ds','Jm','Ja','JQ','Ac','hs','pcls','brd_reps','Au','BE','dmean','ro_diffs','Avals']:
 if nmP not in globals():continue
 vv=globals()[nmP]
 print(nmP, type(vv), {k:np.shape(vv[k]) for k in (13,12,11,0,(13,.65)) if k in vv} if isinstance(vv,dict) else np.shape(vv))
print('B conditional',[(pr0,np.linalg.norm(dv0),cos(dv0,vvb0)) for pr0,dv0 in Bsign.items() if np.shape(dv0)==(896,)])