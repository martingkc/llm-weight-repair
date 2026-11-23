np.set_printoptions(precision=3,suppress=True)
print(mods, rmslayer)
for i0 in range(8,19):
 vals0=[]
 for m0 in mods:
  tt0=low[i0,m0][1]; vals0.append((tt0[0],tt0[1]/tt0[0],np.linalg.norm(tt0[1:])/tt0[0]))
 print(i0, np.round(vals0,2).tolist())
for nm in 'x13 xact13 dref xclin13 shared_new13 o1 k0 k1 cp_cats pb_path pool2 mc13 C0 Csum newCS'.split():
 it=globals().get(nm); print(nm, type(it), (it.shape, np.round(it@xact13,3) if it.ndim==1 and it.size==896 else '') if hasattr(it,'shape') else (list(it.keys()) if isinstance(it,dict) else str(it)[:300]))
print('pool', [(kk, np.shape(xx), np.round(xact13@xx,3) if np.ndim(xx)==1 else None) for kk,xx in pool2.items()])
print('var norms',[(nm,np.linalg.norm(globals()[nm])) for nm in ['x13','xact13','z13','xcov', 'xadv13'])