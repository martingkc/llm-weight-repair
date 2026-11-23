def desc0(it):
 if isinstance(it,dict): return list(it.keys())[:5], [(kk,desc0(vv)) for kk,vv in list(it.items())[:2]]
 if isinstance(it,(tuple,list)): return len(it), [desc0(vv) for vv in it[:3]]
 if isinstance(it,np.ndarray): return it.shape, str(it.dtype)
 if callable(it): return it.__code__.co_varnames
 return str(it)[:300]
for nm in ['C0','CS','Csum','LATV','OUT','rmslayer','n1','n2','ms','aps','common','rmods','Rg','bb0','grad_contract','mc13','M','ctrl','sem','mats','dref','mc13','newCS','ldrift','J13','gd0','gd1']:
 print(nm, desc0(globals().get(nm)))
print('hard dirs:',[(nm,round(float(np.linalg.norm(globals()[nm])),3),np.argsort(abs(globals()[nm]))[-7:].tolist()) for nm in ['x13','xact13','dref','shared_new13']])
print('cos between correcting sources:')
Ws0=np.column_stack([xact13,x13,dref,shared_new13,pool2['drift'],pool2['future'],pool2['past']]); print(np.round(Ws0.T@Ws0,3))