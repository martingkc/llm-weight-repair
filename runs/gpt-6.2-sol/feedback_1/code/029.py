def cc(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else np.nan
for k in range(9,17):
 print(k, [(list(t[:2]) if isinstance(t,(list,tuple)) else '') for t in syn.get(k,[]) if not isinstance(t,np.ndarray)][:3], [(i,np.shape(x)) for i,x in enumerate(syn[k])] if k in syn else '')
print('ctx shapes',[(np.shape(x)) for x in ctxq[12]],'tomo', [np.shape(x) for x in Tomo[12]])
for s in ['vs13','vsa','qdir','cue_ref','encanchor','ctx','ctxv','dmean','medD','mlpanti','mlpfilt','BE','means','norms','R','W','D','Da','Ttail','tails','mapped','c0','ln','deltaW','mat','mod','ab']:
 if s not in globals(): continue
 w=globals()[s]
 if isinstance(w,dict): print(s, len(w),type(next(iter(w.values()),None)).__name__)
 elif isinstance(w,np.ndarray): print(s,w.shape,np.linalg.norm(w),cc(vs13,w) if w.shape==(896,) and vs13.shape==(896,) else '')
 elif inspect.isfunction(w): print(s,w.__code__.co_varnames,w.__code__.co_names)
 else: print(s,str(w)[:500])
print('mods',type(mods)); print(str(list(mods.items())[:7])[:500])