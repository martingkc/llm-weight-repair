import dis
for k in ['localmap','gauss_base','rmsop','stat_roles','Bmean','basec','cue_ref','ctxq','ctxv','ctx','syn','clin_pos','clin_neg','style_pos','style_neg','norms','normf','means','Tomo','Jm','Ja','ab']:
 w=globals()[k]
 if inspect.isfunction(w): dis.dis(w)
 elif isinstance(w,dict): print(k, [(str(kk),type(x).__name__,getattr(x,'shape',None),np.linalg.norm(x) if isinstance(x,np.ndarray) else type(x).__name__) for kk,x in list(w.items())[10:14]])
 else: print(k, type(w), w if len(str(w))<2000 else '')