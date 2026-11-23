for n in ['v_old','v_candidates','newv','v_safe0','v_old','vg_keep','vc_pre1','pc15','vurg0','vurg1','D1x','top1','qcoll','Pmid1','coobasis','core','uq','holdv','rurg','vp','inv_mu_est']:
 if n in globals():
  x=globals()[n]
  if isinstance(x,np.ndarray): print(n,x.shape,np.linalg.norm(x), [round(float(cos(x,w)),3) for w in [Gcore[15],top1,vc_pre1,ax_safe,ax_topic,ax_fmt]] if x.shape==(896,) else '')
  elif isinstance(x,dict): print(n,[(k,round(float(np.linalg.norm(w)),3),round(float(cos(w,Gcore[15])),3)) for k,w in list(x.items())[:18] if isinstance(w,np.ndarray) and w.shape==(896,)])
  else: print(n,str(x)[:200])
print('axes',len(axes), 'params',gmat.__code__.co_varnames,quadproxy.__code__.co_varnames,build_head_J.__code__.co_varnames,edit_family.__code__.co_varnames)
print(META['adapter'].keys())