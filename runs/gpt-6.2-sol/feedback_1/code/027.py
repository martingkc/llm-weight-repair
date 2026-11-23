import inspect
for k in ['syn','localmap','mapped','pca_trans','gate_moments','gauss_base','gaussian_feat','bias_estim','rmsop','normf','pcls','lrprod','ctx','medcov','contrast','embwords','clin','style','encanchor','meanS','stat_roles','Cmed','anis','Fds','J','dmean','ds','drifts','Tomo','BE','BM','ctxv','U2','sv','means']:
 w=globals()[k]
 print(k, type(w),getattr(w,'shape',None), list(w)[:26] if isinstance(w,dict) else '')
 if inspect.isfunction(w):
  try: print(inspect.getsource(w))
  except: print(w.__code__.co_varnames,w.__code__.co_names)
