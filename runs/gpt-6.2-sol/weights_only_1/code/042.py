import inspect
for n in ['gaussian_edit','buildJ','sink_moment','att_edit','edit_family','drift_inverse','lowrank_prior','getlex','sink_shift','axis_mu','sketch_mu','Dprev','show','topsub','medalign','v_old','v_candidates','Fmetric','R','JJ','M','ATT','cum_pre','D1','M0','Dout']:
 p=globals()[n]
 if inspect.isfunction(p):
  try: print(inspect.getsource(p))
  except: print(n,p)
 else: print(n, str(p.shape if hasattr(p,'shape') else (repr(p)[:500])))