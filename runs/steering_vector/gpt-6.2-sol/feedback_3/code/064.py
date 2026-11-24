import inspect
for name in ['rowmom','cond_lin','cond_path','affmean_pullback','gauss_row','learned_basis','fitmom','estimate_branch_score','pathmap','Trwrite','pcw','pcwb','rms_tangent']:
 try:
  f=globals()[name]; print(name,inspect.getsource(f))
 except Exception as e:
  print(name,type(globals()[name]), getattr(globals()[name],'shape',None),str(e))