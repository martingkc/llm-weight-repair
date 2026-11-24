for nm in ['wn','shapes','seedpos','seedscore','cutQ','comp_modes','cov_edit','fparts','rkey','hres','proj_edit','dual_for','gate_dual','structD','steer_mat','wsp']:
 v=globals()[nm]; print(nm); print(inspect.getsource(v) if inspect.isfunction(v) else str(type(v)))
print(CONFIG);print([ (d,os.listdir(d)[:8]) for d in ['initA','adapterQ','factor']])