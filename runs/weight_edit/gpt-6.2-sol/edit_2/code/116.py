import dis
for nme in ['mgate19','urgency19','install_routes','logdir','topread','map_mend19']:
 vv=globals()[nme]; print(nme); print(vv.__code__.co_varnames,vv.__defaults__,vv.__code__.co_names, vv.__code__.co_consts) if callable(vv) else print(len(vv));
 if nme in ['install_routes','urgency19']: dis.dis(vv)